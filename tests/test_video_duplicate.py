import asyncio
import gzip
import hashlib
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


class _Logger:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _Filter:
    @staticmethod
    def event_message_type(*_args, **_kwargs):
        return lambda function: function

    @staticmethod
    def command(*_args, **_kwargs):
        return lambda function: function


class _Component:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Plain(_Component):
    def __init__(self, text="", **kwargs):
        super().__init__(text=text, **kwargs)


class _Image(_Component):
    @staticmethod
    def fromFileSystem(path):
        return _Image(path=path, file=path)


class _Video(_Component):
    def __init__(self, file="", **kwargs):
        super().__init__(file=file, **kwargs)


class _Reply(_Component):
    pass


class _Forward(_Component):
    pass


def _install_astrbot_stubs():
    astrbot = types.ModuleType("astrbot")
    astrbot.__path__ = []
    api = types.ModuleType("astrbot.api")
    api.__path__ = []
    api.logger = _Logger()

    event = types.ModuleType("astrbot.api.event")
    event.__path__ = []
    event.filter = _Filter()
    event.AstrMessageEvent = object

    event_filter = types.ModuleType("astrbot.api.event.filter")
    event_filter.EventMessageType = types.SimpleNamespace(GROUP_MESSAGE="group")

    star = types.ModuleType("astrbot.api.star")
    star.Context = object
    star.Star = object
    star.StarTools = types.SimpleNamespace(
        get_data_dir=lambda: Path(tempfile.gettempdir())
    )

    components = types.ModuleType("astrbot.api.message_components")
    components.Plain = _Plain
    components.Image = _Image
    components.Video = _Video
    components.Reply = _Reply
    components.Forward = _Forward

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.event.filter": event_filter,
            "astrbot.api.star": star,
            "astrbot.api.message_components": components,
        }
    )


_install_astrbot_stubs()

from main import (  # noqa: E402
    DEFAULT_VIDEO_FRAME_HASH_THRESHOLD,
    FORWARD_HASH_VERSION,
    MemoryRebootPlugin,
    VIDEO_FRAME_POSITIONS,
)


class VideoDuplicateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.plugin = object.__new__(MemoryRebootPlugin)
        self.plugin.config = {}
        self.plugin.data_dir = self.temp_dir.name
        self.plugin.plugin_dir = self.temp_dir.name
        self.plugin._cache = {}
        self.plugin._video_analysis_semaphore = asyncio.Semaphore(1)
        self.plugin._ffmpeg_path = None
        self.plugin._ffprobe_path = None

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_terminate_does_not_rewrite_clean_cache(self):
        self.plugin._cache["group-1"] = {
            "messages": [{"timestamp": time.time(), "content": "loaded"}],
            "last_load": time.time(),
            "dirty_dates": {},
        }

        with patch.object(
            self.plugin,
            "_write_daily_message_batches",
        ) as write_batches:
            await self.plugin.terminate()

        write_batches.assert_not_called()
        self.assertEqual(self.plugin._cache, {})

    async def test_terminate_writes_only_dirty_date(self):
        current_timestamp = time.time()
        current_date = self.plugin._message_date(
            {"timestamp": current_timestamp}
        )
        old_timestamp = current_timestamp - 86400
        old_date = self.plugin._message_date({"timestamp": old_timestamp})
        current_messages = [
            {"timestamp": current_timestamp, "content": "current"}
        ]
        self.plugin._cache["group-2"] = {
            "messages": [
                {"timestamp": old_timestamp, "content": "old"},
                *current_messages,
            ],
            "last_load": time.time(),
            "dirty_dates": {current_date: 1},
        }

        await self.plugin.terminate()

        current_path = os.path.join(
            self.temp_dir.name,
            "group-2",
            f"{current_date}.json.gz",
        )
        old_path = os.path.join(
            self.temp_dir.name,
            "group-2",
            f"{old_date}.json.gz",
        )
        with gzip.open(current_path, "rt", encoding="utf-8") as file:
            saved_messages = json.load(file)

        self.assertEqual(saved_messages, current_messages)
        self.assertFalse(os.path.exists(old_path))

    async def test_append_offloads_periodic_write_from_event_loop(self):
        timestamp = time.time()
        date_str = self.plugin._message_date({"timestamp": timestamp})
        to_thread_result = {date_str}

        with patch(
            "main.asyncio.to_thread",
            new=AsyncMock(return_value=to_thread_result),
        ) as to_thread:
            await self.plugin._append_message(
                "group-3",
                {"timestamp": timestamp, "content": "first"},
            )

        to_thread.assert_awaited_once()
        self.assertEqual(
            self.plugin._cache["group-3"]["dirty_dates"],
            {},
        )

    async def test_flush_keeps_new_revision_dirty_during_write(self):
        timestamp = time.time()
        date_str = self.plugin._message_date({"timestamp": timestamp})
        cache_entry = {
            "messages": [{"timestamp": timestamp, "content": "first"}],
            "last_load": time.time(),
            "dirty_dates": {date_str: 1},
        }
        self.plugin._cache["group-4"] = cache_entry

        async def finish_after_new_message(_function, *_args):
            cache_entry["dirty_dates"][date_str] = 2
            return {date_str}

        with patch(
            "main.asyncio.to_thread",
            new=AsyncMock(side_effect=finish_after_new_message),
        ):
            await self.plugin._flush_cache("group-4")

        self.assertEqual(cache_entry["dirty_dates"], {date_str: 2})

    @staticmethod
    def _video(
        *,
        sha256=None,
        hashes=None,
        duration_ms=10_000,
        metadata_hash=None,
        analysis="perceptual",
    ):
        return {
            "version": 1,
            "sha256": sha256,
            "frame_hashes": hashes or [],
            "duration_ms": duration_ms,
            "metadata_hash": metadata_hash,
            "analysis": analysis,
        }

    def test_filename_normalization_and_metadata_hash(self):
        name = self.plugin._normalize_video_filename(
            "https://example.test/path/My%20Video.MP4?token=temporary"
        )
        self.assertEqual(name, "my video.mp4")
        self.assertEqual(
            self.plugin._build_video_metadata_hash(name, 123),
            hashlib.sha256(b"my video.mp4\n123").hexdigest(),
        )
        self.assertIsNone(self.plugin._build_video_metadata_hash("", 123))
        self.assertIsNone(
            self.plugin._build_video_metadata_hash("video.mp4", None)
        )

    def test_extracts_raw_onebot_file_size(self):
        component = _Video(
            file="opaque.mp4",
            url="https://stale.example/old-video.mp4",
        )
        raw_url = "https://example.test/video.mp4"
        event = types.SimpleNamespace(
            message_obj=types.SimpleNamespace(
                message=[component],
                raw_message={
                    "message": [
                        {
                            "type": "video",
                            "data": {
                                "file": "content-id.mp4",
                                "url": raw_url,
                                "file_size": "4096",
                            },
                        }
                    ]
                },
            )
        )

        results = self.plugin._extract_video_components(event)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["size"], 4096)
        self.assertEqual(results[0]["name"], "content-id.mp4")
        self.assertEqual(results[0]["source"], raw_url)
        self.assertEqual(results[0]["file_reference"], "content-id.mp4")
        self.assertEqual(results[0]["declared_size"], 4096)
        self.assertEqual(results[0]["source_method"], "raw_url")
        self.assertTrue(results[0]["source_conflict"])
        self.assertEqual(
            results[0]["_debug_source_candidates"],
            {
                "component_path": None,
                "component_url": "https://stale.example/old-video.mp4",
                "component_file": "opaque.mp4",
                "raw_url": raw_url,
                "raw_file": "content-id.mp4",
                "file_reference": "content-id.mp4",
            },
        )

    async def test_local_video_generates_chunked_sha_without_ffmpeg(self):
        video_path = os.path.join(self.temp_dir.name, "sample.mp4")
        payload = b"not-a-real-video-but-hashable"
        with open(video_path, "wb") as file:
            file.write(payload)

        result = await self.plugin._analyze_video(
            {
                "source": video_path,
                "name": "sample.mp4",
                "size": len(payload),
            }
        )

        self.assertEqual(result["analysis"], "sha256")
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result["size"], len(payload))
        self.assertEqual(result["frame_hashes"], [])
        self.assertEqual(result["source_method"], "local")

    async def test_missing_local_video_is_retrieved_with_onebot_file_reference(
        self,
    ):
        video_path = os.path.join(self.temp_dir.name, "retrieved.mp4")
        payload = b"retrieved-video"
        with open(video_path, "wb") as file:
            file.write(payload)
        call_action = AsyncMock(
            return_value={
                "status": "ok",
                "data": {
                    "file": video_path,
                    "url": "/napcat/inaccessible/retrieved.mp4",
                    "file_size": str(len(payload)),
                    "file_name": "retrieved.mp4",
                },
            }
        )
        event = types.SimpleNamespace(
            bot=types.SimpleNamespace(call_action=call_action),
            message_obj=types.SimpleNamespace(self_id="bot-1"),
        )

        result = await self.plugin._analyze_video(
            {
                "source": "/app/.config/QQ/missing/Ori/retrieved.mp4",
                "file_reference": "napcat-context-file-code",
                "name": "retrieved.mp4",
                "size": len(payload),
            },
            event,
        )

        self.assertEqual(result["analysis"], "sha256")
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result["source_method"], "get_file_local")
        call_action.assert_awaited_once_with(
            "get_file",
            file="napcat-context-file-code",
            self_id="bot-1",
        )

    async def test_http_video_prefers_message_bound_get_file_local_source(self):
        video_path = os.path.join(self.temp_dir.name, "current.mp4")
        payload = b"current-message-video"
        declared_size = len(payload) + 100
        with open(video_path, "wb") as file:
            file.write(payload)
        call_action = AsyncMock(
            return_value={
                "data": {
                    "file": video_path,
                    "url": "https://stale.example/old-video.mp4",
                    "file_size": str(len(payload)),
                    "file_name": "current.mp4",
                }
            }
        )
        event = types.SimpleNamespace(
            bot=types.SimpleNamespace(call_action=call_action),
            message_obj=types.SimpleNamespace(),
        )
        self.plugin._download_video = AsyncMock()

        result = await self.plugin._analyze_video(
            {
                "source": "https://example.test/current.mp4",
                "file_reference": "current-message-file-code",
                "name": "current.mp4",
                "size": declared_size,
                "source_method": "raw_url",
                "_debug_source_candidates": {
                    "raw_url": "https://example.test/current.mp4",
                },
            },
            event,
        )

        self.assertEqual(result["analysis"], "sha256")
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result["source_method"], "get_file_local")
        self.assertTrue(result["size_mismatch"])
        self.assertEqual(
            result["size_delta_bytes"],
            len(payload) - declared_size,
        )
        self.assertEqual(
            result["_debug_source_candidates"]["get_file_url"],
            "https://stale.example/old-video.mp4",
        )
        self.assertEqual(
            result["_debug_source_candidates"]["selected_source"],
            video_path,
        )
        self.plugin._download_video.assert_not_awaited()
        call_action.assert_awaited_once_with(
            "get_file",
            file="current-message-file-code",
        )

    async def test_declared_size_mismatch_rejects_matchable_fingerprint(self):
        video_path = os.path.join(self.temp_dir.name, "mismatch.mp4")
        payload = b"different-size"
        with open(video_path, "wb") as file:
            file.write(payload)

        result = await self.plugin._analyze_video(
            {
                "source": video_path,
                "name": "mismatch.mp4",
                "size": len(payload) + 1,
            }
        )

        self.assertEqual(result["analysis"], "unavailable")
        self.assertIsNone(result["sha256"])
        self.assertEqual(result["size"], len(payload))
        self.assertEqual(result["declared_size"], len(payload) + 1)
        self.assertTrue(result["size_mismatch"])
        self.assertEqual(result["size_delta_bytes"], -1)

    async def test_get_file_unreadable_path_keeps_video_unavailable(self):
        call_action = AsyncMock(
            return_value={
                "data": {
                    "file": "/napcat/still-inaccessible/video.mp4",
                    "url": "/napcat/still-inaccessible/video.mp4",
                }
            }
        )
        event = types.SimpleNamespace(
            bot=types.SimpleNamespace(call_action=call_action),
            message_obj=types.SimpleNamespace(),
        )

        result = await self.plugin._analyze_video(
            {
                "source": "/app/.config/QQ/missing/video.mp4",
                "file_reference": "napcat-context-file-code",
                "name": "video.mp4",
                "size": 1024,
            },
            event,
        )

        self.assertEqual(result["analysis"], "unavailable")
        self.assertIsNone(result["sha256"])
        call_action.assert_awaited_once()

    def test_video_embedding_excludes_synthetic_label_and_filename(self):
        content = "[视频] private-name.mp4"

        self.assertEqual(
            self.plugin._embedding_content(
                {
                    "has_video": True,
                    "video_text": "",
                },
                content,
            ),
            "",
        )
        self.assertEqual(
            self.plugin._embedding_content(
                {
                    "has_video": True,
                    "video_text": "用户输入的正文",
                },
                f"用户输入的正文 {content}",
            ),
            "用户输入的正文",
        )

    def test_judge_context_is_time_bounded_compact_and_counts_repeats(self):
        self.plugin.config.update(
            {
                "judge_context_window_seconds": 300,
                "judge_context_max_messages": 4,
            }
        )
        current_timestamp = 10_000
        repeated_text = "我怀疑你在黑乌贼哥哥，而且有证据"
        messages = [
            {
                "id": "matched",
                "sender_id": "old",
                "sender_name": "最早发送者",
                "content": repeated_text,
                "timestamp": current_timestamp - 1200,
            },
            {
                "id": "outside-window",
                "sender_id": "outside",
                "sender_name": "窗口外用户",
                "content": repeated_text,
                "timestamp": current_timestamp - 301,
            },
            {
                "id": "repeat-1",
                "sender_id": "repeat-1",
                "sender_name": "复读者1",
                "content": repeated_text,
                "timestamp": current_timestamp - 240,
            },
            {
                "id": "repeat-2",
                "sender_id": "repeat-2",
                "sender_name": "复读者2",
                "content": repeated_text,
                "timestamp": current_timestamp - 180,
            },
            {
                "id": "other",
                "sender_id": "other",
                "sender_name": "其他用户",
                "content": "中间的其他消息",
                "timestamp": current_timestamp - 60,
            },
            {
                "id": "current",
                "sender_id": "current",
                "sender_name": "当前用户",
                "content": repeated_text,
                "timestamp": current_timestamp,
            },
        ]

        history_ctx, current_ctx, stats = (
            self.plugin._build_judge_contexts(messages, matched_idx=0)
        )

        self.assertLessEqual(len(history_ctx), 4)
        self.assertEqual(len(current_ctx), 3)
        self.assertTrue(
            all(
                message["timestamp"] >= current_timestamp - 300
                for message in current_ctx
            )
        )
        self.assertEqual(stats["messages_in_window"], 4)
        self.assertEqual(stats["exact_repeat_count"], 3)
        self.assertEqual(stats["exact_unique_senders"], 3)
        self.assertEqual(stats["exact_span_seconds"], 240)
        self.assertEqual(stats["longest_consecutive_exact"], 2)

    def test_recent_match_does_not_duplicate_history_context(self):
        self.plugin.config.update(
            {
                "judge_context_window_seconds": 600,
                "judge_context_max_messages": 12,
            }
        )
        messages = [
            {
                "id": "matched",
                "sender_id": "first",
                "content": "相同文本",
                "timestamp": 950,
            },
            {
                "id": "current",
                "sender_id": "second",
                "content": "相同文本",
                "timestamp": 1000,
            },
        ]

        history_ctx, current_ctx, stats = (
            self.plugin._build_judge_contexts(messages, matched_idx=0)
        )

        self.assertEqual(history_ctx, [])
        self.assertEqual(len(current_ctx), 1)
        self.assertEqual(stats["exact_repeat_count"], 2)
        self.assertEqual(stats["history_context_messages"], 0)

    async def test_judge_prompt_contains_compact_repeat_statistics(self):
        response = types.SimpleNamespace(
            completion_text=json.dumps(
                {
                    "should_remind": False,
                    "reason": "近期多人连续复读",
                },
                ensure_ascii=False,
            )
        )
        provider = types.SimpleNamespace(
            text_chat=AsyncMock(return_value=response)
        )
        self.plugin.context = types.SimpleNamespace(
            get_provider_by_id=lambda _provider_id: provider
        )
        self.plugin.config.update(
            {
                "judge_provider_id": "judge-provider",
                "min_unique_senders": 3,
            }
        )
        matched = {
            "sender_name": "最早发送者",
            "content": "相同文本",
            "timestamp": time.time() - 300,
        }
        context_stats = {
            "window_seconds": 600,
            "messages_in_window": 8,
            "exact_repeat_count": 6,
            "exact_unique_senders": 6,
            "exact_first_timestamp": time.time() - 300,
            "exact_last_timestamp": time.time(),
            "exact_span_seconds": 300,
            "longest_consecutive_exact": 4,
        }

        should_remind = await self.plugin._judge_remind(
            "相同文本",
            "当前发送者",
            matched,
            [],
            [],
            context_stats,
            unique_count=6,
        )

        self.assertFalse(should_remind)
        prompt = provider.text_chat.await_args.kwargs["prompt"]
        self.assertIn("待判断文本精确出现：6次", prompt)
        self.assertIn("最长连续相同消息：4条", prompt)
        self.assertIn("与近期上下文重合，已省略", prompt)

    def test_reminder_summary_uses_exact_and_relative_time(self):
        matched = {
            "sender_name": "历史发送者",
            "timestamp": 1234567890,
            "content": "历史消息正文",
        }

        with (
            patch.object(
                self.plugin,
                "_format_time",
                return_value="07-29 12:34:56",
            ),
            patch.object(
                self.plugin,
                "_format_time_ago",
                return_value="5分钟前",
            ),
        ):
            summary = self.plugin._build_message_summary(matched)

        self.assertIn(
            "历史发送者 · 07-29 12:34:56（5分钟前）",
            summary,
        )

    async def test_local_video_promotes_to_perceptual_after_five_frames(self):
        video_path = os.path.join(self.temp_dir.name, "sample.mp4")
        with open(video_path, "wb") as file:
            file.write(b"video")
        hashes = ["0" * 64 for _ in VIDEO_FRAME_POSITIONS]
        self.plugin._probe_video = AsyncMock(
            return_value={
                "duration_ms": 10_000,
                "width": 1920,
                "height": 1080,
            }
        )
        self.plugin._extract_video_frame_hashes = AsyncMock(
            return_value=hashes
        )

        result = await self.plugin._analyze_video(
            {
                "source": video_path,
                "name": "sample.mp4",
                "size": 5,
            }
        )

        self.assertEqual(result["analysis"], "perceptual")
        self.assertEqual(result["frame_hashes"], hashes)
        self.assertEqual(result["duration_ms"], 10_000)
        self.assertEqual(result["width"], 1920)
        self.assertEqual(result["height"], 1080)

    async def test_oversize_declared_video_is_not_downloaded(self):
        self.plugin._download_video = AsyncMock()
        result = await self.plugin._analyze_video(
            {
                "source": "https://example.test/large.mp4",
                "name": "large.mp4",
                "size": 101 * 1024 * 1024,
            }
        )

        self.assertEqual(result["analysis"], "metadata")
        self.assertIsNotNone(result["metadata_hash"])
        self.plugin._download_video.assert_not_awaited()

    def test_frame_match_requires_duration_and_four_strong_frames(self):
        identical = ["0" * 64 for _ in VIDEO_FRAME_POSITIONS]
        one_bad = list(identical)
        one_bad[-1] = ("f" * 32) + ("0" * 32)
        two_bad = list(one_bad)
        two_bad[-2] = ("f" * 32) + ("0" * 32)

        current = self._video(hashes=identical)
        self.assertTrue(
            self.plugin._video_frames_match(
                current,
                self._video(hashes=one_bad),
            )
        )
        self.assertFalse(
            self.plugin._video_frames_match(
                current,
                self._video(hashes=two_bad),
            )
        )
        self.assertFalse(
            self.plugin._video_frames_match(
                current,
                self._video(hashes=identical, duration_ms=12_000),
            )
        )
        self.assertEqual(
            self.plugin._get_video_frame_threshold(),
            DEFAULT_VIDEO_FRAME_HASH_THRESHOLD,
        )

    def test_file_sha_has_priority_over_earlier_frame_match(self):
        hashes = ["0" * 64 for _ in VIDEO_FRAME_POSITIONS]
        current = self._video(sha256="exact", hashes=hashes)
        messages = [
            {
                "sender_id": "frame",
                "videos": [self._video(sha256="other", hashes=hashes)],
            },
            {
                "sender_id": "exact",
                "videos": [self._video(sha256="exact")],
            },
        ]

        matched, index, match_type = self.plugin._find_video_match(
            messages,
            [current],
        )

        self.assertIs(matched, messages[1])
        self.assertEqual(index, 1)
        self.assertEqual(match_type, "video_sha256")

    def test_metadata_match_only_applies_to_oversize_current_video(self):
        metadata_hash = hashlib.sha256(b"large.mp4\n200").hexdigest()
        stored = self._video(
            metadata_hash=metadata_hash,
            analysis="metadata",
        )
        current_small = self._video(
            metadata_hash=metadata_hash,
            analysis="sha256",
        )
        current_large = self._video(
            metadata_hash=metadata_hash,
            analysis="metadata",
        )

        self.assertIsNone(
            self.plugin._video_match_type([current_small], [stored])
        )
        self.assertEqual(
            self.plugin._video_match_type([current_large], [stored]),
            "video_metadata",
        )

    def test_forward_video_key_ignores_temporary_url_token(self):
        first = {
            "file": "https://cdn.test/video/content.mp4?token=first",
            "file_size": 2048,
        }
        second = {
            "file": "https://cdn.test/video/content.mp4?token=second",
            "file_size": 2048,
        }
        changed_size = dict(second, file_size=4096)

        first_key = self.plugin._stable_forward_media_key(
            first,
            media_type="video",
        )
        second_key = self.plugin._stable_forward_media_key(
            second,
            media_type="video",
        )

        self.assertEqual(first_key, second_key)
        self.assertNotEqual(
            first_key,
            self.plugin._stable_forward_media_key(
                changed_size,
                media_type="video",
            ),
        )
        self.assertEqual(FORWARD_HASH_VERSION, 4)

    def test_video_debug_log_is_default_off(self):
        event = types.SimpleNamespace(
            message_obj=types.SimpleNamespace(message_id="message-1")
        )
        with patch("main.logger") as test_logger:
            self.plugin._log_video_debug(
                event,
                "group-1",
                [],
                None,
                None,
            )
        test_logger.info.assert_not_called()

    def test_video_debug_log_outputs_source_diagnostics(self):
        self.plugin.config["video_debug_log"] = "true"
        event = types.SimpleNamespace(
            message_obj=types.SimpleNamespace(message_id="message-2")
        )
        video = {
            "version": 1,
            "name": "private-name.mp4",
            "source": "https://private.example/video.mp4",
            "analysis": "perceptual",
            "size": 1234,
            "declared_size": 1234,
            "sha256": "a" * 64,
            "frame_hashes": ["b" * 64 for _ in VIDEO_FRAME_POSITIONS],
            "duration_ms": 5000,
            "width": 1280,
            "height": 720,
            "metadata_hash": "c" * 64,
            "source_method": "get_file_local",
            "source_conflict": True,
            "size_mismatch": True,
            "size_delta_bytes": -100,
            "size_ratio": 0.925,
            "_debug_source_url": "https://private.example/video.mp4",
            "_debug_source_candidates": {
                "raw_url": "https://raw.example/video.mp4",
                "get_file_file": "/shared/current.mp4",
                "get_file_url": "https://private.example/video.mp4",
                "selected_source": "/shared/current.mp4",
            },
        }
        matched = {"id": "stored-record"}

        with patch("main.logger") as test_logger:
            self.plugin._log_video_debug(
                event,
                "group-2",
                [video],
                "video_frame_hash",
                matched,
            )

        lines = [
            call.args[0]
            for call in test_logger.info.call_args_list
        ]
        part_lines = [line for line in lines if " data=" in line]
        serialized = "".join(
            line.split(" data=", 1)[1]
            for line in part_lines
        )
        payload = json.loads(serialized)

        self.assertTrue(lines[-1].endswith("END"))
        self.assertEqual(payload["message_id"], "message-2")
        self.assertEqual(payload["match_type"], "video_frame_hash")
        self.assertEqual(payload["matched_record_id"], "stored-record")
        self.assertEqual(
            payload["fingerprints"][0]["frame_hashes"],
            video["frame_hashes"],
        )
        self.assertEqual(
            payload["fingerprints"][0]["source_method"],
            "get_file_local",
        )
        self.assertTrue(
            payload["fingerprints"][0]["source_conflict"]
        )
        self.assertEqual(
            payload["fingerprints"][0]["declared_size"],
            1234,
        )
        self.assertEqual(
            payload["fingerprints"][0]["source_url"],
            "https://private.example/video.mp4",
        )
        self.assertEqual(
            payload["fingerprints"][0]["source_candidates"],
            video["_debug_source_candidates"],
        )
        self.assertTrue(
            payload["fingerprints"][0]["size_mismatch"]
        )
        self.assertIn("包含来源URL、路径和资源标识", lines[0])
        self.assertNotIn("private-name.mp4", serialized)
        self.assertIn("private.example", serialized)

    def test_image_description_search_does_not_require_cached_files(self):
        messages = [
            {
                "content": "[图片内容: 一只橙色的猫趴在窗台]",
                "timestamp": 300,
            },
            {
                "image_description": "橙色 猫 正在晒太阳",
                "timestamp": 100,
            },
        ]

        matches = self.plugin._find_images_by_description(
            messages,
            "橙色 猫",
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0]["image_description"], "橙色 猫 正在晒太阳")
        self.assertEqual(matches[1]["image_description"], "一只橙色的猫趴在窗台")

    def test_image_description_prefers_newest_when_scores_are_equal(self):
        matches = self.plugin._find_images_by_description(
            [
                {
                    "image_description": "Night SKY",
                    "timestamp": 100,
                },
                {
                    "image_description": "night   sky",
                    "timestamp": 200,
                },
            ],
            "NIGHT sky",
        )

        self.assertEqual(matches[0]["timestamp"], 200)

    def test_image_description_fuzzy_search_tolerates_inserted_character(self):
        matches = self.plugin._find_images_by_description(
            [
                {
                    "image_description": "窗台上有一只橙色的猫",
                    "timestamp": 100,
                },
            ],
            "橙色猫",
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["_image_search_match_type"], "fuzzy")
        self.assertGreaterEqual(
            matches[0]["_image_search_fuzzy_similarity"],
            0.65,
        )

    def test_image_description_semantic_search_uses_existing_embedding(self):
        matches = self.plugin._find_images_by_description(
            [
                {
                    "image_description": "一只猫趴在沙发上",
                    "embedding": [1.0, 0.0],
                    "timestamp": 100,
                },
                {
                    "image_description": "一架飞机正在降落",
                    "embedding": [0.0, 1.0],
                    "timestamp": 200,
                },
            ],
            "宠物",
            [1.0, 0.0],
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["image_description"], "一只猫趴在沙发上")
        self.assertEqual(matches[0]["_image_search_match_type"], "semantic")

    def test_image_description_exact_match_outranks_semantic_match(self):
        matches = self.plugin._find_images_by_description(
            [
                {
                    "image_description": "宠物用品清单",
                    "embedding": [0.0, 1.0],
                    "timestamp": 100,
                },
                {
                    "image_description": "一只猫趴在沙发上",
                    "embedding": [1.0, 0.0],
                    "timestamp": 200,
                },
            ],
            "宠物",
            [1.0, 0.0],
        )

        self.assertEqual(matches[0]["image_description"], "宠物用品清单")
        self.assertEqual(matches[0]["_image_search_match_type"], "phrase")

    async def test_local_image_hash_does_not_create_persistent_cache(self):
        image_path = os.path.join(self.plugin.plugin_dir, "source.jpg")
        Path(image_path).write_bytes(b"image")

        with patch.object(
            self.plugin,
            "_compute_image_hash",
            return_value="image-hash",
        ) as compute_hash:
            result = await self.plugin._compute_image_source_hash(image_path)

        self.assertEqual(result, "image-hash")
        compute_hash.assert_called_once_with(image_path)
        self.assertFalse(
            os.path.exists(os.path.join(self.plugin.plugin_dir, "image_cache"))
        )

    async def test_remote_image_hash_deletes_temporary_file(self):
        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            @staticmethod
            async def read():
                return b"image"

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            @staticmethod
            def get(_source):
                return Response()

        hashed_paths = []

        def compute_hash(path):
            self.assertTrue(os.path.isfile(path))
            hashed_paths.append(path)
            return "remote-image-hash"

        with (
            patch(
                "main.aiohttp",
                new=types.SimpleNamespace(
                    ClientSession=lambda: Session(),
                ),
            ),
            patch.object(
                self.plugin,
                "_compute_image_hash",
                side_effect=compute_hash,
            ),
        ):
            result = await self.plugin._compute_image_source_hash(
                "https://example.test/image.jpg"
            )

        self.assertEqual(result, "remote-image-hash")
        self.assertEqual(len(hashed_paths), 1)
        self.assertFalse(os.path.exists(hashed_paths[0]))

    async def test_search_image_command_replies_without_resending_image(self):
        group_id = "group-3"
        message = {
            "message_id": "platform-message-1",
            "sender_name": "Alice",
            "image_description": "橙色 猫 正在窗台晒太阳",
            "timestamp": 100,
        }

        class SearchEvent:
            message_str = "/搜图 橙色 猫"
            message_obj = types.SimpleNamespace(self_id="bot-1")
            bot = types.SimpleNamespace(
                call_action=AsyncMock(
                    return_value={
                        "status": "ok",
                        "data": {"message_id": "platform-message-1"},
                    }
                )
            )

            @staticmethod
            def get_group_id():
                return group_id

            @staticmethod
            def plain_result(text):
                return ("plain", text)

            @staticmethod
            def chain_result(chain):
                return chain

        with (
            patch.object(
                self.plugin,
                "_load_messages",
                return_value=[message],
            ),
            patch.object(
                self.plugin,
                "_get_embedding",
                new=AsyncMock(return_value=[1.0, 0.0]),
            ) as get_embedding,
        ):
            results = [
                result
                async for result in self.plugin.search_image(
                    SearchEvent(),
                    "橙色",
                )
            ]

        chain = results[0]
        self.assertEqual(chain[0].id, "platform-message-1")
        self.assertIsInstance(chain[1], _Plain)
        self.assertIn("找到1张", chain[1].text)
        self.assertIn("橙色 猫 正在窗台晒太阳", chain[1].text)
        self.assertFalse(any(isinstance(item, _Image) for item in chain))
        get_embedding.assert_not_awaited()
        SearchEvent.bot.call_action.assert_awaited_once_with(
            "get_msg",
            message_id="platform-message-1",
            self_id="bot-1",
        )

    async def test_search_image_command_embeds_semantic_fallback_once(self):
        group_id = "group-semantic"
        message = {
            "message_id": "",
            "sender_name": "Carol",
            "image_description": "一只猫趴在沙发上",
            "embedding": [1.0, 0.0],
            "timestamp": 300,
        }

        class SearchEvent:
            message_str = "/搜图 宠物"

            @staticmethod
            def get_group_id():
                return group_id

            @staticmethod
            def plain_result(text):
                return ("plain", text)

            @staticmethod
            def chain_result(chain):
                return chain

        with (
            patch.object(
                self.plugin,
                "_load_messages",
                return_value=[message],
            ),
            patch.object(
                self.plugin,
                "_get_embedding",
                new=AsyncMock(return_value=[1.0, 0.0]),
            ) as get_embedding,
        ):
            results = [
                result
                async for result in self.plugin.search_image(
                    SearchEvent(),
                    "宠物",
                )
            ]

        get_embedding.assert_awaited_once_with("宠物")
        self.assertIn("一只猫趴在沙发上", results[0][1])

    async def test_search_image_command_explains_unavailable_reply(self):
        group_id = "group-4"
        message = {
            "message_id": "404",
            "sender_name": "Bob",
            "image_description": "蓝色天空和白云",
            "timestamp": 200,
        }

        class SearchEvent:
            message_str = "/搜图 天空"
            message_obj = types.SimpleNamespace(self_id="bot-1")
            bot = types.SimpleNamespace(
                call_action=AsyncMock(
                    return_value={"status": "failed", "data": None}
                )
            )

            @staticmethod
            def get_group_id():
                return group_id

            @staticmethod
            def plain_result(text):
                return ("plain", text)

            @staticmethod
            def chain_result(chain):
                return chain

        with patch.object(
            self.plugin,
            "_load_messages",
            return_value=[message],
        ):
            results = [
                result
                async for result in self.plugin.search_image(SearchEvent(), "天空")
            ]

        self.assertEqual(results[0][0], "plain")
        self.assertIn("原消息已无法引用", results[0][1])
        self.assertIn("蓝色天空和白云", results[0][1])
        self.assertIn("Bob", results[0][1])


if __name__ == "__main__":
    unittest.main()
