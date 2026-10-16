import hashlib
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import test_video_duplicate as fixtures

from video_compat import VideoFileCapabilities, image_voice_only

LIMITATION = (
    "file_id not found in the image/voice cache. "
    "get_file only resolves cached image/voice ids; "
    "for group/normal files use get_group_file_url or get_private_file_url"
)


class SnowLumaVideoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.case = fixtures.VideoDuplicateTests()
        self.case.setUp()
        self.plugin = self.case.plugin
        self.plugin.config["video_debug_log"] = True
        self.digest = hashlib.sha256(b"video").hexdigest()
        self.plugin._download_video = AsyncMock(return_value=(5, self.digest, False))

    async def asyncTearDown(self):
        self.case.tearDown()

    def event(self, action, self_id="one"):
        return types.SimpleNamespace(
            bot=types.SimpleNamespace(call_action=action),
            message_obj=types.SimpleNamespace(self_id=self_id),
        )

    async def analyze(self, event):
        return await self.plugin._analyze_video(
            {
                "source": "https://example.test/video.mp4",
                "source_method": "raw_url",
                "file_reference": "video-resource",
                "size": 5,
            },
            event,
        )

    async def test_explicit_snowluma_skips_api_but_still_hashes_video(self):
        self.plugin.config["video_file_backend"] = "snowluma"
        action = AsyncMock(side_effect=AssertionError("no API should be called"))
        result = await self.analyze(self.event(action))
        action.assert_not_awaited()
        self.assertEqual(result["sha256"], self.digest)
        self.assertEqual(
            result["_debug_retrieval"]["get_file"]["reason"],
            "snowluma_image_voice_only",
        )

    async def test_specific_failure_cached_per_connection_and_account(self):
        class ActionFailed(Exception):
            result = {"status": "failed", "retcode": 100, "wording": LIMITATION}

        action = AsyncMock(side_effect=ActionFailed())
        event = self.event(action)
        with patch("main.logger") as log:
            first = await self.analyze(event)
            second = await self.analyze(event)
            log.warning.assert_not_called()
        self.assertEqual(action.await_count, 1)
        self.assertEqual(first["sha256"], self.digest)
        self.assertEqual(second["sha256"], self.digest)
        self.assertEqual(
            first["_debug_retrieval"]["get_file"]["outcome"], "unsupported"
        )
        self.assertEqual(
            second["_debug_retrieval"]["get_file"]["reason"], "cached_image_voice_only"
        )
        # A shared OneBot transport may route multiple accounts.
        event.message_obj.self_id = "two"
        await self.analyze(event)
        self.assertEqual(action.await_count, 2)
        other = AsyncMock(
            return_value={"data": {"url": "https://example.test/new.mp4"}}
        )
        await self.analyze(self.event(other))
        other.assert_awaited_once_with("get_file", file="video-resource", self_id="one")

    async def test_failed_envelope_is_not_treated_as_a_video_file(self):
        action = AsyncMock(
            return_value={"status": "failed", "retcode": 100, "wording": LIMITATION}
        )
        event = self.event(action)
        first = await self.analyze(event)
        await self.analyze(event)
        self.assertEqual(action.await_count, 1)
        self.assertEqual(
            first["_debug_retrieval"]["get_file"]["outcome"], "unsupported"
        )
        generic = AsyncMock(
            return_value={
                "status": "failed",
                "retcode": 100,
                "wording": "temporary failure",
            }
        )
        event = self.event(generic)
        result = await self.analyze(event)
        await self.analyze(event)
        self.assertEqual(generic.await_count, 2)
        self.assertEqual(result["_debug_retrieval"]["get_file"]["outcome"], "error")

    async def test_raw_local_video_path_is_used_without_api(self):
        path = Path(self.case.temp_dir.name) / "raw.mp4"
        path.write_bytes(b"video")
        action = AsyncMock()
        event = self.event(action)
        event.message_obj.message = [
            fixtures.Video(file="opaque", path="/unreadable/path")
        ]
        event.message_obj.raw_message = {
            "message": [
                {
                    "type": "video",
                    "data": {
                        "file": "opaque",
                        "path": str(path),
                        "url": "https://example.test/stale.mp4",
                    },
                }
            ]
        }
        source = self.plugin._extract_video_components(event)[0]
        self.assertEqual(source["source_method"], "raw_path")
        result = await self.plugin._analyze_video(source, event)
        self.assertEqual(result["sha256"], self.digest)
        action.assert_not_awaited()
        self.plugin._download_video.assert_not_awaited()

    async def test_missing_url_does_not_guess_group_file_api(self):
        self.plugin.config["video_file_backend"] = "snowluma"
        action = AsyncMock()
        result = await self.plugin._analyze_video(
            {"source": "opaque.mp4", "file_reference": "opaque-id"}, self.event(action)
        )
        self.assertEqual(result["analysis"], "unavailable")
        action.assert_not_awaited()
        self.plugin._download_video.assert_not_awaited()

    async def test_cache_expiry_and_generic_cache_miss_not_capability_failure(self):
        self.assertFalse(
            image_voice_only({"retcode": 100, "wording": "image/voice cache miss"})
        )
        cache = VideoFileCapabilities()
        bot = object()
        with patch("video_compat.time.monotonic", return_value=100):
            cache.remember_unsupported(bot, "one")
            self.assertTrue(cache.skip_reason("auto", bot, "one"))
            self.assertIsNone(cache.skip_reason("napcat", bot, "one"))
        with patch("video_compat.time.monotonic", return_value=701):
            self.assertIsNone(cache.skip_reason("auto", bot, "one"))
