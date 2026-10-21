import asyncio
import copy
import json
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from aiohttp import web
from PIL import Image

# isort: off
import test_video_duplicate  # noqa: F401 - installs isolated AstrBot API stubs
from main import MemoryRebootPlugin, Plain, Image as MessageImage
from memory_embedding.provider import LiteRTProvider, EmbeddingError
from memory_embedding.manager import EmbeddingManager
from memory_embedding.matching import compare, candidates, search_images
# isort: on


def features(text="", images=(), vector=None, space="test", complete=True):
    return {
        "schema": 1,
        "space": space,
        "text": text,
        "text_vector": vector,
        "images": [
            {"sha256": str(i), "vector": v, "dhash": "0" * 64} for i, v in images
        ],
        "image_count": len(images),
        "complete": complete,
        "fusion_vector": None,
    }


def message(f, sender="a", timestamp=None):
    return {
        "multimodal": f,
        "sender_id": sender,
        "sender_name": sender,
        "content": f["text"] or "[images]",
        "has_image": bool(f["images"]),
        "message_id": sender,
        "timestamp": timestamp or time.time() - 7200,
    }


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.statuses = []
        self.body = {"data": [{"embedding": [3.0, 4.0], "index": 0}]}

        async def endpoint(request):
            self.requests.append(await request.json())
            status = self.statuses.pop(0) if self.statuses else 200
            return web.json_response(self.body, status=status)

        app = web.Application()
        app.router.add_post("/v1/embeddings", endpoint)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.config = {
            "embedding_api_url": f"http://127.0.0.1:{port}/v1",
            "embedding_dimensions": 2,
            "embedding_max_retries": 1,
        }
        self.provider = LiteRTProvider(self.config)

    async def asyncTearDown(self):
        await self.provider.close()
        await self.runner.cleanup()

    async def test_text_image_and_fusion_wire_formats(self):
        self.assertEqual(await self.provider.embed_text(" hello "), [0.6, 0.8])
        await self.provider.embed_image("data:image/png;base64,AAAA")
        await self.provider.embed_multimodal("caption", ["data:image/png;base64,AAAA"])
        self.assertEqual(
            self.requests[0]["input"], "task: sentence similarity | text: hello"
        )
        self.assertEqual(self.requests[1]["input"][0]["type"], "image_url")
        self.assertEqual(
            [v["type"] for v in self.requests[2]["input"]], ["text", "image_url"]
        )
        self.assertEqual(self.requests[2]["dimensions"], 2)

    async def test_query_role_cache_and_space(self):
        a = self.provider.space
        await self.provider.embed_text("cat")
        await self.provider.embed_text("cat")
        await self.provider.embed_text("cat", task="search")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(a, self.provider.space)
        other = LiteRTProvider({**self.config, "embedding_model_revision": "v2"})
        self.assertNotEqual(other.space, a)
        await other.close()

    async def test_retries_rate_limit_not_auth(self):
        self.statuses = [429, 200]
        await self.provider.embed_text("one")
        self.assertEqual(len(self.requests), 2)
        self.statuses = [401, 200]
        with self.assertRaisesRegex(EmbeddingError, "401"):
            await self.provider.embed_text("two")
        self.assertEqual(len(self.requests), 3)

    async def test_invalid_vectors_and_joint_response(self):
        for value in ([], [0, 0], [True, 1], [float("nan"), 1], [float("inf"), 1], [1]):
            self.body = {"data": [{"embedding": value}]}
            with self.subTest(value=value), self.assertRaises(EmbeddingError):
                await self.provider.embed_text(str(value))
        self.body = {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]}
        with self.assertRaisesRegex(EmbeddingError, "one_joint"):
            await self.provider.embed_text("batch is not fusion")

    async def test_timeout_is_sanitized(self):
        async def slow(request):
            await asyncio.sleep(2)
            return web.json_response(self.body)

        app = web.Application()
        app.router.add_post("/v1/embeddings", slow)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        provider = LiteRTProvider(
            {
                **self.config,
                "embedding_api_url": f"http://127.0.0.1:{port}/v1",
                "embedding_timeout": 1,
                "embedding_max_retries": 0,
            }
        )
        try:
            with self.assertRaisesRegex(EmbeddingError, "^embedding_network_error$"):
                await provider.embed_text("private input")
        finally:
            await provider.close()
            await runner.cleanup()

    async def test_unsupported_multi_image_and_long_text(self):
        with self.assertRaisesRegex(EmbeddingError, "fusion_image_limit"):
            await self.provider.embed_multimodal("", ["a"] * 5)
        with self.assertRaisesRegex(EmbeddingError, "budget"):
            await self.provider.embed_text("x" * 901)
        self.assertFalse(self.requests)


class MatchingTests(unittest.TestCase):
    def test_recompressed_image_can_match_semantically(self):
        a = message(features(images=[("original", [1, 0])]))
        b = message(features(images=[("recompressed", [0.999, 0.01])]))
        self.assertEqual(compare(a, b, {})["kind"], "semantic")

    def test_dhash_collision_does_not_confirm(self):
        a = message(features(images=[("one", [1, 0])]))
        b = message(features(images=[("two", [0, 1])]))
        self.assertIsNone(compare(a, b, {}))

    def test_all_images_require_one_to_one_coverage(self):
        a = message(features(images=[("a", [1, 0]), ("b", [0, 1])]))
        b = message(features(images=[("x", [1, 0]), ("y", [1, 0])]))
        self.assertIsNone(compare(a, b, {}))
        c = message(features(images=[("x", [0, 1]), ("y", [1, 0])]))
        self.assertIsNotNone(compare(a, c, {}))

    def test_changed_caption_and_number_conflicts(self):
        a = message(features("sale 2026 price 100", [("a", [1, 0])], [1, 0]))
        b = message(features("sale 2026 price 200", [("a", [1, 0])], [1, 0]))
        self.assertIsNone(compare(a, b, {}))
        a["multimodal"]["text"] = "a cat"
        b["multimodal"]["text"] = "database upgrade"
        b["multimodal"]["text_vector"] = [0, 1]
        self.assertIsNone(compare(a, b, {}))

    def test_missing_image_and_extra_caption_are_not_ignored(self):
        a = message(features(images=[("a", [1, 0])], complete=False))
        b = message(features(images=[("a", [1, 0])]))
        self.assertIsNone(compare(a, b, {}))
        a["multimodal"]["complete"] = True
        a["multimodal"]["text"] = "new caption"
        self.assertIsNone(compare(a, b, {}))

    def test_model_spaces_and_legacy_are_isolated(self):
        a = message(features("cat", vector=[1, 0]))
        b = message(features("kitten", vector=[1, 0], space="different"))
        self.assertIsNone(compare(a, b, {}))
        self.assertIsNone(compare(a, {"embedding": [1, 0]}, {}))
        b["multimodal"]["text"] = "cat"
        self.assertEqual(compare(a, b, {})["kind"], "exact")

    def test_cross_modal_and_search_share_image_vector(self):
        image = message(features(images=[("a", [1, 0])]))
        text = message(features("a cat", vector=[1, 0]))
        self.assertIsNotNone(compare(image, text, {}))
        found = search_images([image], [1, 0], "test", {})
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["_image_index"], 0)
        self.assertFalse(search_images([image], [1, 0], "other", {}))

    def test_expired_records_not_recalled(self):
        old = message(
            features("same", vector=[1, 0]), timestamp=time.time() - 9 * 86400
        )
        now = message(features("same", vector=[1, 0]))
        self.assertFalse(candidates([old], now, {}))

    def test_offline_exact_text_and_image(self):
        a = message(features("same", [("a", None)]))
        b = copy.deepcopy(a)
        self.assertEqual(compare(a, b, {})["kind"], "exact")


class ImageAndManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        Image.new("RGB", (32, 32), "red").save(self.image)
        self.manager = EmbeddingManager(
            {"embedding_api_url": "http://127.0.0.1:1/v1", "embedding_dimensions": 2}
        )
        self.manager.provider.embed_text = AsyncMock(return_value=[1, 0])
        self.manager.provider.embed_image = AsyncMock(return_value=[1, 0])
        self.manager.provider.embed_multimodal = AsyncMock(return_value=[1, 0])

    async def asyncTearDown(self):
        await self.manager.close()
        self.temp.cleanup()

    async def test_multi_image_no_vision_no_persistent_copy(self):
        before = list(self.root.iterdir())
        result = await self.manager.features(
            "caption", [str(self.image), str(self.image)]
        )
        self.assertEqual(len(result["images"]), 2)
        self.assertTrue(result["complete"])
        self.manager.provider.embed_multimodal.assert_awaited_once()
        self.assertNotIn("data_uri", json.dumps(result))
        self.assertEqual(before, list(self.root.iterdir()))

    async def test_failed_image_preserves_partial_search_features(self):
        result = await self.manager.features(
            "", [str(self.image), str(self.root / "missing")]
        )
        self.assertFalse(result["complete"])
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(len(result["images"]), 1)

    async def test_excess_images_mark_incomplete(self):
        self.manager.config["multimodal_max_images"] = 1
        result = await self.manager.features("", [str(self.image)] * 2)
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["images"]), 1)

    async def test_offline_retains_exact_fingerprints(self):
        self.manager.provider.embed_image.side_effect = EmbeddingError("offline")
        result = await self.manager.features("", [str(self.image)])
        self.assertTrue(result["complete"])
        self.assertTrue(result["images"][0]["sha256"])
        self.assertIsNone(result["images"][0]["vector"])

    async def test_ssrf_and_untrusted_paths_rejected(self):
        for source in [
            "http://127.0.0.1/image",
            "http://169.254.169.254/metadata",
            "file://remote/share/image",
            "https://user:pass@example.com/x",
        ]:
            with self.subTest(source=source), self.assertRaises(EmbeddingError):
                await self.manager.images.load(source)
        with self.assertRaises(EmbeddingError):
            await self.manager.images.load(str(Path(__file__).resolve()))

    async def test_optional_sticker_unknown_not_dropped(self):
        result = await self.manager.features(
            "", [str(self.image)], AsyncMock(return_value=None)
        )
        self.assertEqual(len(result["images"]), 1)
        result = await self.manager.features(
            "", [str(self.image)], AsyncMock(return_value=False)
        )
        self.assertEqual(result["image_count"], 0)

    async def test_vision_description_is_attached_to_its_image(self):
        vision = AsyncMock(side_effect=["first description", "second description"])
        result = await self.manager.features(
            "", [str(self.image), str(self.image)], vision
        )
        self.assertEqual(
            result["descriptions"], ["first description", "second description"]
        )
        self.assertEqual(
            [image["description"] for image in result["images"]],
            ["first description", "second description"],
        )
        self.assertEqual(vision.await_count, 2)

    async def test_vision_classification_reaches_diagnostics(self):
        self.manager.config["multimodal_debug_log"] = True

        async def vision(_uri, info):
            info.update(
                sticker=True,
                raw_is_sticker="true",
                json_ok=True,
                sticker_source="is_sticker",
                content_chars=12,
                response_chars=40,
                response_preview='{"is_sticker": true}',
                content_preview="description",
            )
            return "description"

        result = await self.manager.features("", [str(self.image)], vision)
        row = next(
            item for item in result["_diagnostics"] if item["stage"] == "vision"
        )
        self.assertIs(row["classified_sticker"], True)
        self.assertEqual(row["raw_is_sticker"], "true")
        self.assertIs(row["json_ok"], True)
        self.assertEqual(row["sticker_source"], "is_sticker")
        self.assertEqual(row["content_chars"], 12)
        self.assertEqual(row["response_chars"], 40)
        self.assertEqual(row["response_preview"], '{"is_sticker": true}')
        self.assertEqual(row["content_preview"], "description")

    async def test_legacy_one_argument_vision_callback_still_works(self):
        async def vision(_uri):
            return "description"

        result = await self.manager.features("", [str(self.image)], vision)
        self.assertEqual(result["descriptions"], ["description"])

    async def test_image_pixel_limit_and_recompression_hash(self):
        self.manager.images.pixels = 10
        with self.assertRaisesRegex(EmbeddingError, "pixel"):
            await self.manager.images.load(str(self.image))
        self.manager.images.pixels = 100000
        other = self.root / "other.png"
        Image.open(self.image).save(other, compress_level=0)
        a = await self.manager.images.load(str(self.image))
        b = await self.manager.images.load(str(other))
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["pixel_sha256"], b["pixel_sha256"])


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.plugin = object.__new__(MemoryRebootPlugin)
        self.plugin.config = {
            "embedding_backend": "litert_lm",
            "enable_llm_judge": False,
        }
        self.plugin.data_dir = self.temp.name
        self.plugin.plugin_dir = self.temp.name
        self.plugin._cache = {}
        self.f = features("a shared news item", vector=[1, 0])
        self.plugin._mm_manager = types.SimpleNamespace(
            features=AsyncMock(side_effect=lambda *a: copy.deepcopy(self.f)),
            close=AsyncMock(),
        )
        self.event = types.SimpleNamespace(
            message_obj=types.SimpleNamespace(message_id="new"),
            chain_result=lambda x: x,
            plain_result=lambda x: x,
        )

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.temp.cleanup()

    async def process(self, sender="c", group="g"):
        return [
            r
            async for r in self.plugin._on_multimodal_message(
                self.event,
                {"content": "a shared news item", "raw_text": "a shared news item"},
                group,
                sender,
                sender,
            )
        ]

    def seed(self, senders=("a", "b"), age=7200):
        self.plugin._cache["g"] = {
            "messages": [
                message(copy.deepcopy(self.f), s, time.time() - age) for s in senders
            ],
            "last_load": time.time(),
            "dirty_dates": {},
        }

    async def test_three_senders_remind_and_group_isolation(self):
        self.seed()
        self.assertTrue(await self.process())
        self.assertFalse(await self.process(group="other"))

    async def test_same_sender_and_cooldown_preserved(self):
        self.seed()
        self.assertFalse(await self.process(sender="a"))
        self.seed(age=60)
        self.assertFalse(await self.process())

    async def test_partial_images_do_not_count_as_whole_duplicate(self):
        self.f = features(images=[("a", [1, 0])], complete=False)
        self.seed()
        self.assertFalse(await self.process())

    async def test_vision_description_is_enabled_by_default(self):
        self.plugin.config["vision_provider_id"] = "vision"
        self.seed()
        await self.process()
        self.assertIsNotNone(
            self.plugin._mm_manager.features.await_args.args[2]
        )

    async def test_multimodal_vision_parses_json_and_plain_text(self):
        provider = types.SimpleNamespace(
            text_chat=AsyncMock(
                return_value=types.SimpleNamespace(
                    completion_text=(
                        '```json\n{"is_sticker": false, '
                        '"content": "新闻截图"}\n```'
                    )
                )
            )
        )
        self.plugin.config["vision_provider_id"] = "vision"
        self.plugin.context = types.SimpleNamespace(
            get_provider_by_id=lambda _: provider
        )
        info = {}
        self.assertEqual(
            await self.plugin._multimodal_vision(
                "data:image/jpeg;base64,AAAA", info
            ),
            "新闻截图",
        )
        self.assertIs(info["sticker"], False)
        self.assertIs(info["json_ok"], True)
        self.assertEqual(info["raw_is_sticker"], "False")
        self.assertEqual(info["content_preview"], "新闻截图")
        self.assertIn('"is_sticker": false', info["response_preview"])
        self.assertGreater(info["response_chars"], 0)

        provider.text_chat.return_value = types.SimpleNamespace(
            completion_text="纯文本描述"
        )
        info = {}
        self.assertEqual(
            await self.plugin._multimodal_vision(
                "data:image/jpeg;base64,AAAA", info
            ),
            "纯文本描述",
        )
        self.assertIsNone(info["sticker"])
        self.assertIs(info["json_ok"], False)
        self.assertEqual(info["sticker_source"], "parse_failed")
        self.assertEqual(info["content_preview"], "纯文本描述")
        self.assertEqual(info["response_preview"], "纯文本描述")

        self.plugin.config["multimodal_sticker_filter_enabled"] = True
        provider.text_chat.return_value = types.SimpleNamespace(
            completion_text='{"is_sticker": true, "content": "表情包"}'
        )
        info = {}
        self.assertIs(
            await self.plugin._multimodal_vision(
                "data:image/jpeg;base64,AAAA", info
            ),
            False,
        )
        self.assertIs(info["sticker"], True)
        self.assertEqual(info["sticker_source"], "is_sticker")
        self.assertEqual(info["content_preview"], "表情包")

        provider.text_chat.return_value = types.SimpleNamespace(
            completion_text='{"is_sticker": "yes", "content": "表情包"}'
        )
        info = {}
        self.assertIs(
            await self.plugin._multimodal_vision(
                "data:image/jpeg;base64,AAAA", info
            ),
            False,
        )
        self.assertIs(info["sticker"], True)
        self.assertEqual(info["raw_is_sticker"], "yes")

        provider.text_chat.return_value = types.SimpleNamespace(
            completion_text=(
                '{"is_sticker": false, "content": "见 https://private.example/x"}'
            )
        )
        info = {}
        await self.plugin._multimodal_vision(
            "data:image/jpeg;base64,AAAA", info
        )
        self.assertNotIn("https://private.example", info["content_preview"])
        self.assertIn("[resource]", info["content_preview"])

        provider.text_chat.return_value = types.SimpleNamespace(
            completion_text=""
        )
        info = {}
        self.assertIsNone(
            await self.plugin._multimodal_vision(
                "data:image/jpeg;base64,AAAA", info
            )
        )
        self.assertEqual(info["sticker_source"], "empty_response")

    async def test_vision_description_can_be_disabled(self):
        self.plugin.config.update(
            vision_provider_id="vision",
            multimodal_vision_fallback_enabled=False,
        )
        self.seed()
        await self.process()
        self.assertIsNone(
            self.plugin._mm_manager.features.await_args.args[2]
        )

    async def test_image_descriptions_are_persisted_for_reminders(self):
        self.f = features(images=[("a", [1, 0]), ("b", [0, 1])])
        self.f["descriptions"] = ["first description", "second description"]
        self.f["images"][0]["description"] = "first description"
        self.f["images"][1]["description"] = "second description"
        self.seed()
        await self.process()
        stored = self.plugin._cache["g"]["messages"][-1]
        self.assertEqual(
            stored["image_description"],
            "first description\nsecond description",
        )
        self.assertEqual(
            [
                image.get("description")
                for image in stored["multimodal"]["images"]
            ],
            ["first description", "second description"],
        )

    async def test_new_fields_round_trip_without_overwriting_old_embedding(self):
        legacy = {
            "timestamp": time.time(),
            "embedding": [4, 5],
            "content": "old",
            "sender_id": "old",
        }
        await self.plugin._append_message("g", legacy)
        await self.process()
        await self.plugin._flush_cache("g")
        self.plugin._cache.clear()
        loaded = self.plugin._load_messages("g")
        self.assertEqual(loaded[0]["embedding"], [4, 5])
        self.assertIn("multimodal", loaded[1])
        self.assertIsNone(loaded[1]["embedding"])

    async def test_extraction_keeps_all_images_without_vision(self):
        self.event.message_str = "caption [image]"
        self.event.message_obj.message = [
            Plain("caption"),
            MessageImage(path="one"),
            MessageImage(path="two"),
        ]
        self.plugin._image_to_text = AsyncMock()
        result = await self.plugin._extract_content(self.event, "g")
        self.assertEqual(result["image_sources"], ["one", "two"])
        self.assertEqual(result["raw_text"], "caption")
        self.plugin._image_to_text.assert_not_awaited()
