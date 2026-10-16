import asyncio
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import test_multimodal as fixtures
from aiohttp import web
from PIL import Image

from memory_embedding.images import ImageLoader
from memory_embedding.manager import EmbeddingManager
from memory_embedding.provider import EmbeddingError


class ImageFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_or_corrupt_path_then_http_success(self):
        hits = []
        buffer = io.BytesIO()
        Image.new("RGB", (20, 20), "red").save(buffer, format="PNG")

        async def serve(request):
            hits.append(request.path)
            return web.Response(body=buffer.getvalue(), content_type="image/png")

        app = web.Application()
        app.router.add_get("/image", serve)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/image"
        loader = ImageLoader({"multimodal_allowed_image_hosts": ["127.0.0.1"]})
        try:
            with tempfile.TemporaryDirectory() as root:
                invalid = Path(root) / "invalid.png"
                invalid.write_bytes(b"broken")
                for first in (str(Path(root) / "missing.png"), str(invalid)):
                    attempts = []
                    result = await loader.load_candidates(
                        [first, url, url, "unused"], attempts
                    )
                    self.assertTrue(result["dhash"])
                    self.assertEqual([a["ok"] for a in attempts], [False, True])
                    self.assertEqual(attempts[1]["source_type"], "http")
                    self.assertNotIn(url, str(attempts))
                self.assertEqual(len(hits), 2)
        finally:
            await loader.close()
            await runner.cleanup()

    async def test_http_failure_then_another_source_and_dedup(self):
        loader = ImageLoader({})
        loader.load = AsyncMock(
            side_effect=[EmbeddingError("image_http_403"), {"sha256": "ok"}]
        )
        try:
            result = await loader.load_candidates(
                ["first", "first", "second", "unused"]
            )
            self.assertEqual(result["sha256"], "ok")
            self.assertEqual(
                [c.args[0] for c in loader.load.await_args_list], ["first", "second"]
            )
        finally:
            await loader.close()

    async def test_shared_timeout_and_existing_private_host_restriction(self):
        loader = ImageLoader({})
        try:
            with self.assertRaisesRegex(EmbeddingError, "image_address_not_allowed"):
                await loader.load_candidates(["http://127.0.0.1/private"])

            async def slow(source):
                await asyncio.sleep(1)

            loader.load = AsyncMock(side_effect=slow)
            loader.timeout = 0.01
            with self.assertRaisesRegex(EmbeddingError, "image_sources_timeout"):
                await loader.load_candidates(["slow", "unused"])
            self.assertEqual(loader.load.await_count, 1)
        finally:
            await loader.close()

    async def test_all_sources_fail_preserves_next_image_position(self):
        manager = EmbeddingManager(
            {"embedding_api_url": "http://unused/v1", "multimodal_debug_log": True}
        )
        manager.images.load = AsyncMock(
            side_effect=[
                EmbeddingError("image_load_failed"),
                EmbeddingError("image_http_404"),
                {"data_uri": "inline", "sha256": "second", "dhash": "abc"},
            ]
        )
        manager.provider.embed_image = AsyncMock(return_value=[1, 0])
        try:
            result = await manager.features("", [["missing", "bad_url"], "second"])
            self.assertFalse(result["complete"])
            self.assertEqual(result["image_count"], 2)
            self.assertEqual(result["images"][0]["source_index"], 1)
            self.assertEqual(manager.provider.embed_image.await_count, 1)
            self.assertEqual(
                len(
                    [
                        d
                        for d in result["_diagnostics"]
                        if d["stage"] == "image_source_attempt"
                    ]
                ),
                2,
            )
        finally:
            await manager.close()

    async def test_raw_sources_stay_bound_to_correct_image(self):
        p = object.__new__(fixtures.MemoryRebootPlugin)
        a = fixtures.MessageImage(path="a-local", url="a-url", file="a-url")
        b = fixtures.MessageImage(path="b-local")
        event = types.SimpleNamespace(
            message_obj=types.SimpleNamespace(
                message=[a, b],
                raw_message={
                    "message": [
                        {"type": "image", "data": {"url": "a-raw"}},
                        {"type": "image", "data": {"url": "b-raw"}},
                    ]
                },
            )
        )
        self.assertEqual(
            p._image_sources(event),
            [["a-local", "a-url", "a-raw"], ["b-local", "b-raw"]],
        )
        event.message_obj.raw_message["message"].pop()
        # Do not guess source alignment when the adapter changes image count.
        self.assertEqual(
            p._image_sources(event), [["a-local", "a-url"], "b-local"]
        )


    def legacy_plugin_and_event(self):
        plugin = object.__new__(fixtures.MemoryRebootPlugin)
        plugin.config = {}
        plugin._cache = {}
        plugin._image_to_text = AsyncMock(return_value=("description", "content"))
        event = types.SimpleNamespace(
            message_str="",
            message_obj=types.SimpleNamespace(
                message=[fixtures.MessageImage(path="missing", url="valid", file="unused")],
                raw_message={},
            ),
        )
        return plugin, event

    async def test_legacy_fallback_recognizes_once_and_reuses_hash(self):
        plugin, event = self.legacy_plugin_and_event()
        loader = plugin._image_loader()
        loader.load = AsyncMock(side_effect=[
            EmbeddingError("image_load_failed"),
            {"data_uri": "data:image/jpeg;base64,AAAA", "dhash": "abc"},
        ])
        try:
            result = await plugin._extract_content(event)
            self.assertEqual(result["image_hash"], "abc")
            self.assertEqual(result["image_description"], "description")
            plugin._image_to_text.assert_awaited_once_with("data:image/jpeg;base64,AAAA")
            self.assertEqual([c.args[0] for c in loader.load.await_args_list],
                             ["missing", "valid"])
            self.assertFalse(hasattr(plugin, "_mm_manager"))
        finally:
            await plugin.terminate()

    async def test_legacy_sticker_does_not_retry_other_sources(self):
        plugin, event = self.legacy_plugin_and_event()
        plugin._image_to_text.return_value = None
        loader = plugin._image_loader()
        loader.load = AsyncMock(return_value={"data_uri": "inline", "dhash": "abc"})
        try:
            self.assertIsNone(await plugin._extract_content(event))
            loader.load.assert_awaited_once_with("missing")
            plugin._image_to_text.assert_awaited_once_with("inline")
        finally:
            await plugin.terminate()

    async def test_legacy_failed_image_continues_to_next_image(self):
        plugin, event = self.legacy_plugin_and_event()
        event.message_obj.message.append(fixtures.MessageImage(url="second-image"))
        loader = plugin._image_loader()
        loader.load = AsyncMock(side_effect=[
            EmbeddingError("image_load_failed"),
            EmbeddingError("image_http_404"),
            EmbeddingError("image_load_failed"),
            {"data_uri": "second-inline", "dhash": "def"},
        ])
        try:
            result = await plugin._extract_content(event)
            self.assertEqual(result["image_hash"], "def")
            plugin._image_to_text.assert_awaited_once_with("second-inline")
        finally:
            await plugin.terminate()

    async def test_backends_share_loader_and_close_it_once(self):
        plugin, _ = self.legacy_plugin_and_event()
        plugin.config["embedding_api_url"] = "http://unused/v1"
        loader = plugin._image_loader()
        loader.close = AsyncMock()
        manager = plugin._embedding_manager()
        self.assertIs(manager.images, loader)
        await plugin.terminate()
        loader.close.assert_awaited_once()

    async def test_legacy_only_closes_loader(self):
        plugin, _ = self.legacy_plugin_and_event()
        loader = plugin._image_loader()
        loader.close = AsyncMock()
        await plugin.terminate()
        loader.close.assert_awaited_once()
