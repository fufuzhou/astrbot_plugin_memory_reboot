import asyncio
import copy
import json
import tempfile
import time
import types
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import web

import test_multimodal as fixtures
from main import MemoryRebootPlugin
from memory_embedding.images import ImageLoader, PublicResolver
from memory_embedding.manager import EmbeddingManager
from memory_embedding.provider import EmbeddingError


class SafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_dns_resolution_is_rejected(self):
        resolver = PublicResolver([])
        try:
            with patch.object(resolver.resolver, "resolve", new=AsyncMock(return_value=[{"host":"127.0.0.1"}])):
                with self.assertRaises(OSError):
                    await resolver.resolve("public-looking.example")
        finally:
            await resolver.close()

    async def test_redirect_not_followed_and_stream_size_bounded(self):
        hits = []
        async def redirect(request):
            raise web.HTTPFound("/private")
        async def private(request):
            hits.append(True)
            return web.Response(body=b"secret")
        async def oversized(request):
            return web.Response(body=b"x" * 100)
        app=web.Application()
        app.router.add_get("/redirect",redirect)
        app.router.add_get("/private",private)
        app.router.add_get("/oversized",oversized)
        runner=web.AppRunner(app)
        await runner.setup()
        site=web.TCPSite(runner,"127.0.0.1",0)
        await site.start()
        port=site._server.sockets[0].getsockname()[1]
        loader=ImageLoader({"multimodal_allowed_image_hosts":["127.0.0.1"]})
        loader.limit=10
        try:
            with self.assertRaisesRegex(EmbeddingError,"302"):
                await loader.load(f"http://127.0.0.1:{port}/redirect")
            self.assertFalse(hits)
            with self.assertRaisesRegex(EmbeddingError,"too_large"):
                await loader.load(f"http://127.0.0.1:{port}/oversized")
        finally:
            await loader.close()
            await runner.cleanup()

    async def test_invalid_configuration_keeps_exact_text_without_secret_metadata(self):
        manager=EmbeddingManager({"embedding_api_url":"http://user:SECRET@host/v1?key=SECRET"})
        try:
            result=await manager.features("exact text",[])
            self.assertTrue(result["complete"])
            self.assertIsNone(result["text_vector"])
            self.assertNotIn("SECRET",json.dumps(result))
            self.assertEqual(manager.last_error,"invalid_api_url")
        finally:
            await manager.close()

    async def test_concurrent_feature_work_is_bounded(self):
        manager=EmbeddingManager({"embedding_api_url":"http://unused/v1","embedding_max_concurrency":1})
        active=0
        peak=0
        async def embed(text):
            nonlocal active,peak
            active+=1
            peak=max(peak,active)
            await asyncio.sleep(.01)
            active-=1
            return [1,0]
        manager.provider.embed_text=embed
        try:
            await asyncio.gather(*(manager.features(str(i),[]) for i in range(5)))
            self.assertEqual(peak,1)
        finally:
            await manager.close()


class EntryPointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.plugin=object.__new__(MemoryRebootPlugin)
        self.plugin.config={"embedding_backend":"litert_lm","enable_llm_judge":False,"auto_filter_commands":False}
        self.plugin.data_dir=self.temp.name
        self.plugin.plugin_dir=self.temp.name
        self.plugin._cache={}
        self.plugin._video_analysis_semaphore=asyncio.Semaphore(1)
        self.plugin._ffmpeg_path=self.plugin._ffprobe_path=None
        self.event=types.SimpleNamespace(
            get_group_id=lambda:"g",get_sender_id=lambda:"c",get_sender_name=lambda:"Carol",
            get_self_id=lambda:"bot",message_obj=types.SimpleNamespace(message_id="new",message=[]),
            chain_result=lambda chain:chain,plain_result=lambda text:text)
        self.plugin._get_embedding=AsyncMock(side_effect=AssertionError("Must not embed media placeholders"))
        self.plugin._embedding_manager=types.MethodType(
            lambda self: (_ for _ in ()).throw(AssertionError("Must keep legacy exact route")),self.plugin)

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.temp.cleanup()

    async def exercise_exact(self,result,extra):
        history=[{"sender_id":s,"sender_name":s,"timestamp":time.time()-7200,
                  "content":"old","message_id":s,**extra} for s in ["a","b"]]
        self.plugin._cache["g"]={"messages":history,"last_load":time.time(),"dirty_dates":{}}
        self.plugin._extract_content=AsyncMock(return_value=result)
        replies=[r async for r in self.plugin.on_group_message(self.event)]
        self.assertTrue(replies)
        self.plugin._get_embedding.assert_not_awaited()
        self.assertEqual(len(self.plugin._cache["g"]["messages"]),3)

    async def test_url_message_keeps_existing_exact_engine(self):
        await self.exercise_exact({"content":"https://example.com/news","url_only":True,
                                   "url_hash":"url","has_image":False},{"url_hash":"url","url_only":True})

    async def test_forward_placeholder_never_reaches_embedding(self):
        await self.exercise_exact({"content":"[image]","forward_id":"f","forward_hash":"h",
                                   "has_image":True},{"forward_id":"f","forward_hash":"h"})

    async def test_search_without_description_uses_image_vector(self):
        provider=types.SimpleNamespace(space="test",embed_text=AsyncMock(return_value=[1,0]))
        self.plugin._embedding_manager=lambda:types.SimpleNamespace(provider=provider)
        self.plugin._can_reply_to_message=AsyncMock(return_value=False)
        record=fixtures.message(fixtures.features(images=[("a",[1,0])]))
        replies=[r async for r in self.plugin._search_multimodal(self.event,"cat",[record])]
        self.assertEqual(len(replies),1)
        self.assertIn("0.0000",replies[0].replace("1.0000","0.0000"))
        provider.embed_text.assert_awaited_once_with("cat",task="search")

    async def test_filtering_precedes_multimodal_network(self):
        self.plugin._extract_content=AsyncMock(return_value={"content":"/skip","raw_text":"/skip"})
        self.plugin._is_plugin_command=lambda content:True
        self.assertFalse([r async for r in self.plugin.on_group_message(self.event)])
