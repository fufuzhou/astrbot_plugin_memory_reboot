import json
import types
import unittest
from unittest.mock import AsyncMock, patch

import test_multimodal as fixtures

from memory_embedding.manager import EmbeddingManager
from memory_embedding.matching import compare, confirmed_candidates
from memory_embedding.provider import EmbeddingError


class DebugTests(unittest.IsolatedAsyncioTestCase):
    async def test_switch_off_is_silent_and_preserves_decision(self):
        for value in (False, "false", True):
            case = fixtures.PluginTests()
            await case.asyncSetUp()
            try:
                case.plugin.config["multimodal_debug_log"] = value
                case.seed()
                with patch("memory_embedding.plugin.logger") as log:
                    self.assertTrue(await case.process())
                    if value is True:
                        rows = [
                            json.loads(c.args[0].split("[MultimodalDebug] ", 1)[1])
                            for c in log.info.call_args_list
                        ]
                        self.assertIn("candidate", [r["stage"] for r in rows])
                        self.assertIn("gates", [r["stage"] for r in rows])
                        self.assertTrue(rows[-1]["remind"])
                        self.assertEqual(len({r["trace"] for r in rows}), 1)
                        self.assertNotIn(
                            "a shared news item", str(log.info.call_args_list)
                        )
                    else:
                        log.info.assert_not_called()
            finally:
                await case.asyncTearDown()

    async def test_diagnostics_not_saved_to_history(self):
        case = fixtures.PluginTests()
        await case.asyncSetUp()
        try:
            case.plugin.config["multimodal_debug_log"] = True
            case.f["_diagnostics"] = [
                {"stage": "image_embedding", "elapsed_ms": 12, "ok": True}
            ]
            with patch("memory_embedding.plugin.logger"):
                await case.process()
            await case.plugin._flush_cache("g")
            case.plugin._cache.clear()
            self.assertNotIn(
                "_diagnostics", json.dumps(case.plugin._load_messages("g"))
            )
        finally:
            await case.asyncTearDown()

    async def test_manager_records_errors_timings_without_inputs(self):
        manager = EmbeddingManager(
            {
                "multimodal_debug_log": True,
                "embedding_api_url": "http://unused/v1",
                "embedding_api_key": "SECRET_KEY",
            }
        )
        manager.provider.embed_text = AsyncMock(return_value=[1, 0])
        manager.images.load = AsyncMock(
            side_effect=[
                EmbeddingError("image_http_403"),
                {"data_uri": "SECRET_BASE64", "sha256": "hash"},
            ]
        )
        manager.provider.embed_image = AsyncMock(
            side_effect=EmbeddingError("embedding_http_401")
        )
        try:
            result = await manager.features(
                "SECRET_TEXT", ["SECRET_URL", "SECRET_PATH"]
            )
            details = result["_diagnostics"]
            self.assertEqual(
                [d["stage"] for d in details],
                ["text_embedding", "image_load", "image_load", "image_embedding"],
            )
            self.assertEqual(details[1]["error"], "image_http_403")
            self.assertEqual(details[-1]["error"], "embedding_http_401")
            self.assertTrue(all(d["elapsed_ms"] >= 0 for d in details))
            self.assertNotIn("SECRET", json.dumps(details))
            manager.config["multimodal_debug_log"] = False
            result = await manager.features("SECRET_TEXT", [])
            self.assertNotIn("_diagnostics", result)
        finally:
            await manager.close()

    async def test_search_logs_without_query_and_handles_fallback(self):
        case = fixtures.PluginTests()
        await case.asyncSetUp()
        try:
            case.plugin.config["multimodal_debug_log"] = True
            provider = types.SimpleNamespace(
                space="test",
                embed_text=AsyncMock(side_effect=EmbeddingError("embedding_http_401")),
            )
            case.plugin._embedding_manager = lambda: types.SimpleNamespace(
                provider=provider
            )
            case.plugin._find_images_by_description = lambda *a: []
            with patch("memory_embedding.plugin.logger") as log:
                results = [
                    r
                    async for r in case.plugin._search_multimodal(
                        case.event, "SECRET_QUERY", []
                    )
                ]
                self.assertTrue(results)
                text = log.info.call_args.args[0]
                self.assertNotIn("SECRET_QUERY", text)
                row = json.loads(text.split("[MultimodalDebug] ", 1)[1])
                self.assertTrue(row["keyword_fallback"])
                self.assertEqual(row["error"], "embedding_http_401")
                self.assertEqual(row["results"], 0)
        finally:
            await case.asyncTearDown()

    async def test_rejection_reasons_do_not_change_matches(self):
        a = fixtures.message(fixtures.features("price 100", vector=[1, 0]))
        b = fixtures.message(fixtures.features("price 200", vector=[1, 0]))
        details = []
        self.assertEqual(
            confirmed_candidates([b], a, {}, details), confirmed_candidates([b], a, {})
        )
        self.assertEqual(details[0]["reason"], "numbers_conflict")
        self.assertNotIn("price", json.dumps(details))
        a = fixtures.message(fixtures.features(images=[("a", [1, 0]), ("b", [0, 1])]))
        b = fixtures.message(fixtures.features(images=[("a", [1, 0]), ("x", [1, 0])]))
        detail = {}
        self.assertEqual(compare(a, b, {}, detail), compare(a, b, {}))
        self.assertEqual(detail["reason"], "image_coverage_or_threshold")
        self.assertEqual(detail["image_scores"], [[1, 1], [0, 0]])


class JudgeReasonTests(unittest.IsolatedAsyncioTestCase):
    async def test_reason_on_same_trace_and_switch_off_silent(self):
        for active in (False, True):
            case = fixtures.PluginTests()
            await case.asyncSetUp()
            try:
                case.plugin.config.update(
                    multimodal_debug_log=active,
                    enable_llm_judge=True,
                    judge_provider_id="judge",
                    embedding_api_key="SECRET_API",
                )
                case.plugin.context = types.SimpleNamespace(
                    get_provider_by_id=lambda _: types.SimpleNamespace(
                        text_chat=AsyncMock(
                            return_value=types.SimpleNamespace(
                                completion_text=json.dumps(
                                    {
                                        "should_remind": False,
                                        "reason": "normal discussion SECRET_API https://private/image",
                                    }
                                )
                            )
                        )
                    )
                )
                case.seed()
                with patch("memory_embedding.plugin.logger") as log:
                    self.assertFalse(await case.process())
                    if active:
                        rows = [
                            json.loads(c.args[0].split("[MultimodalDebug] ", 1)[1])
                            for c in log.info.call_args_list
                        ]
                        row = next(r for r in rows if r["stage"] == "llm_judge")
                        self.assertEqual(row["status"], "json_response")
                        self.assertIn("normal discussion", row["reason"])
                        self.assertNotIn("SECRET_API", row["reason"])
                        self.assertNotIn("https://private", row["reason"])
                        self.assertEqual(row["trace"], rows[-1]["trace"])
                    else:
                        log.info.assert_not_called()
            finally:
                await case.asyncTearDown()

    async def test_non_model_rejection_has_explicit_status(self):
        case = fixtures.PluginTests()
        await case.asyncSetUp()
        try:
            args = ("text", "sender", fixtures.message(case.f), [], [], {}, 2)
            details = {}
            self.assertFalse(
                await case.plugin._judge_remind(*args, diagnostics=details)
            )
            self.assertEqual(details["status"], "judge_provider_not_configured")
            case.plugin.config["judge_provider_id"] = "judge"
            provider = types.SimpleNamespace(text_chat=AsyncMock(return_value=None))
            case.plugin.context = types.SimpleNamespace(
                get_provider_by_id=lambda _: provider
            )
            await case.plugin._judge_remind(*args, diagnostics=details)
            self.assertEqual(details["status"], "empty_response")
            provider.text_chat.return_value = types.SimpleNamespace(
                completion_text="not json"
            )
            await case.plugin._judge_remind(*args, diagnostics=details)
            self.assertEqual(details["status"], "json_parse_failed_text_fallback")
            self.assertNotIn("reason", details)
            provider.text_chat.side_effect = RuntimeError("request failed")
            await case.plugin._judge_remind(*args, diagnostics=details)
            self.assertEqual(details["status"], "judge_request_failed")
        finally:
            await case.asyncTearDown()

    async def test_reason_redacted_and_bounded(self):
        from memory_embedding.diagnostics import safe_reason

        value = safe_reason(
            "Bearer abc sk-secret data:image/png;base64,AAAA " + "x" * 1000, {}
        )
        self.assertLessEqual(len(value), 500)
        self.assertNotIn("abc", value)
        self.assertNotIn("sk-secret", value)
        self.assertNotIn("AAAA", value)
