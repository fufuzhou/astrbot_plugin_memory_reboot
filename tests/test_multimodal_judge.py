import copy
import json
import time
import types
import unittest
from unittest.mock import AsyncMock, patch

import test_multimodal as fixtures

from memory_embedding.judge import exact_repeat, match_evidence
from memory_embedding.matching import compare


class JudgeEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_distinct_images_with_same_placeholder_are_not_repeats(self):
        case = fixtures.PluginTests()
        await case.asyncSetUp()
        try:
            now = time.time()

            def record(key, sender, age):
                item = fixtures.message(
                    fixtures.features(images=[(key, [1, 0])]), sender, now - age
                )
                item["content"] = "[图片×1]"
                return item

            original = record("same", "a", 100)
            unrelated = record("different", "b", 50)
            current = record("same", "c", 0)
            old = dict(unrelated, multimodal=None)
            _, _, stats = case.plugin._build_judge_contexts(
                [original, unrelated, old, current], 0, exact_matcher=exact_repeat
            )
            self.assertEqual(stats["exact_repeat_count"], 2)
            self.assertEqual(stats["exact_unique_senders"], 2)
            self.assertEqual(stats["longest_consecutive_exact"], 1)
            self.assertEqual(stats["exact_span_seconds"], 100)
            self.assertIn(
                "不按图片占位符", case.plugin._format_judge_context_stats(stats)
            )
            # Preserve legacy text-only statistics when the new matcher is omitted.
            _, _, legacy = case.plugin._build_judge_contexts(
                [original, unrelated, current], 0
            )
            self.assertEqual(legacy["exact_repeat_count"], 3)
            self.assertNotIn("repeat_basis", legacy)
        finally:
            await case.asyncTearDown()

    async def test_exact_repeat_requires_all_images_and_caption(self):
        a = fixtures.message(
            fixtures.features("caption", [("a", [1, 0]), ("b", [0, 1])], [1, 0])
        )
        b = copy.deepcopy(a)
        b["multimodal"]["images"].reverse()
        self.assertTrue(exact_repeat(a, b))
        b["multimodal"]["images"][0]["sha256"] = "new"
        self.assertFalse(exact_repeat(a, b))
        b = copy.deepcopy(a)
        b["multimodal"]["text"] = "other caption"
        self.assertFalse(exact_repeat(a, b))
        b = copy.deepcopy(a)
        b["multimodal"]["complete"] = False
        self.assertFalse(exact_repeat(a, b))

    async def test_semantic_evidence_reaches_custom_prompt_and_debug(self):
        case = fixtures.PluginTests()
        await case.asyncSetUp()
        try:
            case.plugin.config.update(
                enable_llm_judge=True,
                judge_provider_id="judge",
                judge_prompt="Custom contextual judge",
                multimodal_debug_log=True,
            )
            case.f = fixtures.features(images=[("historical", [1, 0])])
            case.seed()
            case.f = fixtures.features(images=[("current", [0.99, 0.1])])
            provider = types.SimpleNamespace(
                text_chat=AsyncMock(
                    return_value=types.SimpleNamespace(
                        completion_text='{"should_remind":false,"reason":"test"}'
                    )
                )
            )
            case.plugin.context = types.SimpleNamespace(
                get_provider_by_id=lambda _: provider
            )
            with patch("memory_embedding.plugin.logger") as log:
                await case.process()
            kwargs = provider.text_chat.call_args.kwargs
            prompt = kwargs["prompt"]
            self.assertTrue(prompt.startswith("Custom contextual judge"))
            self.assertIn("语义匹配", prompt)
            self.assertIn("0.994937", prompt)
            self.assertIn("当前1张、历史1张", prompt)
            self.assertIn("全部图片已通过一一匹配", prompt)
            self.assertIn("不是事实相同的概率", prompt)
            self.assertIn("本请求没有提供原图", prompt)
            self.assertNotIn("image_urls", kwargs)
            rows = [
                json.loads(c.args[0].split("[MultimodalDebug] ", 1)[1])
                for c in log.info.call_args_list
            ]
            evidence = next(r for r in rows if r["stage"] == "judge_evidence")
            self.assertEqual(evidence["exact_repeat_count"], 1)
            self.assertEqual(evidence["repeat_basis"], "multimodal_exact")
        finally:
            await case.asyncTearDown()

    async def test_exact_and_cross_modal_evidence_are_distinct(self):
        a = fixtures.message(fixtures.features(images=[("a", [1, 0])]))
        exact = match_evidence(a, a, compare(a, a, {}), {})
        self.assertIn("精确匹配", exact)
        self.assertIn("1.000000", exact)
        b = fixtures.message(fixtures.features("caption", vector=[1, 0]))
        cross = match_evidence(a, b, compare(a, b, {}), {})
        self.assertIn("文字与单图跨模态匹配", cross)
        self.assertIn("未进行两图逐图比较", cross)
        self.assertNotIn("全部图片已通过一一匹配", cross)
