"""Integrate multimodal features without changing legacy media handlers."""
import asyncio
import json
import time
import uuid

from astrbot.api import logger
from astrbot.api.message_components import Plain, Reply

from .manager import EmbeddingManager
from .matching import candidates, search_images, compare
from .provider import EmbeddingError, integer


class MultimodalMixin:
    def _multimodal_enabled(self):
        return self.config.get("embedding_backend", "legacy") == "litert_lm"

    def _embedding_manager(self):
        manager = getattr(self, "_mm_manager", None)
        if manager is None:
            manager = self._mm_manager = EmbeddingManager(self.config)
        return manager

    async def _multimodal_vision(self, uri):
        provider_id = self.config.get("vision_provider_id", "")
        if not provider_id:
            return None
        try:
            provider = self.context.get_provider_by_id(provider_id)
            if provider is None:
                return None
            response = await asyncio.wait_for(
                provider.text_chat(prompt=self.config.get("vision_prompt") or self.DEFAULT_VISION_PROMPT,
                                   contexts=[], image_urls=[uri]),
                timeout=integer(self.config, "embedding_timeout", 30, 1, 180))
            content = response.completion_text.strip()
            fence = chr(96) * 3
            if fence in content:
                content = content.split(fence)[1].removeprefix("json").strip()
            value = json.loads(content)
            sticker = str(value.get("is_sticker")).lower() == "true"
            if sticker and self.config.get("multimodal_sticker_filter_enabled", False):
                return False
            return str(value.get("content", ""))[:2000]
        except Exception:
            logger.warning("[Memory Reboot] optional vision classification failed")
            return None

    async def _on_multimodal_message(self, event, extracted, group_id, sender_id, sender_name):
        locks = getattr(self, "_mm_group_locks", None)
        if locks is None:
            locks = self._mm_group_locks = {}
        async with locks.setdefault(str(group_id), asyncio.Lock()):
            try:
                manager = self._embedding_manager()
            except EmbeddingError as exc:
                logger.warning(f"[Memory Reboot] multimodal configuration: {exc}")
                return
            use_vision = self.config.get("multimodal_sticker_filter_enabled", False) or self.config.get("multimodal_vision_fallback_enabled", False)
            features = await manager.features(extracted.get("raw_text", ""), extracted.get("image_sources", []),
                                              self._multimodal_vision if use_vision else None)
            if not features["text"] and not features["image_count"]:
                return
            descriptions = features.pop("descriptions", [])
            now = time.time()
            message = {
                "id": str(uuid.uuid4()),
                "message_id": str(getattr(event.message_obj, "message_id", "") or ""),
                "sender_id": sender_id, "sender_name": sender_name, "timestamp": now,
                "content": extracted["content"], "embedding": None,
                "has_image": bool(features["image_count"]), "has_video": False,
                "image_description": "\n".join(descriptions) or None,
                "image_hash": features["images"][0]["dhash"] if len(features["images"]) == 1 else None,
                "multimodal": features,
            }
            messages = list(self._load_messages(group_id))
            matches = await asyncio.to_thread(candidates, messages, message, dict(self.config))
            selected = matches[:integer(self.config, "multimodal_top_k", 10, 1, 50)]
            if not selected:
                await self._append_message(group_id, message)
                return
            anchor = selected[0]
            matched = anchor["message"]
            peers = [m["message"] for m in matches if compare(m["message"], matched, self.config)]
            senders = {m.get("sender_id") for m in peers if m.get("sender_id")} | {sender_id}
            skip = (sender_id == matched.get("sender_id")
                    or len(senders) < int(self.config.get("min_unique_senders", 3))
                    or now - matched.get("timestamp", 0) < int(self.config.get("cooldown_seconds", 3600)))
            should_remind = False
            if not skip:
                if self.config.get("enable_llm_judge", True):
                    history, current, stats = self._build_judge_contexts(messages + [message], anchor["index"])
                    judge_content = message["content"] + (
                        "\n[匹配依据：多模态语义/完整内容匹配。相似度不代表事实完全相同。"
                        "没有提供历史原图；不可声称看到了图片细节。仅依据可见正文和上下文判断是否适合提醒。]")
                    if message.get("image_description"):
                        judge_content += "\n已有当前图片描述：" + message["image_description"]
                    judge_match = dict(matched)
                    if matched.get("image_description"):
                        judge_match["content"] += "\n已有图片描述：" + matched["image_description"]
                    should_remind = await self._judge_remind(
                        judge_content, sender_name, judge_match, history, current, stats, len(senders))
                    should_remind = should_remind is True or str(should_remind).lower() == "true"
                else:
                    should_remind = anchor["kind"] == "exact" or self.config.get("semantic_auto_remind", True)
            await self._append_message(group_id, message)
            if should_remind:
                async for result in self._send_reminder(event, matched):
                    yield result

    async def _search_multimodal(self, event, query, messages):
        matches = []
        try:
            manager = self._embedding_manager()
            vector = await manager.provider.embed_text(query, task="search")
            matches = await asyncio.to_thread(
                search_images, list(messages), vector, manager.provider.space, dict(self.config))
        except EmbeddingError as exc:
            logger.warning(f"[Memory Reboot] multimodal search: {exc}")
        if not matches:
            cutoff = time.time() - int(self.config.get("data_retention_days", 7)) * 86400
            matches = self._find_images_by_description([m for m in messages if m.get("timestamp", 0) >= cutoff], query)
        if not matches:
            yield event.plain_result(f"没找到与“{query}”相关的图片记录")
            return
        matched = matches[0]
        description = self._get_image_description(matched) or "无图片描述（按原生图片向量检索）"
        score = matched.get("_image_search_semantic_similarity")
        score_text = f"\n相似度：{score:.4f}" if score is not None else ""
        index = matched.get("_image_index")
        position = f" · 原消息第{index + 1}张图" if index is not None else ""
        details = (f"找到{len(matches)}个图片结果{position}\n{description[:160]}"
                   f"\n{matched.get('sender_name') or '未知用户'} · "
                   f"{self._format_time(matched.get('timestamp', 0))}{score_text}")
        message_id = str(matched.get("message_id") or "")
        if await self._can_reply_to_message(event, message_id):
            yield event.chain_result([Reply(id=message_id), Plain("\n" + details)])
        else:
            yield event.plain_result("原消息已无法引用\n" + details)

    def _multimodal_status(self, messages):
        if not self._multimodal_enabled():
            return "legacy（旧模式）"
        try:
            manager = self._embedding_manager()
            current = [m["multimodal"] for m in messages if (m.get("multimodal") or {}).get("space") == manager.provider.space]
            count = sum(bool(f.get("text_vector")) + bool(f.get("fusion_vector"))
                        + sum(bool(i.get("vector")) for i in f.get("images", [])) for f in current)
            return (f"LiteRT-LM / {manager.provider.model} / {manager.provider.actual_dimension}维"
                    f" / 当前空间向量{count}个 / 最近错误：{manager.last_error or manager.provider.last_error or '无'}"
                    "\n阈值待校准；多图采用逐图向量，未启用多图融合")
        except EmbeddingError as exc:
            return f"配置错误：{exc}"
