"""Integrate multimodal features without changing legacy media handlers."""

import asyncio
import json
import time
import uuid

from astrbot.api import logger
from astrbot.api.message_components import Image, Plain, Reply

from .diagnostics import elapsed, enabled, safe_reason
from .images import ImageLoader
from .judge import exact_repeat, match_evidence
from .manager import EmbeddingManager
from .matching import candidates, compare, confirmed_candidates, search_images
from .provider import EmbeddingError, integer


class MultimodalMixin:
    def _mm_debug(self, trace, stage, **fields):
        if enabled(self.config):
            logger.info(
                "[Memory Reboot][MultimodalDebug] "
                + json.dumps(
                    {"trace": trace, "stage": stage, **fields}, ensure_ascii=False
                )
            )

    def _image_sources(self, event):
        message = getattr(event, "message_obj", None)
        images = [c for c in getattr(message, "message", []) if isinstance(c, Image)]
        raw = getattr(message, "raw_message", None)
        segments = raw.get("message", []) if isinstance(raw, dict) else []
        raw_images = (
            [
                s.get("data", {})
                for s in segments
                if isinstance(s, dict)
                and s.get("type") == "image"
                and isinstance(s.get("data"), dict)
            ]
            if isinstance(segments, list)
            else []
        )
        aligned = len(images) == len(raw_images)
        result = []
        for index, component in enumerate(images):
            candidates = [
                getattr(component, key, None) for key in ("path", "url", "file")
            ]
            if aligned:
                candidates += [
                    raw_images[index].get(key) for key in ("path", "url", "file")
                ]
            candidates = list(
                dict.fromkeys(
                    str(s).strip()
                    for s in candidates
                    if isinstance(s, str) and s.strip()
                )
            )
            result.append(candidates[0] if len(candidates) == 1 else candidates)
        return result

    def _multimodal_enabled(self):
        return self.config.get("embedding_backend", "legacy") == "litert_lm"

    def _image_loader(self):
        loader = getattr(self, "_shared_image_loader", None)
        if loader is None:
            loader = self._shared_image_loader = ImageLoader(self.config)
        return loader

    def _embedding_manager(self):
        manager = getattr(self, "_mm_manager", None)
        if manager is None:
            manager = self._mm_manager = EmbeddingManager(self.config, images=self._image_loader())
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
                provider.text_chat(
                    prompt=self.config.get("vision_prompt")
                    or self.DEFAULT_VISION_PROMPT,
                    contexts=[],
                    image_urls=[uri],
                ),
                timeout=integer(self.config, "embedding_timeout", 30, 1, 180),
            )
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

    async def _on_multimodal_message(
        self, event, extracted, group_id, sender_id, sender_name
    ):
        locks = getattr(self, "_mm_group_locks", None)
        if locks is None:
            locks = self._mm_group_locks = {}
        async with locks.setdefault(str(group_id), asyncio.Lock()):
            try:
                manager = self._embedding_manager()
            except EmbeddingError as exc:
                logger.warning(f"[Memory Reboot] multimodal configuration: {exc}")
                return
            use_vision = self.config.get(
                "multimodal_sticker_filter_enabled", False
            ) or self.config.get("multimodal_vision_fallback_enabled", False)
            trace = uuid.uuid4().hex[:12]
            start = time.perf_counter()
            self._mm_debug(
                trace,
                "start",
                text_chars=len(extracted.get("raw_text", "")),
                source_images=len(extracted.get("image_sources", [])),
            )
            features = await manager.features(
                extracted.get("raw_text", ""),
                extracted.get("image_sources", []),
                self._multimodal_vision if use_vision else None,
            )
            for detail in features.pop("_diagnostics", []):
                self._mm_debug(trace, **detail)
            self._mm_debug(
                trace,
                "features",
                elapsed_ms=elapsed(start),
                complete=features["complete"],
                image_count=features["image_count"],
                loaded_images=len(features["images"]),
                text_vector=bool(features["text_vector"]),
                image_vectors=sum(bool(i.get("vector")) for i in features["images"]),
                fusion_vector=bool(features["fusion_vector"]),
            )
            if not features["text"] and not features["image_count"]:
                self._mm_debug(
                    trace, "decision", remind=False, reason="empty_after_filter"
                )
                return
            descriptions = features.pop("descriptions", [])
            now = time.time()
            message = {
                "id": str(uuid.uuid4()),
                "message_id": str(getattr(event.message_obj, "message_id", "") or ""),
                "sender_id": sender_id,
                "sender_name": sender_name,
                "timestamp": now,
                "content": extracted["content"],
                "embedding": None,
                "has_image": bool(features["image_count"]),
                "has_video": False,
                "image_description": "\n".join(descriptions) or None,
                "image_hash": features["images"][0]["dhash"]
                if len(features["images"]) == 1
                else None,
                "multimodal": features,
            }
            messages = list(self._load_messages(group_id))
            details = [] if enabled(self.config) else None
            matching_start = time.perf_counter()
            selected = await asyncio.to_thread(
                confirmed_candidates, messages, message, dict(self.config), details
            )
            if details is not None:
                for detail in details:
                    self._mm_debug(trace, "candidate", **detail)
            self._mm_debug(
                trace,
                "matching",
                history_records=len(messages),
                recalled=len(details) if details is not None else None,
                confirmed=len(selected),
                elapsed_ms=elapsed(matching_start),
            )
            if not selected:
                self._mm_debug(
                    trace, "decision", remind=False, reason="no_confirmed_candidate"
                )
                await self._append_message(group_id, message)
                return
            anchor = selected[0]
            matched = anchor["message"]
            matches = await asyncio.to_thread(
                candidates, messages, message, dict(self.config)
            )
            peers = [
                m["message"]
                for m in matches
                if compare(m["message"], matched, self.config)
            ]
            senders = {m.get("sender_id") for m in peers if m.get("sender_id")} | {
                sender_id
            }
            same_sender = sender_id == matched.get("sender_id")
            min_senders = int(self.config.get("min_unique_senders", 3))
            age = now - matched.get("timestamp", 0)
            cooldown = int(self.config.get("cooldown_seconds", 3600))
            skip = same_sender or len(senders) < min_senders or age < cooldown
            self._mm_debug(
                trace,
                "gates",
                same_sender=same_sender,
                unique_senders=len(senders),
                min_unique_senders=min_senders,
                history_age_seconds=round(age, 2),
                cooldown_seconds=cooldown,
                skipped=skip,
                anchor_kind=anchor["kind"],
                anchor_score=anchor["score"],
            )
            should_remind = False
            if not skip:
                if self.config.get("enable_llm_judge", True):
                    history, current, stats = self._build_judge_contexts(
                        messages + [message],
                        anchor["index"],
                        exact_matcher=exact_repeat,
                    )
                    evidence = match_evidence(message, matched, anchor, self.config)
                    judge_content = message["content"]
                    self._mm_debug(
                        trace,
                        "judge_evidence",
                        match_kind=anchor["kind"],
                        match_score=anchor["score"],
                        current_images=len(features["images"]),
                        historical_images=len(matched["multimodal"].get("images", [])),
                        repeat_basis=stats.get("repeat_basis"),
                        exact_repeat_count=stats.get("exact_repeat_count"),
                        exact_unique_senders=stats.get("exact_unique_senders"),
                        longest_consecutive_exact=stats.get(
                            "longest_consecutive_exact"
                        ),
                    )
                    if message.get("image_description"):
                        judge_content += (
                            "\n已有当前图片描述：" + message["image_description"]
                        )
                    judge_match = dict(matched)
                    if matched.get("image_description"):
                        judge_match["content"] += (
                            "\n已有图片描述：" + matched["image_description"]
                        )
                    judge_start = time.perf_counter()
                    judge_details = {} if enabled(self.config) else None
                    should_remind = await self._judge_remind(
                        judge_content,
                        sender_name,
                        judge_match,
                        history,
                        current,
                        stats,
                        len(senders),
                        match_evidence=evidence,
                        **(
                            {"diagnostics": judge_details}
                            if judge_details is not None
                            else {}
                        ),
                    )
                    should_remind = (
                        should_remind is True or str(should_remind).lower() == "true"
                    )
                    if judge_details is not None and "reason" in judge_details:
                        judge_details["reason"] = safe_reason(
                            judge_details["reason"], self.config
                        )
                    self._mm_debug(
                        trace,
                        "llm_judge",
                        result=should_remind,
                        elapsed_ms=elapsed(judge_start),
                        **(judge_details or {}),
                    )
                else:
                    should_remind = anchor["kind"] == "exact" or self.config.get(
                        "semantic_auto_remind", True
                    )
            self._mm_debug(
                trace,
                "decision",
                remind=bool(should_remind),
                llm_enabled=bool(self.config.get("enable_llm_judge", True)),
                semantic_auto_remind=bool(
                    self.config.get("semantic_auto_remind", True)
                ),
                elapsed_ms=elapsed(start),
            )
            await self._append_message(group_id, message)
            if should_remind:
                async for result in self._send_reminder(event, matched):
                    yield result

    async def _search_multimodal(self, event, query, messages):
        trace = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        matches = []
        error = None
        fallback = False
        try:
            manager = self._embedding_manager()
            vector = await manager.provider.embed_text(query, task="search")
            matches = await asyncio.to_thread(
                search_images,
                list(messages),
                vector,
                manager.provider.space,
                dict(self.config),
            )
        except EmbeddingError as exc:
            error = str(exc)
            logger.warning(f"[Memory Reboot] multimodal search: {exc}")
        if not matches:
            fallback = True
            cutoff = (
                time.time() - int(self.config.get("data_retention_days", 7)) * 86400
            )
            matches = self._find_images_by_description(
                [m for m in messages if m.get("timestamp", 0) >= cutoff], query
            )
        self._mm_debug(
            trace,
            "search",
            query_chars=len(query),
            history_records=len(messages),
            elapsed_ms=elapsed(start),
            error=error,
            keyword_fallback=fallback,
            results=len(matches),
            top_scores=[
                m.get("_image_search_semantic_similarity") for m in matches[:10]
            ],
        )
        if not matches:
            yield event.plain_result(f"没找到与“{query}”相关的图片记录")
            return
        matched = matches[0]
        description = (
            self._get_image_description(matched) or "无图片描述（按原生图片向量检索）"
        )
        score = matched.get("_image_search_semantic_similarity")
        score_text = f"\n相似度：{score:.4f}" if score is not None else ""
        index = matched.get("_image_index")
        position = f" · 原消息第{index + 1}张图" if index is not None else ""
        details = (
            f"找到{len(matches)}个图片结果{position}\n{description[:160]}"
            f"\n{matched.get('sender_name') or '未知用户'} · "
            f"{self._format_time(matched.get('timestamp', 0))}{score_text}"
        )
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
            current = [
                m["multimodal"]
                for m in messages
                if (m.get("multimodal") or {}).get("space") == manager.provider.space
            ]
            count = sum(
                bool(f.get("text_vector"))
                + bool(f.get("fusion_vector"))
                + sum(bool(i.get("vector")) for i in f.get("images", []))
                for f in current
            )
            return (
                f"LiteRT-LM / {manager.provider.model} / {manager.provider.actual_dimension}维"
                f" / 当前空间向量{count}个 / 最近错误：{manager.last_error or manager.provider.last_error or '无'}"
                f" / 调试日志：{'开启' if enabled(self.config) else '关闭'}"
                "\n阈值待校准；逐图向量与搜图共用；最多4图补充融合"
            )
        except EmbeddingError as exc:
            return f"配置错误：{exc}"
