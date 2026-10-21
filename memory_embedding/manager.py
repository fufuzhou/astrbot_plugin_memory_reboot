"""Feature extraction, bounded vector cache and provider lifetime."""

import asyncio
import time

from .diagnostics import elapsed, enabled
from .images import ImageLoader
from .provider import EmbeddingError, LiteRTProvider, integer


class EmbeddingManager:
    def __init__(self, config, images=None):
        self.config = dict(config)
        self.provider = LiteRTProvider(config)
        self.images = images if images is not None else ImageLoader(config)
        self.last_error = ""
        self.workers = asyncio.Semaphore(
            integer(config, "embedding_max_concurrency", 2, 1, 8)
        )
        self.processed = 0

    async def _embed(self, operation, diagnostics=None, stage="embedding", index=None):
        start = time.perf_counter()
        error = None
        vector = None
        try:
            vector = await operation
            return vector
        except EmbeddingError as exc:
            error = self.last_error = str(exc)
            return None
        finally:
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "stage": stage,
                        "image_index": index,
                        "elapsed_ms": elapsed(start),
                        "ok": vector is not None,
                        "dimensions": len(vector) if vector is not None else None,
                        "error": error,
                    }
                )

    async def features(self, text, sources, vision=None):
        async with self.workers:
            self.last_error = ""
            diagnostics = [] if enabled(self.config) else None
            limit = integer(self.config, "multimodal_max_images", 4, 1, 8)
            result = {
                "schema": 1,
                "provider": "litert_lm",
                "model": self.provider.model,
                "revision": self.provider.signature["revision"],
                "encoding": dict(self.provider.signature),
                "task": "similarity",
                "modalities": (["text"] if text else [])
                + (["image"] if sources else []),
                "text": text,
                "images": [],
                "image_count": len(sources),
                "complete": len(sources) <= limit,
                "text_vector": None,
                "fusion_vector": None,
                "descriptions": [],
            }
            if text:
                result["text_vector"] = await self._embed(
                    self.provider.embed_text(text), diagnostics, "text_embedding"
                )
            uris = []
            excluded = 0
            for source_index, source in enumerate(sources[:limit]):
                start = time.perf_counter()
                attempts = (
                    [] if diagnostics is not None and isinstance(source, list) else None
                )
                try:
                    image = await self.images.load_candidates(source, attempts)
                except EmbeddingError as exc:
                    self.last_error = str(exc)
                    result["complete"] = False
                    if diagnostics is not None:
                        diagnostics.append(
                            {
                                "stage": "image_load",
                                "image_index": source_index,
                                "elapsed_ms": elapsed(start),
                                "ok": False,
                                "error": str(exc),
                            }
                        )
                    continue
                finally:
                    if attempts:
                        diagnostics.extend(
                            {
                                "stage": "image_source_attempt",
                                "image_index": source_index,
                                **item,
                            }
                            for item in attempts
                        )
                if diagnostics is not None:
                    diagnostics.append(
                        {
                            "stage": "image_load",
                            "image_index": source_index,
                            "elapsed_ms": elapsed(start),
                            "ok": True,
                        }
                    )
                image["source_index"] = source_index
                uri = image.pop("data_uri")
                if vision:
                    start = time.perf_counter()
                    vision_info = {}
                    description = await vision(uri, vision_info)
                    if diagnostics is not None:
                        diagnostics.append(
                            {
                                "stage": "vision",
                                "image_index": source_index,
                                "elapsed_ms": elapsed(start),
                                "excluded_sticker": description is False,
                                "has_description": bool(description),
                                "classified_sticker": vision_info.get("sticker"),
                                "raw_is_sticker": vision_info.get("raw_is_sticker"),
                                "json_ok": vision_info.get("json_ok"),
                                "sticker_source": vision_info.get("sticker_source"),
                                "content_chars": vision_info.get("content_chars"),
                            }
                        )
                    if description is False:  # Explicit sticker classification only.
                        excluded += 1
                        continue
                    if description:
                        text_description = str(description).strip()
                        if text_description:
                            result["descriptions"].append(text_description)
                            # Keep the description next to its image so reminders
                            # and /搜图 can show the matching image's caption.
                            image["description"] = text_description
                image["vector"] = await self._embed(
                    self.provider.embed_image(uri),
                    diagnostics,
                    "image_embedding",
                    source_index,
                )
                result["images"].append(image)
                uris.append(uri)
            result["image_count"] -= excluded
            result["complete"] = (
                result["complete"] and len(result["images"]) == result["image_count"]
            )
            if (
                self.config.get("multimodal_fusion_enabled", True)
                and (text or len(uris) > 1)
                and 1 <= len(uris) <= 4
                and result["complete"]
            ):
                result["fusion_vector"] = await self._embed(
                    self.provider.embed_multimodal(text, uris),
                    diagnostics,
                    "fusion_embedding",
                )
            result["space"] = self.provider.space
            result["dimensions"] = self.provider.actual_dimension
            if diagnostics is not None:
                result["_diagnostics"] = diagnostics
            self.processed += 1
            return result

    async def close(self):
        await self.provider.close()
        await self.images.close()
