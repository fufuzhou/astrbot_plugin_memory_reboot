"""Feature extraction, bounded vector cache and provider lifetime."""

import asyncio

from .images import ImageLoader
from .provider import EmbeddingError, LiteRTProvider, integer


class EmbeddingManager:
    def __init__(self, config):
        self.config = dict(config)
        self.provider = LiteRTProvider(config)
        self.images = ImageLoader(config)
        self.last_error = ""
        self.workers = asyncio.Semaphore(
            integer(config, "embedding_max_concurrency", 2, 1, 8)
        )
        self.processed = 0

    async def _embed(self, operation):
        try:
            return await operation
        except EmbeddingError as exc:
            self.last_error = str(exc)
            return None

    async def features(self, text, sources, vision=None):
        async with self.workers:
            self.last_error = ""
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
                    self.provider.embed_text(text)
                )
            uris = []
            excluded = 0
            for source_index, source in enumerate(sources[:limit]):
                try:
                    image = await self.images.load(source)
                except EmbeddingError as exc:
                    self.last_error = str(exc)
                    result["complete"] = False
                    continue
                image["source_index"] = source_index
                uri = image.pop("data_uri")
                if vision:
                    description = await vision(uri)
                    if description is False:  # Explicit sticker classification only.
                        excluded += 1
                        continue
                    if description:
                        result["descriptions"].append(str(description))
                image["vector"] = await self._embed(self.provider.embed_image(uri))
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
                    self.provider.embed_multimodal(text, uris)
                )
            result["space"] = self.provider.space
            result["dimensions"] = self.provider.actual_dimension
            self.processed += 1
            return result

    async def close(self):
        await self.provider.close()
        await self.images.close()
