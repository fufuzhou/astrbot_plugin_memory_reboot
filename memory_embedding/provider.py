"""LiteRT-LM wire adapter. No model inference runs inside AstrBot."""

import asyncio
import hashlib
import json
import math
import time
from collections import OrderedDict
from urllib.parse import urlsplit

import aiohttp
import numpy as np


class EmbeddingError(Exception):
    """Sanitized error safe to show in status output."""


def integer(config, key, default, low, high):
    try:
        return max(low, min(high, int(config.get(key, default))))
    except (TypeError, ValueError):
        return default


def normalized(value, dimension=0):
    if not isinstance(value, list) or not value:
        raise EmbeddingError("invalid_vector")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise EmbeddingError("invalid_vector")
    vector = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 0:
        raise EmbeddingError("invalid_vector")
    if dimension and len(vector) != dimension:
        raise EmbeddingError("dimension_mismatch")
    return (vector / norm).tolist()


class LiteRTProvider:
    def __init__(self, config):
        self.config = dict(config)
        self.url = str(config.get("embedding_api_url", "")).strip().rstrip("/")
        if self.url.endswith("/embeddings"):
            self.url = self.url[: -len("/embeddings")]
        try:
            parsed = urlsplit(self.url)
        except ValueError:
            parsed = urlsplit("")
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            self.url_error = "invalid_api_url"
        else:
            self.url_error = ""
        self.model = str(config.get("embedding_model", "embeddinggemma-2-440m")).strip()
        self.dimension = integer(config, "embedding_dimensions", 768, 0, 16384)
        self.actual_dimension = self.dimension
        self.timeout = integer(config, "embedding_timeout", 30, 1, 180)
        self.retries = integer(config, "embedding_max_retries", 1, 0, 3)
        self.semaphore = asyncio.Semaphore(
            integer(config, "embedding_max_concurrency", 2, 1, 8)
        )
        self.session = None
        self.cache = OrderedDict()
        self.last_error = ""
        self.signature = {
            "schema": 1,
            "model": self.model,
            "revision": str(config.get("embedding_model_revision", "local-v1")),
            "endpoint": self.url if not self.url_error else "invalid",
            "strategy": "eg2-shared-images-v1",
            "preprocessing": "exif-rgb-jpeg90-v1",
            "max_side": integer(config, "multimodal_max_side", 1024, 224, 2048),
        }

    async def _session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                trust_env=False,
            )
        return self.session

    @property
    def space(self):
        data = {**self.signature, "dimensions": self.actual_dimension}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get_capabilities(self):
        return {
            "provider": "litert_lm",
            "model": self.model,
            "text": True,
            "image": True,
            "text_image_fusion": True,
            "multi_image_fusion": True,
            "max_fusion_images": 4,
            "dimensions": self.actual_dimension,
            "space": self.space,
            "max_images": integer(self.config, "multimodal_max_images", 4, 1, 8),
            "service_max_input_tokens": 1024,
        }

    async def _request(self, content):
        if self.url_error:
            self.last_error = self.url_error
            raise EmbeddingError(self.url_error)
        payload = {"model": self.model, "input": content, "encoding_format": "float"}
        if self.dimension:
            payload["dimensions"] = self.dimension
        cache_key = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        async with self.semaphore:
            cached = self.cache.get(cache_key)
            if cached and time.monotonic() - cached[0] < 600:
                self.cache.move_to_end(cache_key)
                return list(cached[1])
            session = await self._session()
            key = str(self.config.get("embedding_api_key", "")).strip()
            headers = {"Authorization": "Bearer " + key} if key else {}
            for attempt in range(self.retries + 1):
                try:
                    async with session.post(
                        self.url + "/embeddings",
                        json=payload,
                        headers=headers,
                        allow_redirects=False,
                    ) as response:
                        if response.status != 200:
                            error = EmbeddingError(
                                "embedding_http_" + str(response.status)
                            )
                            if (
                                response.status in (429, 500, 502, 503, 504)
                                and attempt < self.retries
                            ):
                                await asyncio.sleep(0.5 * 2**attempt)
                                continue
                            raise error
                        raw = bytearray()
                        async for chunk in response.content.iter_chunked(65536):
                            raw.extend(chunk)
                            if len(raw) > 1024 * 1024:
                                raise EmbeddingError("embedding_response_too_large")
                        body = json.loads(raw)
                    rows = body.get("data")
                    if (
                        not isinstance(rows, list)
                        or len(rows) != 1
                        or rows[0].get("index", 0) != 0
                    ):
                        raise EmbeddingError("expected_one_joint_vector")
                    vector = normalized(rows[0].get("embedding"), self.actual_dimension)
                    self.actual_dimension = len(vector)
                    self.cache[cache_key] = (time.monotonic(), vector)
                    while len(self.cache) > 256:
                        self.cache.popitem(last=False)
                    self.last_error = ""
                    return list(vector)
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt < self.retries:
                        await asyncio.sleep(0.5 * 2**attempt)
                        continue
                    self.last_error = "embedding_network_error"
                    raise EmbeddingError(self.last_error) from None
                except (
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                    EmbeddingError,
                ) as exc:
                    self.last_error = (
                        str(exc)
                        if isinstance(exc, EmbeddingError)
                        else "invalid_embedding_response"
                    )
                    raise EmbeddingError(self.last_error) from None

    async def embed_text(self, text, task="similarity"):
        if len(text.encode("utf-8")) > integer(
            self.config, "embedding_max_text_bytes", 900, 64, 20000
        ):
            raise EmbeddingError("text_budget_exceeded")
        prefix = (
            "task: search query | text: "
            if task == "search"
            else "task: sentence similarity | text: "
        )
        return await self._request(prefix + text.strip())

    async def embed_image(self, image):
        return await self._request([{"type": "image_url", "image_url": {"url": image}}])

    async def embed_multimodal(self, text, images):
        if len(text.encode("utf-8")) > integer(
            self.config, "embedding_max_text_bytes", 900, 64, 20000
        ):
            raise EmbeddingError("text_budget_exceeded")
        if not 1 <= len(images) <= 4:
            raise EmbeddingError("fusion_image_limit")
        parts = []
        if text.strip():
            parts.append(
                {
                    "type": "text",
                    "text": "task: sentence similarity | text: " + text.strip(),
                }
            )
        parts.extend(
            {"type": "image_url", "image_url": {"url": image}} for image in images
        )
        return await self._request(parts)

    async def test_connection(self):
        await self.embed_text("connection test")
        return self.get_capabilities()

    async def close(self):
        if self.session is not None:
            await self.session.close()
        self.cache.clear()
