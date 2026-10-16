"""Bounded image intake; never persist chat images or log their source URLs."""

import asyncio
import base64
import hashlib
import io
import ipaddress
import socket
import tempfile
import warnings
from pathlib import Path
from urllib.parse import unquote, urlsplit

import aiohttp
from PIL import Image, ImageOps

from .provider import EmbeddingError, integer


class PublicResolver(aiohttp.abc.AbstractResolver):
    def __init__(self, allowed_hosts):
        self.allowed_hosts = set(allowed_hosts)
        self.resolver = aiohttp.resolver.ThreadedResolver()

    async def resolve(self, host, port=0, family=socket.AF_UNSPEC):
        entries = await self.resolver.resolve(host, port, family)
        if host.lower() not in self.allowed_hosts:
            for entry in entries:
                if not ipaddress.ip_address(entry["host"]).is_global:
                    raise OSError("image_address_not_allowed")
        return entries

    async def close(self):
        await self.resolver.close()


class ImageLoader:
    def __init__(self, config):
        self.limit = (
            integer(config, "multimodal_max_image_size_mb", 10, 1, 20) * 1024 * 1024
        )
        self.pixels = integer(
            config, "multimodal_max_pixels", 20000000, 100000, 40000000
        )
        self.side = integer(config, "multimodal_max_side", 1024, 224, 2048)
        self.timeout = integer(config, "embedding_timeout", 30, 1, 180)
        self.allowed_hosts = {
            str(x).lower() for x in config.get("multimodal_allowed_image_hosts", [])
        }
        configured_roots = config.get("multimodal_local_image_roots", [])
        self.roots = [Path(x).resolve() for x in configured_roots]
        # AstrBot adapters commonly materialize incoming images in these folders.
        self.roots += [
            Path(tempfile.gettempdir()).resolve(),
            (Path.cwd() / "data/temp").resolve(),
        ]
        self.session = None
        self.semaphore = asyncio.Semaphore(2)

    def _local_bytes(self, source):
        path = Path(source).resolve()
        if not any(path.is_relative_to(root) for root in self.roots):
            raise EmbeddingError("image_path_not_allowed")
        with path.open("rb") as file:
            raw = file.read(self.limit + 1)
        if len(raw) > self.limit:
            raise EmbeddingError("image_too_large")
        return raw

    async def _bytes(self, source):
        if source.startswith("data:"):
            header, encoded = source.split(",", 1)
            if (
                not header.startswith("data:image/")
                or ";base64" not in header
                or len(encoded) > (self.limit * 4 // 3 + 8)
            ):
                raise EmbeddingError("invalid_image_data")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > self.limit:
                raise EmbeddingError("image_too_large")
            return raw
        if source.startswith("base64://"):
            return await self._bytes("data:image/jpeg;base64," + source[9:])
        parsed = urlsplit(source)
        if parsed.scheme in ("http", "https"):
            if parsed.username or parsed.password or not parsed.hostname:
                raise EmbeddingError("invalid_image_url")
            try:
                address = ipaddress.ip_address(parsed.hostname)
            except ValueError:
                address = None
            if (
                address
                and not address.is_global
                and parsed.hostname.lower() not in self.allowed_hosts
            ):
                raise EmbeddingError("image_address_not_allowed")
            if self.session is None:
                resolver = PublicResolver(self.allowed_hosts)
                self.session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(resolver=resolver),
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    trust_env=False,
                )
            async with self.session.get(source, allow_redirects=False) as response:
                if response.status != 200:
                    raise EmbeddingError("image_http_" + str(response.status))
                if response.content_length and response.content_length > self.limit:
                    raise EmbeddingError("image_too_large")
                raw = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    raw.extend(chunk)
                    if len(raw) > self.limit:
                        raise EmbeddingError("image_too_large")
                return bytes(raw)
        if parsed.scheme == "file":
            if parsed.netloc:
                raise EmbeddingError("image_network_path_not_allowed")
            source = unquote(parsed.path)
        elif parsed.scheme and not Path(source).is_absolute():
            raise EmbeddingError("unsupported_image_source")
        if source.startswith(("\\\\", "//")):
            raise EmbeddingError("image_network_path_not_allowed")
        return await asyncio.to_thread(self._local_bytes, source)

    def _prepare(self, raw):
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > self.pixels:
                    raise EmbeddingError("image_pixel_limit")
                image.seek(
                    0
                )  # Animated images use the first frame, explicitly versioned.
                image = ImageOps.exif_transpose(image).convert("RGB")
                pixels_hash = hashlib.sha256(
                    str(image.size).encode() + image.tobytes()
                ).hexdigest()
                small = image.convert("L").resize((17, 16), Image.Resampling.LANCZOS)
                values = list(small.tobytes())
                bits = "".join(
                    "1" if values[y * 17 + x] > values[y * 17 + x + 1] else "0"
                    for y in range(16)
                    for x in range(16)
                )
                dhash = f"{int(bits, 2):064x}"
                image.thumbnail((self.side, self.side), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=90)
        return {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "pixel_sha256": pixels_hash,
            "dhash": dhash,
            "data_uri": "data:image/jpeg;base64,"
            + base64.b64encode(output.getvalue()).decode(),
        }

    async def load(self, source):
        async with self.semaphore:
            try:
                raw = await self._bytes(str(source))
                return await asyncio.to_thread(self._prepare, raw)
            except EmbeddingError:
                raise
            except Exception:
                raise EmbeddingError("image_load_failed") from None

    async def close(self):
        if self.session is not None:
            await self.session.close()
