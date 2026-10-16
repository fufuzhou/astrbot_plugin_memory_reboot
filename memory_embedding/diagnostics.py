"""Small opt-in diagnostics helpers; never serialize message or request objects."""

import re
import time


def enabled(config):
    value = config.get("multimodal_debug_log", False)
    return value is True or (isinstance(value, str) and value.lower() == "true")


def elapsed(start):
    return round((time.perf_counter() - start) * 1000, 2)


def safe_reason(value, config):
    """Bound diagnostic text and redact credentials and resource URLs."""
    text = str(value)
    secret = str(config.get("embedding_api_key", "") or "")
    if secret:
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?i)(?:https?://|data:image/|base64://)\S+", "[resource]", text)
    text = re.sub(r"(?i)\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]+", "[redacted]", text)
    return text[:500]
