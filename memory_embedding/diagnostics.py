"""Small opt-in diagnostics helpers; never serialize message or request objects."""

import time


def enabled(config):
    value = config.get("multimodal_debug_log", False)
    return value is True or (isinstance(value, str) and value.lower() == "true")


def elapsed(start):
    return round((time.perf_counter() - start) * 1000, 2)
