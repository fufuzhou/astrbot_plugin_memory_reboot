"""OneBot video get_file capability hints, scoped to a connection and account."""

import time
from collections import OrderedDict


def image_voice_only(value):
    if isinstance(value, dict):
        text = str(value.get("wording", "")) + " " + str(value.get("message", ""))
    else:
        result = getattr(value, "result", None)
        text = str(value)
        if isinstance(result, dict):
            text += " " + str(result.get("wording", ""))
    return "get_file only resolves cached image/voice ids" in text.lower()


class VideoFileCapabilities:
    def __init__(self):
        self.unsupported = OrderedDict()

    def _expire(self):
        now = time.monotonic()
        for key, (_, expiry) in list(self.unsupported.items()):
            if expiry <= now:
                del self.unsupported[key]

    def remember_unsupported(self, bot, self_id):
        self._expire()
        key = (id(bot), str(self_id or ""))
        self.unsupported[key] = (bot, time.monotonic() + 600)
        self.unsupported.move_to_end(key)
        while len(self.unsupported) > 64:
            self.unsupported.popitem(last=False)

    def skip_reason(self, backend, bot, self_id):
        if backend == "snowluma":
            return "snowluma_image_voice_only"
        self._expire()
        if backend == "auto" and (id(bot), str(self_id or "")) in self.unsupported:
            return "cached_image_voice_only"
        return None
