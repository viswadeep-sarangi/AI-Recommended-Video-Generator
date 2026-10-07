from .base import VideoProvider
from .providers import GenericHTTPProvider, SoraProvider

PROVIDERS = {"sora": SoraProvider, "generic_http": GenericHTTPProvider}


def register_video_provider(name, cls):
    PROVIDERS[name] = cls


class VideoGenerator:
    """Single entry point. Picks providers by topic category and falls back on failure
    (e.g. when tokens run out), so the backend can be swapped from config alone.

    cfg: {"providers": {name: {"type": ..., ...}}, "default": [names...], "by_category": {cat: [names...]}}
    """

    def __init__(self, cfg, providers=None):
        self.cfg = cfg
        self.providers = providers or {
            n: PROVIDERS[p["type"]](**{k: v for k, v in p.items() if k != "type"})
            for n, p in cfg["providers"].items()
        }

    def order(self, category=None):
        return self.cfg.get("by_category", {}).get(category) or self.cfg["default"]

    def generate(self, prompt, duration, category=None):
        errors = []
        for name in self.order(category):
            try:
                return name, self.providers[name].generate(prompt, duration)
            except Exception as e:
                errors.append(f"{name}: {e}")
        raise RuntimeError("all video providers failed: " + "; ".join(errors))
