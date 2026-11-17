"""Video providers: services that turn a text description into an mp4 clip.

Only Seedance (ByteDance, through BytePlus ModelArk) is set up for now. The
"provider" part of the "video" config section picks it:

    "provider": {"type": "seedance", "api_key": "env:ARK_API_KEY", ...}

To add another service later, write a VideoProvider subclass and register it
with register_video_provider("name", TheClass).
"""
from .base import Clip, ModelUnavailable, VideoProvider
from .seedance import SeedanceProvider

PROVIDERS = {"seedance": SeedanceProvider}


def register_video_provider(name, cls):
    PROVIDERS[name] = cls


def create_video_provider(cfg):
    """cfg: {"type": "seedance", ...settings for that provider}"""
    cfg = dict(cfg)
    return PROVIDERS[cfg.pop("type", "seedance")](**cfg)


def create_video_providers(video_cfg):
    """The providers to try in order, from the "video" config section.

    "provider" holds the shared settings. An optional "models" list gives the
    models to fall back through, each entry overriding the shared settings:

        "provider": {"type": "seedance", "api_key": "env:ARK_API_KEY", "resolution": "480p", ...},
        "models": [
          {"model": "seedance-1-0-pro-fast-251015", "usd_per_million_tokens": 1.0, "free_quota_only": true},
          {"model": "seedance-1-0-pro-250528", "usd_per_million_tokens": 2.5, "free_quota_only": true}
        ]

    Without "models", only the "provider" itself is used.
    """
    shared = video_cfg["provider"]
    models = video_cfg.get("models") or [{}]
    return [create_video_provider({**shared, **model}) for model in models]
