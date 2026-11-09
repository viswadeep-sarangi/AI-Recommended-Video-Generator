"""Video providers: services that turn a text description into an mp4 clip.

Only Seedance (ByteDance, through BytePlus ModelArk) is set up for now. The
"provider" part of the "video" config section picks it:

    "provider": {"type": "seedance", "api_key": "env:ARK_API_KEY", ...}

To add another service later, write a VideoProvider subclass and register it
with register_video_provider("name", TheClass).
"""
from .base import Clip, VideoProvider
from .seedance import SeedanceProvider

PROVIDERS = {"seedance": SeedanceProvider}


def register_video_provider(name, cls):
    PROVIDERS[name] = cls


def create_video_provider(cfg):
    """cfg: {"type": "seedance", ...settings for that provider}"""
    cfg = dict(cfg)
    return PROVIDERS[cfg.pop("type", "seedance")](**cfg)
