"""Video providers: services that turn a text description into an mp4 clip.

Only Kling AI is set up for now. The "provider" part of the "video" config
section picks it:

    "provider": {"type": "kling", "access_key": "env:KLING_ACCESS_KEY", ...}

To add another service later, write a VideoProvider subclass and register it
with register_video_provider("name", TheClass).
"""
from .base import VideoProvider
from .kling import KlingProvider

PROVIDERS = {"kling": KlingProvider}


def register_video_provider(name, cls):
    PROVIDERS[name] = cls


def create_video_provider(cfg):
    """cfg: {"type": "kling", ...settings for that provider}"""
    cfg = dict(cfg)
    return PROVIDERS[cfg.pop("type", "kling")](**cfg)
