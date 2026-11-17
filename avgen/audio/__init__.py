"""Audio providers: a voice that reads the script aloud, and optional background music.

The "audio" config section picks them:

    "voice": {"provider": "openai_tts", "api_key": "env:OPENAI_API_KEY", ...},
    "music": {"provider": "none"},
    "captions": {"enabled": true, "provider": "openai", "model": "whisper-1", ...}

To add another service, write a VoiceProvider (or MusicProvider) subclass and
register it with register_voice("name", TheClass) (or register_music).
"""
from .base import MusicProvider, Narration, VoiceProvider
from .music import StaticMusic
from .openai_tts import OpenAITTS
from .transcribe import OpenAITranscriber, Transcript

VOICE = {"openai_tts": OpenAITTS}
MUSIC = {"static": StaticMusic}
TRANSCRIBERS = {"openai": OpenAITranscriber}


def register_voice(name, cls):
    VOICE[name] = cls


def register_music(name, cls):
    MUSIC[name] = cls


def register_transcriber(name, cls):
    TRANSCRIBERS[name] = cls


def create_voice(cfg):
    """cfg: {"provider": "openai_tts", ...settings for that voice}"""
    cfg = dict(cfg)
    return VOICE[cfg.pop("provider", "openai_tts")](**cfg)


def create_music(cfg):
    """cfg: {"provider": "static", "url": ...}; None or provider "none" means no music."""
    cfg = dict(cfg or {})
    provider = cfg.pop("provider", "none")
    return None if provider == "none" else MUSIC[provider](**cfg)


def create_transcriber(cfg):
    """cfg: {"enabled": true, "provider": "openai", ...}; None, or enabled false, means no subtitles timing."""
    cfg = dict(cfg or {})
    if not cfg.pop("enabled", False):
        return None
    return TRANSCRIBERS[cfg.pop("provider", "openai")](**cfg)
