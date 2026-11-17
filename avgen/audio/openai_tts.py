"""OpenAI text-to-speech: POST /v1/audio/speech returns the narration as an mp3.

The speech endpoint doesn't say what a request cost, so the cost is
estimated from the price in the config: per minute of audio for
gpt-4o-mini-tts (about $0.015/min), or per million characters for tts-1
($15) and tts-1-hd ($30). Check the real charge on OpenAI's usage page.

Note: OpenAI has announced that tts-1, tts-1-hd and gpt-4o-mini-tts will be
removed from the API on 6 January 2027.
"""
import io
import logging

import requests

from ..log import check_response
from .base import Narration, VoiceProvider

log = logging.getLogger(__name__)

# The speech endpoint rejects longer input.
MAX_INPUT_CHARS = 4096


class OpenAITTS(VoiceProvider):
    name = "openai_tts"

    def __init__(self, api_key, model="gpt-4o-mini-tts", voice="alloy", instructions=None, speed=None,
                 base_url="https://api.openai.com/v1", usd_per_minute=None, usd_per_million_characters=None,
                 timeout_seconds=300, **_):
        if not api_key:
            log.error("OpenAI api_key is empty; is OPENAI_API_KEY set?")
        self.api_key = api_key
        self.model = model
        self.voice = voice                  # e.g. alloy, ash, coral, echo, fable, nova, onyx, sage, shimmer
        self.instructions = instructions    # tone of voice, e.g. "Energetic, curious storyteller" (gpt-4o-mini-tts only)
        self.speed = speed                  # 0.25 to 4.0; None = the default (1.0)
        self.base_url = base_url.rstrip("/")
        self.usd_per_minute = usd_per_minute
        self.usd_per_million_characters = usd_per_million_characters
        self.timeout_seconds = timeout_seconds

    def narrate(self, text):
        text = text.strip()
        if not text:
            raise ValueError("the script is empty; nothing to narrate")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"the script is {len(text)} characters; OpenAI TTS reads at most {MAX_INPUT_CHARS}")

        request = {"model": self.model, "voice": self.voice, "input": text, "response_format": "mp3"}
        if self.instructions:
            request["instructions"] = self.instructions
        if self.speed:
            request["speed"] = self.speed
        log.debug("OpenAI TTS request: %s", {k: v for k, v in request.items() if k != "input"})

        reply = requests.post(f"{self.base_url}/audio/speech", json=request, timeout=self.timeout_seconds,
                              headers={"Authorization": f"Bearer {self.api_key}"})
        check_response(reply, "OpenAI text-to-speech")
        audio = reply.content
        seconds = mp3_seconds(audio)
        return Narration(
            audio=audio,
            model=self.model,
            voice=self.voice,
            characters=len(text),
            seconds=seconds,
            cost_usd=self._cost(len(text), seconds),
            details={k: v for k, v in request.items() if k != "input"} | {"bytes": len(audio)},
        )

    def _cost(self, characters, seconds):
        if self.usd_per_million_characters is not None:
            return round(characters * self.usd_per_million_characters / 1_000_000, 6)
        if self.usd_per_minute is not None and seconds is not None:
            return round(seconds / 60 * self.usd_per_minute, 6)
        return None


def mp3_seconds(audio):
    """Length of an mp3 in seconds, or None if it can't be read."""
    try:
        from mutagen.mp3 import MP3
        return round(MP3(io.BytesIO(audio)).info.length, 2)
    except Exception as error:  # mutagen missing or an unreadable file: the length is just unknown
        log.warning("couldn't read the narration's length (%s); its cost will be unknown", error)
        return None
