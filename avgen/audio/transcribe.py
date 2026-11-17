"""Transcription: when each word of the narration is spoken, for the subtitles.

The audio stage sends each narration here right after it is made and saves
the word timings (`audio_files.words`); the assemble stage turns them into
subtitles.

OpenAI's transcription endpoint (POST /v1/audio/transcriptions) returns word
timings with response_format "verbose_json" and timestamp_granularities
"word". whisper-1 supports this (about $0.006 per minute); pick the model
with "model" in the "captions" config section.
"""
import logging
from dataclasses import dataclass, field

import requests

from ..log import check_response

log = logging.getLogger(__name__)

# Whisper only reads the first ~224 tokens of the prompt (the script, used
# so names and unusual words are spelled the way the script spells them).
MAX_PROMPT_CHARS = 800


@dataclass
class Transcript:
    words: list                   # [{"word": "The", "start": 0.0, "end": 0.18}, ...], seconds into the narration
    model: str = None
    cost_usd: float = None        # estimated, from the price in the config
    details: dict = field(default_factory=dict)


class OpenAITranscriber:
    name = "openai"

    def __init__(self, api_key, model="whisper-1", language=None, base_url="https://api.openai.com/v1",
                 usd_per_minute=None, timeout_seconds=300, **_):
        if not api_key:
            log.error("OpenAI api_key is empty; is OPENAI_API_KEY set?")
        self.api_key = api_key
        self.model = model
        self.language = language  # e.g. "en"; None lets the model detect it
        self.base_url = base_url.rstrip("/")
        self.usd_per_minute = usd_per_minute
        self.timeout_seconds = timeout_seconds

    def transcribe(self, audio, seconds=None, script_text=None):
        """Word timings for an mp3. `seconds` (its length) is only used for the cost estimate."""
        form = {"model": self.model, "response_format": "verbose_json", "timestamp_granularities[]": "word"}
        if self.language:
            form["language"] = self.language
        if script_text:
            form["prompt"] = script_text[:MAX_PROMPT_CHARS]
        log.debug("transcription request: %s", {k: v for k, v in form.items() if k != "prompt"})

        reply = requests.post(f"{self.base_url}/audio/transcriptions", data=form, timeout=self.timeout_seconds,
                              files={"file": ("narration.mp3", audio, "audio/mpeg")},
                              headers={"Authorization": f"Bearer {self.api_key}"})
        check_response(reply, "OpenAI transcription")
        body = reply.json()
        words = [{"word": str(w["word"]).strip(), "start": float(w["start"]), "end": float(w["end"])}
                 for w in body.get("words") or [] if str(w.get("word", "")).strip()]
        if not words:
            raise RuntimeError(f"the transcription has no word timings (model {self.model} may not support them)")

        length = seconds or body.get("duration")
        cost = None
        if self.usd_per_minute is not None and length:
            cost = round(float(length) / 60 * self.usd_per_minute, 6)
        return Transcript(words=words, model=self.model, cost_usd=cost,
                          details={"text": body.get("text"), "language": body.get("language")})
