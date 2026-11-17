from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Narration:
    """A finished voice-over and the details of how it was made.

    Everything except `audio` is saved in the script's `audio_files` row.
    Details a provider doesn't know are left as None.
    """
    audio: bytes                  # the mp3 itself
    model: str = None             # e.g. "gpt-4o-mini-tts"
    voice: str = None             # e.g. "alloy"
    characters: int = None        # length of the text that was read out
    seconds: float = None         # length of the audio
    cost_usd: float = None        # estimated cost, from the price in the config
    details: dict = field(default_factory=dict)  # anything else worth keeping (settings sent, file size)


class VoiceProvider(ABC):
    name = "unknown"  # saved in `audio_files.provider` for every narration it makes

    @abstractmethod
    def narrate(self, text: str) -> Narration:
        """Read `text` aloud and return it as an mp3."""


class MusicProvider(ABC):
    name = "unknown"

    @abstractmethod
    def music(self, prompt: str, seconds: float) -> bytes:
        """Background music (mp3) for a video about `prompt`, about `seconds` long."""
