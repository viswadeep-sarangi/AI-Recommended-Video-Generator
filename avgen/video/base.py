from abc import ABC, abstractmethod


class VideoProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, duration: float) -> bytes:
        """Generate a clip for the prompt and return mp4 bytes."""
