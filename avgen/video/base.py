from abc import ABC, abstractmethod


class VideoProvider(ABC):
    name = "unknown"  # saved in `video_scenes.provider` for every clip it makes

    @abstractmethod
    def generate(self, prompt: str, duration: float) -> bytes:
        """Generate a clip for the prompt and return mp4 bytes."""
