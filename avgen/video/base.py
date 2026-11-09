from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Clip:
    """A finished clip and the details of how it was made.

    Everything except `video` is saved in the scene's `video_scenes` row.
    Details a provider doesn't know are left as None.
    """
    video: bytes                  # the mp4 itself
    job_id: str = None            # the provider's id for the job, handy when contacting their support
    model: str = None             # e.g. "seedance-1-0-pro-fast-251015"
    resolution: str = None        # e.g. "480p"
    ratio: str = None             # aspect ratio, e.g. "9:16"
    seconds: float = None         # clip length as made (can be longer than the scene)
    tokens: int = None            # what the provider billed, in its own units
    cost_usd: float = None        # estimated cost, from the price in the config
    details: dict = field(default_factory=dict)  # the provider's full reply, for anything else


class VideoProvider(ABC):
    name = "unknown"  # saved in `video_scenes.provider` for every clip it makes

    @abstractmethod
    def generate(self, prompt: str, duration: float) -> Clip:
        """Generate a clip for the prompt."""
