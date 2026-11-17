import requests

from ..log import check_response
from .base import MusicProvider


class StaticMusic(MusicProvider):
    """The same background track every time, downloaded from a configured URL."""
    name = "static"

    def __init__(self, url, **_):
        self.url = url

    def music(self, prompt, seconds):
        reply = requests.get(self.url, timeout=300)
        check_response(reply, "background music download")
        return reply.content
