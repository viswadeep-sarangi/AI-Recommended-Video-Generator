import logging
import time

import requests

from ..log import check_response
from .base import LLM

log = logging.getLogger(__name__)


class OpenAILLM(LLM):
    def __init__(self, api_key, model="gpt-4o-mini", base_url="https://api.openai.com/v1", temperature=0.7, **_):
        self.api_key, self.model, self.base_url, self.temperature = api_key, model, base_url.rstrip("/"), temperature
        if not api_key:
            log.error("OpenAI api_key is empty; is OPENAI_API_KEY set?")

    def complete(self, prompt, system=""):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        started = time.monotonic()
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": "Bearer " + self.api_key},
            json={"model": self.model, "messages": messages, "temperature": self.temperature},
            timeout=180,
        )
        check_response(r, f"OpenAI chat completion ({self.model})")
        data = r.json()
        usage = data.get("usage") or {}
        log.info("OpenAI %s replied in %.1fs (tokens: prompt=%s completion=%s)", self.model,
                 time.monotonic() - started, usage.get("prompt_tokens"), usage.get("completion_tokens"))
        return data["choices"][0]["message"]["content"]
