import requests

from .base import LLM


class OpenAILLM(LLM):
    def __init__(self, api_key, model="gpt-4o-mini", base_url="https://api.openai.com/v1", temperature=0.7, **_):
        self.api_key, self.model, self.base_url, self.temperature = api_key, model, base_url.rstrip("/"), temperature

    def complete(self, prompt, system=""):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": "Bearer " + self.api_key},
            json={"model": self.model, "messages": messages, "temperature": self.temperature},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
