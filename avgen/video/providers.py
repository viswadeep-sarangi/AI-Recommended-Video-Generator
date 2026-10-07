import time

import requests

from .base import VideoProvider


class SoraProvider(VideoProvider):
    """OpenAI Videos API."""

    def __init__(self, api_key, model="sora-2", size="720x1280", base_url="https://api.openai.com/v1",
                 poll_seconds=10, timeout_seconds=900, **_):
        self.h = {"Authorization": "Bearer " + api_key}
        self.model, self.size, self.base = model, size, base_url.rstrip("/")
        self.poll, self.timeout = poll_seconds, timeout_seconds

    def generate(self, prompt, duration):
        seconds = str(min([4, 8, 12], key=lambda x: abs(x - duration)))
        r = requests.post(f"{self.base}/videos", headers=self.h, timeout=120,
                          data={"model": self.model, "prompt": prompt, "seconds": seconds, "size": self.size})
        r.raise_for_status()
        vid = r.json()["id"]
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            s = requests.get(f"{self.base}/videos/{vid}", headers=self.h, timeout=60)
            s.raise_for_status()
            status = s.json().get("status")
            if status == "completed":
                c = requests.get(f"{self.base}/videos/{vid}/content", headers=self.h, timeout=600)
                c.raise_for_status()
                return c.content
            if status == "failed":
                raise RuntimeError(f"Sora job failed: {s.json()}")
            time.sleep(self.poll)
        raise TimeoutError("Sora job timed out")


def _dig(obj, dotted):
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else obj[part]
    return obj


class GenericHTTPProvider(VideoProvider):
    """Async submit/poll HTTP API (e.g. Kling gateways), fully described in JSON config.

    Config: submit_url, status_url ("{id}" placeholder), headers, body_template
    (strings may contain {prompt} and {duration}), id_path, status_path,
    done_values, failed_values, result_url_path (dotted paths into the JSON).
    """

    def __init__(self, submit_url, status_url, id_path, status_path, result_url_path, headers=None,
                 body_template=None, done_values=("succeed", "completed"), failed_values=("failed",),
                 poll_seconds=10, timeout_seconds=900, **_):
        self.__dict__.update(locals())

    def _fill(self, v, prompt, duration):
        if isinstance(v, str):
            return v.replace("{prompt}", prompt).replace("{duration}", str(int(round(duration))))
        if isinstance(v, dict):
            return {k: self._fill(x, prompt, duration) for k, x in v.items()}
        return v

    def generate(self, prompt, duration):
        body = self._fill(self.body_template or {"prompt": "{prompt}", "duration": "{duration}"}, prompt, duration)
        r = requests.post(self.submit_url, json=body, headers=self.headers or {}, timeout=120)
        r.raise_for_status()
        job = _dig(r.json(), self.id_path)
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            s = requests.get(self.status_url.replace("{id}", str(job)), headers=self.headers or {}, timeout=60)
            s.raise_for_status()
            data = s.json()
            status = _dig(data, self.status_path)
            if status in self.done_values:
                c = requests.get(_dig(data, self.result_url_path), timeout=600)
                c.raise_for_status()
                return c.content
            if status in self.failed_values:
                raise RuntimeError(f"video job failed: {data}")
            time.sleep(self.poll_seconds)
        raise TimeoutError("video job timed out")
