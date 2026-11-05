"""Kling AI text-to-video, through Kling's official API.

How a clip is made:
  1. send the scene's description to POST /v1/videos/text2video, which
     starts a job and returns its task id,
  2. ask GET /v1/videos/text2video/{task_id} every `poll_seconds` until the
     job's status is "succeed" (or "failed", or `timeout_seconds` pass),
  3. download the mp4 from the URL in the finished job.

Every request is signed with a short-lived token made from your API access
key and secret key (from the API keys page of the Kling developer console).
API credits are bought separately from the Kling app's subscription plans.
"""
import base64
import hashlib
import hmac
import json
import logging
import time

import requests

from ..log import check_response
from .base import VideoProvider

log = logging.getLogger(__name__)

# Kling's job statuses.
STATUS_DONE = "succeed"
STATUS_FAILED = "failed"

# Each signed token is valid this long; a fresh one is made for every request.
TOKEN_SECONDS = 30 * 60


class KlingProvider(VideoProvider):
    name = "kling"

    def __init__(self, access_key, secret_key, base_url="https://api-singapore.klingai.com",
                 model="kling-v2-1-master", mode="std", aspect_ratio="9:16", clip_seconds=(5, 10),
                 negative_prompt="", poll_seconds=10, timeout_seconds=900, **_):
        if not access_key or not secret_key:
            log.error("Kling access_key or secret_key is empty; are KLING_ACCESS_KEY and KLING_SECRET_KEY set?")
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.model = model                  # which Kling model makes the clip
        self.mode = mode                    # "std" (cheaper) or "pro" (higher quality)
        self.aspect_ratio = aspect_ratio    # "9:16" is vertical, for Shorts/Reels/TikTok
        self.clip_seconds = clip_seconds    # the clip lengths Kling offers
        self.negative_prompt = negative_prompt  # things the clip should avoid
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt, duration):
        """Make a clip for `prompt` and return its mp4 bytes.

        Kling only makes clips of certain lengths, so the closest one to
        `duration` is asked for (e.g. 3s -> 5s, 8s -> 10s).
        """
        seconds = min(self.clip_seconds, key=lambda option: abs(option - duration))
        task_id = self._start_job(prompt, seconds)
        video_url = self._wait_for_job(task_id)
        log.info("    downloading the finished clip")
        response = requests.get(video_url, timeout=600)
        check_response(response, "download of the Kling clip")
        return response.content

    # --- The three steps ---------------------------------------------------

    def _start_job(self, prompt, seconds):
        body = {
            "model_name": self.model,
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "mode": self.mode,
            "aspect_ratio": self.aspect_ratio,
            "duration": str(seconds),
        }
        log.debug("Kling request: %s", body)
        response = requests.post(f"{self.base_url}/v1/videos/text2video", json=body,
                                 headers=self._headers(), timeout=120)
        task_id = _data(response, "Kling job submission")["task_id"]
        log.info("    Kling job %s started (%s, %s, %ss, %s)", task_id, self.model, self.mode, seconds,
                 self.aspect_ratio)
        return task_id

    def _wait_for_job(self, task_id):
        """Wait until the job finishes; returns the finished clip's URL."""
        started = time.monotonic()
        last_status = None
        while time.monotonic() - started < self.timeout_seconds:
            response = requests.get(f"{self.base_url}/v1/videos/text2video/{task_id}",
                                    headers=self._headers(), timeout=60)
            job = _data(response, f"Kling status check of job {task_id}")
            status = job.get("task_status")
            if status != last_status:
                log.info("    Kling job %s: %s (after %.0fs)", task_id, status, time.monotonic() - started)
                last_status = status
            if status == STATUS_DONE:
                return job["task_result"]["videos"][0]["url"]
            if status == STATUS_FAILED:
                raise RuntimeError(f"Kling job {task_id} failed: {job.get('task_status_msg') or job}")
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"Kling job {task_id} didn't finish within {self.timeout_seconds}s "
                           f"(last status: {last_status})")

    # --- Signing requests --------------------------------------------------

    def _headers(self):
        return {"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"}

    def _token(self):
        """A JWT signed with the secret key (HS256), as Kling's API requires."""
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        claims = {"iss": self.access_key, "exp": now + TOKEN_SECONDS, "nbf": now - 5}
        unsigned = f"{_b64(json.dumps(header))}.{_b64(json.dumps(claims))}"
        signature = hmac.new(self.secret_key.encode(), unsigned.encode(), hashlib.sha256).digest()
        return f"{unsigned}.{_b64(signature)}"


def _data(response, what):
    """The "data" part of a Kling reply; raises with Kling's message if it reports an error."""
    check_response(response, what)
    reply = response.json()
    if reply.get("code") not in (0, None):
        raise RuntimeError(f"{what} failed: Kling error {reply.get('code')}: {reply.get('message')}")
    return reply["data"]


def _b64(data):
    """Base64url without padding, as JWTs use."""
    if isinstance(data, str):
        data = data.encode()
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()
