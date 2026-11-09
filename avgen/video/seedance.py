"""Seedance (ByteDance) text-to-video, through the BytePlus ModelArk API.

How a clip is made:
  1. send the scene's description to POST /contents/generations/tasks,
     which starts a job and returns its task id,
  2. ask GET /contents/generations/tasks/{id} every `poll_seconds` until the
     job's status is "succeeded" (or "failed", or `timeout_seconds` pass),
  3. download the mp4 from the job's `video_url`. The link expires after
     24 hours, so the pipeline saves the clip to its own storage right away.

Settings such as resolution and aspect ratio are passed the way BytePlus's
own examples do: as flags at the end of the prompt, e.g.
    "A foggy village at dawn --ratio 9:16 --resolution 480p --duration 5"

Requests are signed with an API key from the ModelArk console (API keys page).
Each model must be activated in the console (Model activation) before use.
"""
import logging
import math
import time

import requests

from ..log import check_response
from .base import Clip, VideoProvider

log = logging.getLogger(__name__)

# Seedance's job statuses ("queued" and "running" mean "not finished yet").
STATUS_DONE = "succeeded"
STATUS_GAVE_UP = ("failed", "expired", "cancelled")


class SeedanceProvider(VideoProvider):
    name = "seedance"

    def __init__(self, api_key, model="seedance-1-0-pro-fast-251015",
                 base_url="https://ark.ap-southeast.bytepluses.com/api/v3", resolution="480p", ratio="9:16",
                 min_seconds=2, max_seconds=12, camera_fixed=False, usd_per_million_tokens=None,
                 poll_seconds=5, timeout_seconds=600, **_):
        if not api_key:
            log.error("Seedance api_key is empty; is ARK_API_KEY set?")
        self.api_key = api_key
        self.model = model              # model id from the ModelArk console
        self.base_url = base_url.rstrip("/")
        self.resolution = resolution    # "480p" (cheapest), "720p" or "1080p"
        self.ratio = ratio              # "9:16" is vertical, for Shorts/Reels/TikTok
        self.min_seconds = min_seconds  # the shortest and longest clip the model accepts
        self.max_seconds = max_seconds
        self.camera_fixed = camera_fixed  # True keeps the camera still
        self.usd_per_million_tokens = usd_per_million_tokens  # price, used to estimate each clip's cost
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt, duration):
        """Make a clip for `prompt`; returns it as a Clip with its details.

        Seedance makes clips of whole seconds, so `duration` is rounded up
        (e.g. 3.5s -> 4s) and kept between min_seconds and max_seconds.
        """
        seconds = min(max(math.ceil(duration), self.min_seconds), self.max_seconds)
        task_id = self._start_job(prompt, seconds)
        job = self._wait_for_job(task_id)
        log.info("    downloading the finished clip")
        response = requests.get(job["content"]["video_url"], timeout=600)
        check_response(response, "download of the Seedance clip")

        tokens = (job.get("usage") or {}).get("total_tokens")
        return Clip(
            video=response.content,
            job_id=task_id,
            model=job.get("model", self.model),
            resolution=job.get("resolution", self.resolution),
            ratio=job.get("ratio", self.ratio),
            seconds=job.get("duration", seconds),
            tokens=tokens,
            cost_usd=self._cost(tokens),
            details=job,
        )

    # --- The three steps ---------------------------------------------------

    def _start_job(self, prompt, seconds):
        flags = (f"--ratio {self.ratio} --resolution {self.resolution} --duration {seconds} "
                 f"--camerafixed {str(self.camera_fixed).lower()}")
        body = {"model": self.model, "content": [{"type": "text", "text": f"{prompt} {flags}"}]}
        log.debug("Seedance request: %s", body)
        response = requests.post(f"{self.base_url}/contents/generations/tasks", json=body,
                                 headers=self._headers(), timeout=120)
        check_response(response, "Seedance job submission")
        task_id = response.json()["id"]
        log.info("    Seedance job %s started (%s, %s, %s, %ss)", task_id, self.model, self.resolution, self.ratio,
                 seconds)
        return task_id

    def _wait_for_job(self, task_id):
        """Wait until the job finishes; returns Seedance's reply for the finished job."""
        started = time.monotonic()
        last_status = None
        while time.monotonic() - started < self.timeout_seconds:
            response = requests.get(f"{self.base_url}/contents/generations/tasks/{task_id}",
                                    headers=self._headers(), timeout=60)
            check_response(response, f"Seedance status check of job {task_id}")
            job = response.json()
            status = job.get("status")
            if status != last_status:
                log.info("    Seedance job %s: %s (after %.0fs)", task_id, status, time.monotonic() - started)
                last_status = status
            if status == STATUS_DONE:
                return job
            if status in STATUS_GAVE_UP:
                error = job.get("error") or {}
                raise RuntimeError(f"Seedance job {task_id} {status}: "
                                   f"{error.get('code', '')} {error.get('message', '')}".strip())
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"Seedance job {task_id} didn't finish within {self.timeout_seconds}s "
                           f"(last status: {last_status})")

    # --- Helpers -------------------------------------------------------------

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _cost(self, tokens):
        """Estimated cost in USD from the configured price, e.g. 49005 tokens -> 0.049005.

        None if the token count or the price is unknown. The real charge is
        on the BytePlus console's Usage and Billing pages.
        """
        if tokens is None or not self.usd_per_million_tokens:
            return None
        return round(tokens / 1_000_000 * self.usd_per_million_tokens, 6)
