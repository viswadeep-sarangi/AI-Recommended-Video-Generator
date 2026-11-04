"""Logging setup and HTTP error helpers shared by all stages.

Level comes from --log-level / -v on the CLI or the AVGEN_LOG_LEVEL env var
(default INFO). When running inside GitHub Actions, warnings and errors are
also emitted as workflow annotations so they show up on the run summary page.
"""
import logging
import os
import sys
from contextlib import contextmanager

import requests

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
MAX_BODY = 2000


def in_github_actions():
    return os.environ.get("GITHUB_ACTIONS") == "true"


class _GitHubAnnotationHandler(logging.Handler):
    """Mirror WARNING/ERROR records as ::warning:: / ::error:: annotations."""

    def __init__(self):
        super().__init__(level=logging.WARNING)

    def emit(self, record):
        try:
            kind = "error" if record.levelno >= logging.ERROR else "warning"
            msg = record.getMessage()
            # Annotation messages must be single-line; GitHub decodes %0A back to newlines.
            msg = msg.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            sys.stdout.write(f"::{kind} title={record.name}::{msg}\n")
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


def setup_logging(level=None):
    level = (level or os.environ.get("AVGEN_LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT, "%H:%M:%S"))
    root.addHandler(handler)
    if in_github_actions():
        root.addHandler(_GitHubAnnotationHandler())
    root.setLevel(level)
    # Our own DEBUG output is useful; urllib3's connection chatter is not.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@contextmanager
def log_group(title):
    """Collapsible section in GitHub Actions logs; a plain banner elsewhere."""
    if in_github_actions():
        print(f"::group::{title}", flush=True)
    else:
        logging.getLogger("avgen").info("=== %s ===", title)
    try:
        yield
    finally:
        if in_github_actions():
            print("::endgroup::", flush=True)


def check_response(r, what):
    """Like r.raise_for_status(), but the error carries the response body.

    `what` describes the request (e.g. "Supabase upsert into youtube_videos").
    The URL is deliberately left out: for some APIs (YouTube) it contains the key.
    """
    try:
        r.raise_for_status()
    except requests.HTTPError:
        body = (getattr(r, "text", "") or "").strip()
        if len(body) > MAX_BODY:
            body = body[:MAX_BODY] + "...(truncated)"
        status = getattr(r, "status_code", "?")
        # `from None`: the original exception's message contains the full URL.
        raise requests.HTTPError(f"{what} failed: HTTP {status}: {body or '(empty body)'}", response=r) from None
