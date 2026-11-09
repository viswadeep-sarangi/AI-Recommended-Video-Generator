"""Shared naming for files the pipeline saves in Supabase storage.

Every stage stores its files directly inside its own bucket, grouped by the
day (UTC) the script was written, and labelled with the script number and a
short piece of the topic title. Everything for one video therefore lines up
across buckets:

    scripts       2026-11-05/topic_42_the-town-that-vanished_070000Z.txt
    scenes        2026-11-05/script_7_the-town-that-vanished.json
    video-scenes  2026-11-09/script_7_the-town-that-vanished/scene_000.mp4  (*)
    audio         2026-11-05/script_7_the-town-that-vanished/narration.mp3
    final-videos  2026-11-05/script_7_the-town-that-vanished.mp4

(*) The exception: clips are dated the day the script's first clip was made,
since the video stage works through a backlog and may reach a script days
after it was written. The script number in the name still ties them together.
See VideoPipeline._clip_folder.

Storage folders are just part of the path, so they appear automatically with
the first upload.
"""
import re
import unicodedata
from datetime import datetime, timezone

# How much of the topic title goes into file and folder names.
TITLE_CHARS_IN_NAMES = 30


def day_folder(when=None):
    """The YYYY-MM-DD folder for a moment in time (UTC); now if not given."""
    return (when or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%d")


def title_snippet(title, max_chars=TITLE_CHARS_IN_NAMES):
    """Turn a title into a short, storage-safe snippet.

    "The Town That Vanished Overnight!" -> "the-town-that-vanished"
    Accents are stripped and anything other than letters and digits becomes
    a hyphen; the result is cut at a word boundary to at most `max_chars`.
    """
    ascii_title = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode("ascii")
    words = re.sub(r"[^a-z0-9]+", " ", ascii_title.lower()).split()
    snippet = ""
    for word in words:
        candidate = f"{snippet}-{word}" if snippet else word
        if len(candidate) > max_chars:
            break
        snippet = candidate
    if not snippet and words:
        snippet = words[0][:max_chars]  # a single very long first word
    return snippet or "untitled"


def script_location(db, script):
    """(day folder, label) for a script's downstream files.

    The day is when the script was written, so a video's clips, audio and
    final cut share its folder even if they're made on later days.
    e.g. ("2026-11-05", "script_7_the-town-that-vanished")
    """
    return _script_day(script), f"script_{script['id']}_{title_snippet(_topic_title(db, script))}"


def _script_day(script):
    created = script.get("created_at")
    if created:
        try:
            return day_folder(datetime.fromisoformat(str(created).replace("Z", "+00:00")))
        except ValueError:
            pass
    return day_folder()


def _topic_title(db, script):
    if script.get("topic_id") is None:
        return ""
    rows = db.select("topics", {"id": f"eq.{script['topic_id']}"})
    return rows[0].get("title", "") if rows else ""
