"""Video stage: turns each scene into a short video clip with Seedance.

The scene generator saves every script as a "scene file": a list of scenes,
each with a description of what to show (`visual_prompt`) and how long it
lasts. This stage picks the oldest scene files that still have scenes without
a clip (at most `max_scene_files_per_run` per run, 1 by default) and, for each
of those scenes:
  1. asks Seedance for a clip of the scene's description,
  2. saves the clip in the `videos` bucket, in a folder named after the
     script and dated the day its first clip was made, e.g.
         2026-11-09/script_7_the-town-that-vanished/scene_000.mp4
  3. records it in the `video_scenes` table: "done" with how the clip was
     made (model, resolution, aspect ratio, length, tokens, estimated
     cost in USD and the provider's full reply), or "failed" with the error.

Each scene has exactly one row in `video_scenes`, which counts its
`attempts`. A failed scene is tried again on the next run, up to
`max_attempts_per_scene` (3 by default) attempts in total. After that it is
given up on, so a prompt that can never work doesn't keep costing credits or
hold up the scripts behind it. Given-up scenes are listed as warnings on every
run. To try them again, raise "max_attempts_per_scene" in the config, or set
the scene's `attempts` back to 0 in the table.

Once every scene of a script has a clip, the assemble stage joins the clips
with the narration and music into the final video.

Both limits live in the "video" section of the config:

    "max_scene_files_per_run": 1,
    "max_attempts_per_scene": 3,
"""
import json
import logging
import time
from datetime import datetime, timezone

from .storage_paths import day_folder, script_location

log = logging.getLogger(__name__)

# How many scene files (= scripts) to make clips for in one run, when the
# config doesn't say. Each scene file is usually 6-15 clips, and every clip
# costs credits, so this starts small. Raise it with "max_scene_files_per_run".
DEFAULT_MAX_SCENE_FILES_PER_RUN = 1

# How many times to try one scene before giving up on it, when the config
# doesn't say. Change it with "max_attempts_per_scene".
DEFAULT_MAX_ATTEMPTS_PER_SCENE = 3

# How much of a scene's description to show on one INFO log line.
PREVIEW_CHARS = 80

# Longest error message stored in the `video_scenes` table.
MAX_ERROR_CHARS = 500


class VideoGenerationError(RuntimeError):
    """Every scene tried in this run failed."""


class VideoPipeline:
    def __init__(self, db, provider, cfg=None):
        self.db = db              # Supabase: tables and file storage
        self.provider = provider  # the video service that makes the clips (Seedance)
        self.cfg = cfg or {}      # the "video" section of the config

        self.max_attempts = self.cfg.get("max_attempts_per_scene", DEFAULT_MAX_ATTEMPTS_PER_SCENE)

    # --- Main entry point -------------------------------------------------

    def run(self, limit=None):
        """Make clips for the oldest scene files that still have scenes to try.

        `limit` is how many scene files to work on; it defaults to
        "max_scene_files_per_run" from the config, then to
        DEFAULT_MAX_SCENE_FILES_PER_RUN.

        A scene that fails is logged and recorded; the others still run.
        Raises VideoGenerationError only if every scene tried failed.
        Returns the `video_scenes` rows written this run (done and failed).
        """
        how_many = limit or self.cfg.get("max_scene_files_per_run", DEFAULT_MAX_SCENE_FILES_PER_RUN)
        log.info("settings: provider=%s max_scene_files_per_run=%s max_attempts_per_scene=%s",
                 self.provider.name, how_many, self.max_attempts)

        waiting = self._scene_files_waiting_for_clips()
        if not waiting:
            log.info("no scene has a clip left to try; nothing to generate")
            return []

        batch = waiting[:how_many]
        log.info("%d scene file(s) have clips left to make; working on %d this run:", len(waiting), len(batch))
        for scene_file, clips_done in batch:
            log.info("  scene file #%s (script #%s): %d of %s clip(s) done | %s", scene_file["id"],
                     scene_file["script_id"], clips_done, scene_file.get("scene_count", "?"),
                     scene_file["file_path"])
        left_over = waiting[how_many:]
        if left_over:
            log.info("%d scene file(s) left for later runs (max_scene_files_per_run=%d): #%s", len(left_over),
                     how_many, ", #".join(str(scene_file["id"]) for scene_file, _ in left_over))

        rows = []
        for number, (scene_file, _) in enumerate(batch, start=1):
            log.info("[%d/%d] scene file #%s (script #%s)", number, len(batch), scene_file["id"],
                     scene_file["script_id"])
            rows += self.generate_clips(scene_file)

        self._log_summary(rows)
        failed = [row for row in rows if row["status"] == "failed"]
        if failed and len(failed) == len(rows):
            raise VideoGenerationError(f"all {len(failed)} scene(s) failed; see the errors above")
        return rows

    def generate_clips(self, scene_file):
        """Make, save and record a clip for every scene in this file still worth trying."""
        script = self._script_for(scene_file)
        rows = self._rows_by_scene(scene_file)  # scene index -> its `video_scenes` row, if any
        folder = self._clip_folder(script, rows)
        log.info("  clips go in: %s/", folder)

        scenes = self._download_scenes(scene_file)
        to_try = [scene for scene in scenes if self._worth_trying(rows.get(scene["index"]))]
        done = sum(1 for row in rows.values() if row["status"] == "done")
        log.info("  %d scene(s) in the file: %d done, %d given up, %d to try now",
                 len(scenes), done, len(scenes) - done - len(to_try), len(to_try))

        results = []
        for number, scene in enumerate(to_try, start=1):
            previous = rows.get(scene["index"])
            results.append(self._make_one_clip(scene_file, scene, previous, folder, number, len(to_try)))
        return results

    # --- Making one clip ---------------------------------------------------

    def _make_one_clip(self, scene_file, scene, previous, folder, number, total):
        """Generate, upload and record the clip for one scene.

        `previous` is the scene's existing `video_scenes` row (from an
        earlier failed attempt), or None the first time. Never raises: a
        failure is logged and recorded as "failed".
        """
        attempt = (previous["attempts"] if previous else 0) + 1
        # e.g. 2026-11-05/script_7_the-town-that-vanished/scene_000.mp4
        file_path = f"{folder}/scene_{scene['index']:03d}.mp4"
        log.info("  (%d/%d) scene %d, attempt %d/%d: %.1fs | %s", number, total, scene["index"], attempt,
                 self.max_attempts, scene["duration"], _preview(scene["visual_prompt"]))
        log.debug("  scene %d full visual prompt: %s", scene["index"], scene["visual_prompt"])

        started = time.monotonic()
        try:
            clip = self.provider.generate(scene["visual_prompt"], scene["duration"])
            log.info("    clip made in %.1fs: %s, %s %s, %ss, %s tokens, %s", time.monotonic() - started,
                     _size(len(clip.video)), clip.resolution, clip.ratio, clip.seconds, clip.tokens,
                     _dollars(clip.cost_usd))
            self.db.upload("videos", file_path, clip.video, "video/mp4")
            log.info("    saved to the videos bucket: %s", file_path)
            result = {"status": "done", "file_path": file_path, "error": None, **_clip_details(clip)}
        except Exception as error:
            if attempt < self.max_attempts:
                next_step = "the next run will try it again"
            else:
                next_step = "that was the last attempt, so it is given up on"
            log.exception("    scene %d failed after %.1fs (attempt %d/%d): %s. Moving on; %s.", scene["index"],
                          time.monotonic() - started, attempt, self.max_attempts, error, next_step)
            result = {"status": "failed", "file_path": None, "error": str(error)[:MAX_ERROR_CHARS]}

        return self._record_attempt(scene_file, scene, previous, attempt, result)

    def _worth_trying(self, row):
        """True if a scene with this `video_scenes` row (None = never tried) should be tried now."""
        if row is None:
            return True
        return row["status"] != "done" and row["attempts"] < self.max_attempts

    # --- Logging ------------------------------------------------------------

    def _log_summary(self, rows):
        done = [row for row in rows if row["status"] == "done"]
        failed = [row for row in rows if row["status"] == "failed"]
        costs = [row["cost_usd"] for row in done if row.get("cost_usd") is not None]
        log.info("done: made %d of %d clip(s), %d failed; estimated cost this run: %s", len(done), len(rows),
                 len(failed), _dollars(sum(costs)) if costs else "unknown")
        for row in done:
            log.info("  ok     script #%s scene %s -> %s (%s)", row["script_id"], row["scene_index"],
                     row["file_path"], _dollars(row.get("cost_usd")))
        for row in failed:
            log.error("  FAILED script #%s scene %s (attempt %s/%s): %s", row["script_id"], row["scene_index"],
                      row["attempts"], self.max_attempts, row.get("error"))

    def _warn_about_given_up(self, given_up):
        """One warning per scene that ran out of attempts, so they aren't forgotten."""
        for row in given_up:
            log.warning("script #%s scene %s was given up on after %s attempt(s), so that script can't be "
                        "assembled. Last error: %s. To try again, raise max_attempts_per_scene (now %s) or set "
                        "attempts to 0 for video_scenes row #%s.", row["script_id"], row["scene_index"],
                        row["attempts"], row.get("error"), self.max_attempts, row["id"])

    # --- Reading and writing Supabase ---------------------------------------

    def _scene_files_waiting_for_clips(self):
        """(scene file, clips done) for every scene file with scenes still to try, oldest first.

        A scene file is finished when each of its scenes is either done or
        given up on. This uses the `scene_count` saved with each scene file,
        so no file needs downloading just to find out.
        """
        settled, done, given_up = {}, {}, []
        for row in self.db.select("video_scenes"):
            if row["status"] == "done":
                done[row["scene_file_id"]] = done.get(row["scene_file_id"], 0) + 1
            elif not self._worth_trying(row):
                given_up.append(row)
            else:
                continue
            settled[row["scene_file_id"]] = settled.get(row["scene_file_id"], 0) + 1
        self._warn_about_given_up(given_up)

        scene_files = self.db.select("scene_files", order="id.asc")
        waiting = [(scene_file, done.get(scene_file["id"], 0)) for scene_file in scene_files
                   if scene_file.get("scene_count") is None
                   or settled.get(scene_file["id"], 0) < scene_file["scene_count"]]
        log.info("found %d scene file(s) in total, %d finished", len(scene_files), len(scene_files) - len(waiting))
        return waiting

    def _rows_by_scene(self, scene_file):
        """This file's `video_scenes` rows, keyed by scene index."""
        rows = self.db.select("video_scenes", {"scene_file_id": f"eq.{scene_file['id']}"})
        return {row["scene_index"]: row for row in rows}

    def _record_attempt(self, scene_file, scene, previous, attempt, result):
        """Write the outcome to the scene's one `video_scenes` row (made on its first attempt)."""
        now = datetime.now(timezone.utc).isoformat()
        values = {**result, "provider": self.provider.name, "attempts": attempt, "updated_at": now}
        if previous:
            return self.db.update("video_scenes", {"id": f"eq.{previous['id']}"}, values)[0]
        return self.db.insert("video_scenes", {
            **values,
            "scene_file_id": scene_file["id"],
            "scene_index": scene["index"],
            "script_id": scene_file["script_id"],
            "created_at": now,
        })

    def _download_scenes(self, scene_file):
        """The list of scenes the scene generator saved for this script."""
        log.info("  downloading scenes: %s", scene_file["file_path"])
        return json.loads(self.db.download("scenes", scene_file["file_path"]))["scenes"]

    def _script_for(self, scene_file):
        """The `scripts` row this scene file was made from (just the id if it's gone)."""
        rows = self.db.select("scripts", {"id": f"eq.{scene_file['script_id']}"})
        if not rows:
            log.warning("script #%s not found; naming its clip folder without a title", scene_file["script_id"])
        return rows[0] if rows else {"id": scene_file["script_id"]}

    def _clip_folder(self, script, rows):
        """Folder for one script's clips, e.g. 2026-11-09/script_7_the-town-that-vanished.

        The date is the day the script's first clip was made (UTC), not the
        day the script was written: with a backlog, a script written on
        Nov 5 may only get its clips on Nov 9, and they belong in the Nov 9
        folder. Once a script has a clip, later clips for it (such as
        retries on another day) go in that same folder, so one script's
        clips are never split across days.

        `rows` is the script's existing `video_scenes` rows, by scene index.
        See storage_paths.py for how the other stages name files.
        """
        for row in rows.values():
            if row.get("file_path"):
                return row["file_path"].rsplit("/", 1)[0]
        _, label = script_location(self.db, script)
        return f"{day_folder()}/{label}"


# --- Small helpers -----------------------------------------------------------

def _preview(text, limit=PREVIEW_CHARS):
    """The start of `text` on one line, for log messages."""
    one_line = " ".join(str(text or "").split())
    return one_line if len(one_line) <= limit else one_line[:limit - 3] + "..."


def _clip_details(clip):
    """The `video_scenes` columns that describe how a clip was made."""
    return {
        "job_id": clip.job_id,
        "model": clip.model,
        "resolution": clip.resolution,
        "ratio": clip.ratio,
        "clip_seconds": clip.seconds,
        "tokens": clip.tokens,
        "cost_usd": clip.cost_usd,
        "details": clip.details,
    }


def _dollars(amount):
    """e.g. 0.049005 -> "~$0.049"; "cost unknown" if None."""
    return "cost unknown" if amount is None else f"~${amount:.3f}"


def _size(n_bytes):
    """e.g. 2457600 -> "2.3 MB"."""
    return f"{n_bytes / 1024 / 1024:.1f} MB" if n_bytes >= 1024 * 1024 else f"{n_bytes / 1024:.0f} KB"
