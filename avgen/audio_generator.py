"""Audio stage: reads each script aloud (and adds background music, if set up).

This stage picks the oldest scripts that still need a narration (at most
`max_scripts_per_run` per run, 10 by default) and, for each one:
  1. sends the script's text to the voice service (OpenAI text-to-speech),
  2. saves the mp3 in the `audio` bucket, in a folder named after the
     script and dated the day the narration was made, e.g.
         2026-11-17/script_7_the-town-that-vanished/narration.mp3
     (plus music.mp3 next to it, if music is set up),
  3. records it in the `audio_files` table: "done" with how it was made
     (model, voice, characters, length, estimated cost in USD), or
     "failed" with the error.

By default (`wait_for_clips`: true) a script is only narrated once all its
clips are made: until then it can't be assembled anyway, and its narration
lands in the same date folder as (or a later one than) its clips.

Each script has exactly one row in `audio_files`, which counts its
`attempts`. A script that fails doesn't stop the others; it is tried again
on the next run, up to `max_attempts_per_script` (3 by default) attempts in
total, then given up on and listed as a warning on every run. To try it
again, raise "max_attempts_per_script" in the config, or set the row's
`attempts` back to 0 in the table.

The limits live in the "audio" section of the config:

    "max_scripts_per_run": 10,
    "max_attempts_per_script": 3,
    "wait_for_clips": true,
"""
import logging
import time
from datetime import datetime, timezone

from .storage_paths import day_folder, script_location

log = logging.getLogger(__name__)

# How many scripts to narrate in one run, when the config doesn't say.
# A narration costs about a cent, so this can be generous. Change it with
# "max_scripts_per_run".
DEFAULT_MAX_SCRIPTS_PER_RUN = 10

# How many times to try one script before giving up on it, when the config
# doesn't say. Change it with "max_attempts_per_script".
DEFAULT_MAX_ATTEMPTS_PER_SCRIPT = 3

# How much of a script to show on one INFO log line.
PREVIEW_CHARS = 80

# Longest error message stored in the `audio_files` table.
MAX_ERROR_CHARS = 500


class AudioGenerationError(RuntimeError):
    """Every script tried in this run failed."""


class AudioGenerator:
    def __init__(self, db, voice, music=None, cfg=None):
        self.db = db          # Supabase: tables and file storage
        self.voice = voice    # the service that reads the script aloud (OpenAI TTS)
        self.music = music    # background music, or None for none
        self.cfg = cfg or {}  # the "audio" section of the config

        self.max_attempts = self.cfg.get("max_attempts_per_script", DEFAULT_MAX_ATTEMPTS_PER_SCRIPT)
        self.wait_for_clips = self.cfg.get("wait_for_clips", True)

    # --- Main entry point -------------------------------------------------

    def run(self, limit=None):
        """Narrate the oldest scripts that still need it.

        `limit` is how many scripts to narrate; it defaults to
        "max_scripts_per_run" from the config, then to
        DEFAULT_MAX_SCRIPTS_PER_RUN.

        A script that fails is logged and recorded; the others still run.
        Raises AudioGenerationError only if every script tried failed.
        Returns the `audio_files` rows written this run (done and failed).
        """
        how_many = limit or self.cfg.get("max_scripts_per_run", DEFAULT_MAX_SCRIPTS_PER_RUN)
        log.info("settings: voice=%s music=%s max_scripts_per_run=%s max_attempts_per_script=%s wait_for_clips=%s",
                 self.voice.name, self.music.name if self.music else "none", how_many, self.max_attempts,
                 self.wait_for_clips)

        waiting = self._scripts_waiting_for_narration()
        if not waiting:
            log.info("no script is ready for narration; nothing to do")
            return []

        batch = waiting[:how_many]
        log.info("%d script(s) ready for narration; doing %d this run: #%s", len(waiting), len(batch),
                 ", #".join(str(script["id"]) for script, _ in batch))
        left_over = waiting[how_many:]
        if left_over:
            log.info("%d script(s) left for later runs (max_scripts_per_run=%d): #%s", len(left_over), how_many,
                     ", #".join(str(script["id"]) for script, _ in left_over))

        rows = []
        for number, (script, previous) in enumerate(batch, start=1):
            rows.append(self._narrate_one(script, previous, number, len(batch)))

        self._log_summary(rows)
        failed = [row for row in rows if row["status"] == "failed"]
        if failed and len(failed) == len(rows):
            raise AudioGenerationError(f"all {len(failed)} script(s) failed; see the errors above")
        return rows

    # --- Narrating one script ------------------------------------------------

    def _narrate_one(self, script, previous, number, total):
        """Narrate, upload and record one script.

        `previous` is the script's existing `audio_files` row (from an
        earlier failed attempt), or None the first time. Never raises: a
        failure is logged and recorded as "failed".
        """
        attempt = (previous["attempts"] if previous else 0) + 1
        _, label = script_location(self.db, script)
        folder = f"{day_folder()}/{label}"  # dated today: the day the narration is made
        log.info("[%d/%d] script #%s, attempt %d/%d -> %s/", number, total, script["id"], attempt,
                 self.max_attempts, folder)

        started = time.monotonic()
        try:
            text = self.db.download("scripts", script["file_path"])
            text = text.decode("utf-8") if isinstance(text, bytes) else text
            log.info("  %d characters | %s", len(text), _preview(text))
            log.debug("  full script: %s", text)

            narration = self.voice.narrate(text)
            log.info("  narration made in %.1fs: %s, %ss, voice %s, %s", time.monotonic() - started,
                     _size(len(narration.audio)), narration.seconds, narration.voice, _dollars(narration.cost_usd))
            narration_path = f"{folder}/narration.mp3"
            self.db.upload("audio", narration_path, narration.audio, "audio/mpeg")
            log.info("  saved to the audio bucket: %s", narration_path)

            music_path = None
            if self.music:
                prompt = self.cfg.get("music_prompt") or text[:200]
                track = self.music.music(prompt, narration.seconds or script.get("target_seconds") or 30)
                music_path = f"{folder}/music.mp3"
                self.db.upload("audio", music_path, track, "audio/mpeg")
                log.info("  music saved to the audio bucket: %s", music_path)

            result = {"status": "done", "narration_path": narration_path, "music_path": music_path, "error": None,
                      **_narration_details(narration)}
        except Exception as error:
            if attempt < self.max_attempts:
                next_step = "the next run will try it again"
            else:
                next_step = "that was the last attempt, so it is given up on"
            log.exception("  script #%s failed after %.1fs (attempt %d/%d): %s. Moving on; %s.", script["id"],
                          time.monotonic() - started, attempt, self.max_attempts, error, next_step)
            result = {"status": "failed", "error": str(error)[:MAX_ERROR_CHARS]}

        return self._record_attempt(script, previous, attempt, result)

    def _worth_trying(self, row):
        """True if a script with this `audio_files` row (None = never tried) should be narrated now."""
        if row is None:
            return True
        return row.get("status", "done") != "done" and (row.get("attempts") or 0) < self.max_attempts

    # --- Logging ------------------------------------------------------------

    def _log_summary(self, rows):
        done = [row for row in rows if row["status"] == "done"]
        failed = [row for row in rows if row["status"] == "failed"]
        costs = [row["cost_usd"] for row in done if row.get("cost_usd") is not None]
        log.info("done: narrated %d of %d script(s), %d failed; estimated cost this run: %s", len(done), len(rows),
                 len(failed), _dollars(sum(costs)) if costs else "unknown")
        for row in done:
            log.info("  ok     script #%s -> %s (%s)", row["script_id"], row["narration_path"],
                     _dollars(row.get("cost_usd")))
        for row in failed:
            log.error("  FAILED script #%s (attempt %s/%s): %s", row["script_id"], row["attempts"], self.max_attempts,
                      row.get("error"))

    def _warn_about_given_up(self, given_up):
        """One warning per script that ran out of attempts, so they aren't forgotten."""
        for row in given_up:
            log.warning("script #%s's narration was given up on after %s attempt(s), so it can't be assembled. "
                        "Last error: %s. To try again, raise max_attempts_per_script (now %s) or set attempts to 0 "
                        "for audio_files row #%s.", row["script_id"], row["attempts"], row.get("error"),
                        self.max_attempts, row["id"])

    # --- Reading and writing Supabase ---------------------------------------

    def _scripts_waiting_for_narration(self):
        """(script, its `audio_files` row or None) for every script to narrate now, oldest first."""
        rows = {row["script_id"]: row for row in self.db.select("audio_files")}
        self._warn_about_given_up([row for row in rows.values()
                                   if row.get("status", "done") != "done" and not self._worth_trying(row)])

        scripts = self.db.select("scripts", order="id.asc")
        needing = [script for script in scripts if self._worth_trying(rows.get(script["id"]))]
        ready = needing
        if self.wait_for_clips:
            with_clips = self._scripts_with_all_clips()
            ready = [script for script in needing if script["id"] in with_clips]
            log.info("found %d script(s): %d narrated or given up, %d still waiting for their clips, %d ready",
                     len(scripts), len(scripts) - len(needing), len(needing) - len(ready), len(ready))
        else:
            log.info("found %d script(s): %d narrated or given up, %d ready", len(scripts),
                     len(scripts) - len(needing), len(ready))
        return [(script, rows.get(script["id"])) for script in ready]

    def _scripts_with_all_clips(self):
        """Ids of scripts with a scene file whose every scene has a finished clip."""
        done = {}
        for row in self.db.select("video_scenes", {"status": "eq.done"}):
            done[row["scene_file_id"]] = done.get(row["scene_file_id"], 0) + 1
        return {scene_file["script_id"] for scene_file in self.db.select("scene_files")
                if scene_file.get("scene_count") and done.get(scene_file["id"], 0) >= scene_file["scene_count"]}

    def _record_attempt(self, script, previous, attempt, result):
        """Write the outcome to the script's one `audio_files` row (made on its first attempt)."""
        now = datetime.now(timezone.utc).isoformat()
        values = {**result, "provider": self.voice.name, "attempts": attempt, "updated_at": now}
        if previous:
            return self.db.update("audio_files", {"id": f"eq.{previous['id']}"}, values)[0]
        return self.db.insert("audio_files", {**values, "script_id": script["id"], "created_at": now})


# --- Small helpers -----------------------------------------------------------

def _narration_details(narration):
    """The `audio_files` columns that describe how a narration was made."""
    return {
        "model": narration.model,
        "voice": narration.voice,
        "characters": narration.characters,
        "audio_seconds": narration.seconds,
        "cost_usd": narration.cost_usd,
        "details": narration.details,
    }


def _preview(text, limit=PREVIEW_CHARS):
    """The start of `text` on one line, for log messages."""
    one_line = " ".join(str(text or "").split())
    return one_line if len(one_line) <= limit else one_line[:limit - 3] + "..."


def _dollars(amount):
    """e.g. 0.0075 -> "~$0.008"; "cost unknown" if None."""
    return "cost unknown" if amount is None else f"~${amount:.3f}"


def _size(n_bytes):
    """e.g. 2457600 -> "2.3 MB"."""
    return f"{n_bytes / 1024 / 1024:.1f} MB" if n_bytes >= 1024 * 1024 else f"{n_bytes / 1024:.0f} KB"
