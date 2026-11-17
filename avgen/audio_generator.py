"""Audio stage: reads each script aloud (and adds background music, if set up).

This stage picks the oldest scripts that still need a narration (at most
`max_scripts_per_run` per run, 10 by default) and, for each one:
  1. sends the script's text to the voice service (OpenAI text-to-speech),
  2. saves the mp3 in the `audio` bucket, in a folder named after the
     script and dated the day the narration was made, e.g.
         2026-11-17/script_7_the-town-that-vanished/narration.mp3
     (plus music.mp3 next to it, if music is set up),
  3. if captions are on, sends the mp3 to a transcription model (whisper-1)
     to learn when each word is spoken, and saves captions.srt next to the
     narration (handy for YouTube's own closed captions). The assemble stage
     turns the word timings into burned-in subtitles,
  4. records it in the `audio_files` table: "done" with how it was made
     (model, voice, characters, length, word timings, captions file,
     estimated cost in USD), or "failed" with the error.

Captions are extra work on top of a finished narration: if they fail, the
narration still counts as done (its video just has no subtitles yet) and
the captions are tried again on later runs, up to `max_attempts_per_script`
times, counted separately in `caption_attempts`. Narrations made before
captions existed, or without a captions.srt, get them on the next run the
same way, without being narrated again.

By default (`wait_for_clips`: true) a script is only narrated once all its
clips are made: until then it can't be assembled anyway, and its narration
lands in the same date folder as (or a later one than) its clips.

Each script has exactly one row in `audio_files`, which counts its
`attempts`. A script that fails doesn't stop the others; it is tried again
on the next run, up to `max_attempts_per_script` (3 by default) attempts in
total, then given up on and listed as a warning on every run. To try it
again, raise "max_attempts_per_script" in the config, or set the row's
`attempts` (or `caption_attempts`) back to 0 in the table.

The settings live in the "audio" section of the config:

    "max_scripts_per_run": 10,
    "max_attempts_per_script": 3,
    "wait_for_clips": true,
    "captions": {"enabled": true, "model": "whisper-1", "srt_min_words": 3, "srt_max_words": 8, ...}
"""
import logging
import time
from datetime import datetime, timezone

from . import captions
from .storage_paths import day_folder, script_location

log = logging.getLogger(__name__)

# How many scripts to narrate in one run, when the config doesn't say.
# A narration costs about a cent, so this can be generous. Change it with
# "max_scripts_per_run".
DEFAULT_MAX_SCRIPTS_PER_RUN = 10

# How many times to try one script before giving up on it, when the config
# doesn't say. Change it with "max_attempts_per_script".
DEFAULT_MAX_ATTEMPTS_PER_SCRIPT = 3

# Words per line in captions.srt, when the config doesn't say. Longer than
# the burned-in subtitles: these are read as closed captions, a phrase at a time.
DEFAULT_SRT_MIN_WORDS = 3
DEFAULT_SRT_MAX_WORDS = 8

# Name of the captions file saved next to each narration.
CAPTIONS_FILE = "captions.srt"

# How much of a script to show on one INFO log line.
PREVIEW_CHARS = 80

# Longest error message stored in the `audio_files` table.
MAX_ERROR_CHARS = 500


class AudioGenerationError(RuntimeError):
    """Every script tried in this run failed."""


class AudioGenerator:
    def __init__(self, db, voice, music=None, cfg=None, transcriber=None):
        self.db = db                    # Supabase: tables and file storage
        self.voice = voice              # the service that reads the script aloud (OpenAI TTS)
        self.music = music              # background music, or None for none
        self.cfg = cfg or {}            # the "audio" section of the config
        self.transcriber = transcriber  # word timings for subtitles (whisper-1), or None for no captions

        self.max_attempts = self.cfg.get("max_attempts_per_script", DEFAULT_MAX_ATTEMPTS_PER_SCRIPT)
        self.wait_for_clips = self.cfg.get("wait_for_clips", True)
        caption_cfg = self.cfg.get("captions") or {}
        self.srt_min_words = caption_cfg.get("srt_min_words", DEFAULT_SRT_MIN_WORDS)
        self.srt_max_words = caption_cfg.get("srt_max_words", DEFAULT_SRT_MAX_WORDS)

    # --- Main entry point -------------------------------------------------

    def run(self, limit=None):
        """Narrate (or add captions to) the oldest scripts that still need it.

        `limit` is how many scripts to work on; it defaults to
        "max_scripts_per_run" from the config, then to
        DEFAULT_MAX_SCRIPTS_PER_RUN.

        A script that fails is logged and recorded; the others still run.
        Raises AudioGenerationError only if every script tried failed.
        Returns the `audio_files` rows written this run.
        """
        how_many = limit or self.cfg.get("max_scripts_per_run", DEFAULT_MAX_SCRIPTS_PER_RUN)
        log.info("settings: voice=%s music=%s captions=%s max_scripts_per_run=%s max_attempts_per_script=%s "
                 "wait_for_clips=%s", self.voice.name, self.music.name if self.music else "none",
                 self.transcriber.model if self.transcriber else "off", how_many, self.max_attempts,
                 self.wait_for_clips)

        waiting = self._scripts_waiting()
        if not waiting:
            log.info("no script needs a narration or captions; nothing to do")
            return []

        batch = waiting[:how_many]
        log.info("%d script(s) to work on; doing %d this run: #%s", len(waiting), len(batch),
                 ", #".join(str(script["id"]) for script, _ in batch))
        left_over = waiting[how_many:]
        if left_over:
            log.info("%d script(s) left for later runs (max_scripts_per_run=%d): #%s", len(left_over), how_many,
                     ", #".join(str(script["id"]) for script, _ in left_over))

        outcomes = []  # (row, worked, cost of this run's work)
        for number, (script, previous) in enumerate(batch, start=1):
            outcomes.append(self._work_on(script, previous, number, len(batch)))

        self._log_summary(outcomes)
        failed = [row for row, worked, _ in outcomes if not worked]
        if failed and len(failed) == len(outcomes):
            raise AudioGenerationError(f"all {len(failed)} script(s) failed; see the errors above")
        return [row for row, _, _ in outcomes]

    # --- One script ---------------------------------------------------------

    def _work_on(self, script, previous, number, total):
        """Narrate the script if it has no finished narration, then add captions if they're missing.

        `previous` is the script's existing `audio_files` row (an earlier
        failed attempt, or a finished narration that still needs captions),
        or None the first time. Never raises: failures are logged and
        recorded. Returns (row, worked, cost of this run's work).
        """
        narrated = previous is not None and previous.get("status", "done") == "done"
        values, cost = {}, []
        try:
            text = self.db.download("scripts", script["file_path"])
            text = text.decode("utf-8") if isinstance(text, bytes) else text
        except Exception as error:
            log.exception("[%d/%d] script #%s: couldn't download the script: %s", number, total, script["id"], error)
            text = None
            if not narrated:
                return self._narration_failed(script, previous, error, 0.0)

        if narrated:
            log.info("[%d/%d] script #%s: narration already made, adding captions (caption attempt %d/%d)",
                     number, total, script["id"], self._caption_attempts(previous) + 1, self.max_attempts)
        else:
            started = time.monotonic()
            try:
                values = self._make_narration(script, previous, text, number, total)
            except Exception as error:
                return self._narration_failed(script, previous, error, time.monotonic() - started)
            cost.append(values.get("cost_usd"))

        worked = True
        if self.transcriber:
            caption_values = self._add_captions({**(previous or {}), "script_id": script["id"], **values}, text)
            values.update(caption_values)
            cost.append(caption_values.get("transcription_cost_usd") if "words" in caption_values else None)
            worked = caption_values.get("caption_error") is None

        return self._save(script, previous, values), worked, _total(cost)

    def _make_narration(self, script, previous, text, number, total):
        """Narrate the script and save the mp3 (and music); returns the row's new values."""
        attempt = (previous["attempts"] if previous else 0) + 1
        _, label = script_location(self.db, script)
        folder = f"{day_folder()}/{label}"  # dated today: the day the narration is made
        log.info("[%d/%d] script #%s, attempt %d/%d -> %s/", number, total, script["id"], attempt,
                 self.max_attempts, folder)
        log.info("  %d characters | %s", len(text), _preview(text))
        log.debug("  full script: %s", text)

        started = time.monotonic()
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

        return {"status": "done", "narration_path": narration_path, "music_path": music_path, "error": None,
                "attempts": attempt, "provider": self.voice.name, "_audio": narration.audio,
                **_narration_details(narration)}

    def _narration_failed(self, script, previous, error, seconds):
        attempt = (previous["attempts"] if previous else 0) + 1
        if attempt < self.max_attempts:
            next_step = "the next run will try it again"
        else:
            next_step = "that was the last attempt, so it is given up on"
        log.exception("  script #%s failed after %.1fs (attempt %d/%d): %s. Moving on; %s.", script["id"], seconds,
                      attempt, self.max_attempts, error, next_step)
        values = {"status": "failed", "error": str(error)[:MAX_ERROR_CHARS], "attempts": attempt,
                  "provider": self.voice.name}
        return self._save(script, previous, values), False, None

    def _add_captions(self, row, text):
        """Word timings (if missing) and an up-to-date captions.srt for a finished narration.

        The words are lined up with the script (captions.align_to_script),
        so the captions show the script's own words: whole numbers, plain
        English, nothing the transcription invented. Those aligned words are
        what's saved in `words`, for the burned-in subtitles too.

        `row` is the narration's row as it stands now. Returns the row's new
        values; on failure, just the counted attempt and the error, so the
        narration itself stays done.
        """
        attempt = self._caption_attempts(row) + 1
        started = time.monotonic()
        try:
            values = {}
            words = row.get("words")
            if not words:
                audio = row.get("_audio") or self.db.download("audio", row["narration_path"])
                transcript = self.transcriber.transcribe(audio, seconds=row.get("audio_seconds"), script_text=text)
                words = transcript.words
                log.info("  word timings from %s in %.1fs: %d word(s), %s", transcript.model,
                         time.monotonic() - started, len(words), _dollars(transcript.cost_usd))
                log.debug("  heard: %s", " ".join(word["word"] for word in words))
                values.update(transcription_model=transcript.model, transcription_cost_usd=transcript.cost_usd)

            words = captions.align_to_script(words, text)
            if not words:
                raise RuntimeError("no usable words to make captions from")
            log.debug("  caption words: %s", " ".join(word["word"] for word in words))
            lines = captions.group_words(words, self.srt_min_words, self.srt_max_words)
            captions_path = f"{row['narration_path'].rsplit('/', 1)[0]}/{CAPTIONS_FILE}"
            self.db.upload("audio", captions_path, captions.to_srt(lines), "application/x-subrip")
            log.info("  captions saved to the audio bucket: %s (%d line(s))", captions_path, len(lines))
            return {**values, "words": words, "captions_path": captions_path, "caption_attempts": attempt,
                    "caption_error": None, "captions_version": captions.CAPTIONS_VERSION}
        except Exception as error:
            if attempt < self.max_attempts:
                next_step = "the next run will try them again"
            else:
                next_step = "that was the last attempt, so they are given up on"
            log.exception("  captions for script #%s failed after %.1fs (caption attempt %d/%d): %s. The narration "
                          "is kept; %s.", row.get("script_id"), time.monotonic() - started, attempt,
                          self.max_attempts, error, next_step)
            return {"caption_attempts": attempt, "caption_error": str(error)[:MAX_ERROR_CHARS],
                    "captions_version": captions.CAPTIONS_VERSION}

    def _needs_captions(self, row):
        """True if captions are on and this narration's word timings or captions.srt are missing or out of date."""
        if self.transcriber is None:
            return False
        up_to_date = row.get("captions_version") == captions.CAPTIONS_VERSION and row.get("caption_error") is None
        return not (row.get("words") and row.get("captions_path") and up_to_date)

    @staticmethod
    def _caption_attempts(row):
        """Caption attempts so far, for the current way of making captions (a new CAPTIONS_VERSION starts at 0)."""
        if row.get("captions_version") != captions.CAPTIONS_VERSION:
            return 0
        return row.get("caption_attempts") or 0

    def _worth_trying(self, row):
        """True if a script with this `audio_files` row (None = never tried) should be worked on now.

        That is: never tried, failed with attempts left, or narrated but
        missing word timings or an up-to-date captions.srt (with caption
        attempts left).
        """
        if row is None:
            return True
        if row.get("status", "done") != "done":
            return (row.get("attempts") or 0) < self.max_attempts
        return self._needs_captions(row) and self._caption_attempts(row) < self.max_attempts

    # --- Logging ------------------------------------------------------------

    def _log_summary(self, outcomes):
        failed = [row for row, worked, _ in outcomes if not worked]
        costs = [cost for _, _, cost in outcomes if cost is not None]
        log.info("done: %d of %d script(s) worked, %d failed; estimated cost this run: %s",
                 len(outcomes) - len(failed), len(outcomes), len(failed), _dollars(sum(costs)) if costs else "unknown")
        for row, worked, cost in outcomes:
            if worked:
                log.info("  ok     script #%s -> %s (captions: %s, %s)", row["script_id"], row["narration_path"],
                         row.get("captions_path") or "off", _dollars(cost))
        for row in failed:
            if row.get("status") == "failed":
                log.error("  FAILED script #%s narration (attempt %s/%s): %s", row["script_id"], row["attempts"],
                          self.max_attempts, row.get("error"))
            else:
                log.error("  FAILED script #%s captions (caption attempt %s/%s): %s", row["script_id"],
                          row.get("caption_attempts"), self.max_attempts, row.get("caption_error"))

    def _warn_about_given_up(self, rows):
        """One warning per script that ran out of attempts, so they aren't forgotten."""
        for row in rows:
            if row.get("status", "done") != "done":
                log.warning("script #%s's narration was given up on after %s attempt(s), so it can't be assembled. "
                            "Last error: %s. To try again, raise max_attempts_per_script (now %s) or set attempts "
                            "to 0 for audio_files row #%s.", row["script_id"], row.get("attempts"), row.get("error"),
                            self.max_attempts, row["id"])
            else:
                log.warning("script #%s's captions were given up on after %s attempt(s), so its video has no "
                            "subtitles. Last error: %s. To try again, raise max_attempts_per_script (now %s) or set "
                            "caption_attempts to 0 for audio_files row #%s.", row["script_id"],
                            row.get("caption_attempts"), row.get("caption_error"), self.max_attempts, row["id"])

    # --- Reading and writing Supabase ---------------------------------------

    def _scripts_waiting(self):
        """(script, its `audio_files` row or None) for every script to work on now, oldest first."""
        rows = {row["script_id"]: row for row in self.db.select("audio_files")}
        given_up = [row for row in rows.values() if not self._worth_trying(row)
                    and (row.get("status", "done") != "done" or self._needs_captions(row))]
        self._warn_about_given_up(given_up)

        scripts = self.db.select("scripts", order="id.asc")
        needing = [script for script in scripts if self._worth_trying(rows.get(script["id"]))]
        needing_captions = sum(1 for script in needing if rows.get(script["id"])
                               and rows[script["id"]].get("status", "done") == "done")
        ready = needing
        if self.wait_for_clips:
            with_clips = self._scripts_with_all_clips()
            ready = [script for script in needing if script["id"] in with_clips
                     or (rows.get(script["id"]) or {}).get("status") == "done"]
        log.info("found %d script(s): %d need captions only, %d need narrating (%d of them still waiting for their "
                 "clips), %d finished or given up", len(scripts), needing_captions, len(needing) - needing_captions,
                 len(needing) - len(ready), len(scripts) - len(needing))
        return [(script, rows.get(script["id"])) for script in ready]

    def _scripts_with_all_clips(self):
        """Ids of scripts with a scene file whose every scene has a finished clip."""
        done = {}
        for row in self.db.select("video_scenes", {"status": "eq.done"}):
            done[row["scene_file_id"]] = done.get(row["scene_file_id"], 0) + 1
        return {scene_file["script_id"] for scene_file in self.db.select("scene_files")
                if scene_file.get("scene_count") and done.get(scene_file["id"], 0) >= scene_file["scene_count"]}

    def _save(self, script, previous, values):
        """Write the new values to the script's one `audio_files` row (made on its first attempt)."""
        now = datetime.now(timezone.utc).isoformat()
        values = {key: value for key, value in values.items() if not key.startswith("_")}  # drop the mp3 itself
        values["updated_at"] = now
        if previous:
            return self.db.update("audio_files", {"id": f"eq.{previous['id']}"}, values)[0]
        return self.db.insert("audio_files", {**values, "script_id": script["id"], "created_at": now})


# --- Small helpers -----------------------------------------------------------

def _total(amounts):
    """Sum of the known amounts; None if none are known."""
    known = [amount for amount in amounts if amount is not None]
    return round(sum(known), 6) if known else None


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
