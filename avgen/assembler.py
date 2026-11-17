"""Assemble stage: joins a script's clips and narration into the final video.

This stage picks the oldest scripts that are ready (every scene has a clip
and the narration is made), at most `max_scripts_per_run` per run (5 by
default), and for each one:
  1. downloads the clips, the narration (and music, if any),
  2. fits the clips to the narration (see "Timing" below) and joins them
     with ffmpeg into one mp4,
  3. saves it in the `final` bucket, dated the day it was made, e.g.
         2026-11-17/script_7_the-town-that-vanished.mp4
  4. records it in the `final_videos` table: "done" with its length, size,
     how the clips were fitted and the estimated cost of the whole video
     (clips + narration), or "failed" with the error.

Timing: the scene generator planned each scene's length from an estimated
speaking pace, and Seedance rounds every clip up to whole seconds, so the
clips rarely add up to the narration. Each scene therefore gets its planned
share of the real narration time: a clip that is longer than its share is
cut short, one that is shorter is slowed down to fill it. The video ends
with the narration, with no sentence cut off and no silent tail.

Each script has exactly one row in `final_videos`, which counts its
`attempts`. A script that fails doesn't stop the others; it is tried again
on the next run, up to `max_attempts_per_script` (3 by default) attempts in
total, then given up on and listed as a warning on every run. To try it
again, raise "max_attempts_per_script" in the config, or set the row's
`attempts` back to 0 in the table.

Output: every video comes out the same: 1080x1920 (9:16), 30 fps, H.264 +
AAC at 48 kHz, ready for YouTube Shorts. Clips of another shape are
cropped to fill the frame ("fit": "crop") or shown whole with black bars
("fit": "pad"). All of it is set in "output" (see DEFAULT_OUTPUT).

Subtitles: if the narration has word timings (the audio stage adds them
when captions are on), they are grouped into short captions and burned into
the picture. Grouping and look are set in "subtitles" (see captions.py).

Needs ffmpeg (with libass, for subtitles) and ffprobe installed. Settings,
in the "assemble" config section:

    "max_scripts_per_run": 5,
    "max_attempts_per_script": 3,
    "music_volume": 0.15,
    "output": {"width": 1080, "height": 1920, "fps": 30, "fit": "crop", ...},
    "subtitles": {"enabled": true, "min_words": 2, "max_words": 4, "style": {...}},
"""
import json
import logging
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from . import captions
from .storage_paths import day_folder, script_location

log = logging.getLogger(__name__)

# How many scripts to assemble in one run, when the config doesn't say.
# Assembling is free (ffmpeg runs on the GitHub runner) and takes about a
# minute per video. Change it with "max_scripts_per_run".
DEFAULT_MAX_SCRIPTS_PER_RUN = 5

# How many times to try one script before giving up on it, when the config
# doesn't say. Change it with "max_attempts_per_script".
DEFAULT_MAX_ATTEMPTS_PER_SCRIPT = 3

# Slowing a clip down by more than this looks unnatural; it still happens
# (so the narration is never cut off), but with a warning in the log.
NOTICEABLE_SLOWDOWN = 1.5

# The finished video, when the config's "output" doesn't say: vertical 1080p for
# YouTube Shorts. "fit" is "crop" (fill the frame, trim what sticks out) or
# "pad" (show the whole clip, black bars). crf is the quality (lower = better,
# bigger; 18-23 is typical) and preset the speed/size trade-off of the encoder.
DEFAULT_OUTPUT = {
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "fit": "crop",
    "crf": 20,
    "preset": "medium",
    "audio_bitrate": "192k",
    "audio_sample_rate": 48000,
}

# Name of the subtitle file next to the other inputs (ffmpeg runs in that folder).
SUBTITLE_FILE = "captions.ass"

# Longest error message stored in the `final_videos` table, and how much of
# ffmpeg's own output to keep when it fails.
MAX_ERROR_CHARS = 500
FFMPEG_OUTPUT_CHARS = 1500


class AssemblyError(RuntimeError):
    """Every script tried in this run failed."""


class Assembler:
    def __init__(self, db, cfg=None):
        self.db = db          # Supabase: tables and file storage
        self.cfg = cfg or {}  # the "assemble" section of the config

        self.max_attempts = self.cfg.get("max_attempts_per_script", DEFAULT_MAX_ATTEMPTS_PER_SCRIPT)
        self.music_volume = self.cfg.get("music_volume", 0.15)
        self.output = {**DEFAULT_OUTPUT, **(self.cfg.get("output") or {})}
        if self.output["fit"] not in ("crop", "pad"):
            raise ValueError(f"output fit must be \"crop\" or \"pad\", got {self.output['fit']!r}")
        self.subtitles = {"enabled": True, **(self.cfg.get("subtitles") or {})}

    # --- Main entry point -------------------------------------------------

    def run(self, limit=None):
        """Assemble the oldest scripts that are ready.

        `limit` is how many scripts to assemble; it defaults to
        "max_scripts_per_run" from the config, then to
        DEFAULT_MAX_SCRIPTS_PER_RUN.

        A script that fails is logged and recorded; the others still run.
        Raises AssemblyError only if every script tried failed.
        Returns the `final_videos` rows written this run (done and failed).
        """
        how_many = limit or self.cfg.get("max_scripts_per_run", DEFAULT_MAX_SCRIPTS_PER_RUN)
        log.info("settings: max_scripts_per_run=%s max_attempts_per_script=%s music_volume=%s", how_many,
                 self.max_attempts, self.music_volume)
        log.info("output: %sx%s, %s fps, fit=%s, crf %s; subtitles %s (%s-%s words per caption)",
                 self.output["width"], self.output["height"], self.output["fps"], self.output["fit"],
                 self.output["crf"], "on" if self.subtitles["enabled"] else "off",
                 self.subtitles.get("min_words", captions.DEFAULT_MIN_WORDS),
                 self.subtitles.get("max_words", captions.DEFAULT_MAX_WORDS))

        ready = self._scripts_ready()
        if not ready:
            log.info("no script is ready to assemble; nothing to do")
            return []

        batch = ready[:how_many]
        log.info("%d script(s) ready; assembling %d this run: #%s", len(ready), len(batch),
                 ", #".join(str(item["script"]["id"]) for item in batch))
        left_over = ready[how_many:]
        if left_over:
            log.info("%d script(s) left for later runs (max_scripts_per_run=%d): #%s", len(left_over), how_many,
                     ", #".join(str(item["script"]["id"]) for item in left_over))

        rows = [self._assemble_one(item, number, len(batch)) for number, item in enumerate(batch, start=1)]

        self._log_summary(rows)
        failed = [row for row in rows if row["status"] == "failed"]
        if failed and len(failed) == len(rows):
            raise AssemblyError(f"all {len(failed)} script(s) failed; see the errors above")
        return rows

    # --- Assembling one script --------------------------------------------

    def _assemble_one(self, item, number, total):
        """Build, upload and record one final video. Never raises: a failure is recorded as "failed"."""
        script, scene_file, clips, audio, previous = (item["script"], item["scene_file"], item["clips"],
                                                      item["audio"], item["previous"])
        attempt = (previous["attempts"] if previous else 0) + 1
        _, label = script_location(self.db, script)
        file_path = f"{day_folder()}/{label}.mp4"  # dated today: the day the video is made
        log.info("[%d/%d] script #%s, attempt %d/%d: %d clip(s) + narration -> %s", number, total, script["id"],
                 attempt, self.max_attempts, len(clips), file_path)

        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                clip_files = []
                for clip in clips:
                    path = tmp / f"scene_{clip['scene_index']:03d}.mp4"
                    path.write_bytes(self.db.download("videos", clip["file_path"]))
                    clip_files.append(path)
                narration = tmp / "narration.mp3"
                narration.write_bytes(self.db.download("audio", audio["narration_path"]))
                music = None
                if audio.get("music_path"):
                    music = tmp / "music.mp3"
                    music.write_bytes(self.db.download("audio", audio["music_path"]))
                log.info("  downloaded %d clip(s) and the narration%s", len(clips), " and music" if music else "")

                narration_seconds = media_seconds(narration)
                clip_seconds = [media_seconds(path) for path in clip_files]
                planned = self._planned_seconds(scene_file, len(clips))
                timing = plan_timing(planned, clip_seconds, narration_seconds)
                self._log_timing(timing, narration_seconds)

                subtitle_file = self._write_subtitles(tmp, audio)

                out = tmp / "final.mp4"
                run_ffmpeg(build_ffmpeg_cmd(clip_files, timing, narration, music, out, self.music_volume,
                                            self.output, subtitle_file), cwd=tmp)
                video = out.read_bytes()
                seconds = media_seconds(out)

            log.info("  video made in %.1fs: %s, %.1fs", time.monotonic() - started, _size(len(video)), seconds)
            self.db.upload("final", file_path, video, "video/mp4")
            log.info("  saved to the final bucket: %s", file_path)

            clips_cost = _total(clip.get("cost_usd") for clip in clips)
            narration_cost = _total([audio.get("cost_usd"), audio.get("transcription_cost_usd")])
            result = {
                "status": "done", "file_path": file_path, "error": None,
                "scene_file_id": scene_file["id"], "audio_file_id": audio["id"],
                "clip_count": len(clips), "seconds": seconds, "size_bytes": len(video),
                "clips_cost_usd": clips_cost, "narration_cost_usd": narration_cost,
                "cost_usd": _total([clips_cost, narration_cost]),
                "details": {"narration_seconds": narration_seconds, "timing": timing,
                            "music": bool(music), "music_volume": self.music_volume if music else None,
                            "output": self.output, "subtitles": self._caption_count},
            }
            log.info("  estimated cost of this video: %s (clips %s + narration %s)", _dollars(result["cost_usd"]),
                     _dollars(clips_cost), _dollars(narration_cost))
        except Exception as error:
            if attempt < self.max_attempts:
                next_step = "the next run will try it again"
            else:
                next_step = "that was the last attempt, so it is given up on"
            log.exception("  script #%s failed after %.1fs (attempt %d/%d): %s. Moving on; %s.", script["id"],
                          time.monotonic() - started, attempt, self.max_attempts, error, next_step)
            result = {"status": "failed", "error": str(error)[:MAX_ERROR_CHARS]}

        return self._record_attempt(script, previous, attempt, result)

    def _write_subtitles(self, folder, audio):
        """Write the captions file for this narration into `folder`; returns its name, or None for none."""
        self._caption_count = 0
        if not self.subtitles["enabled"]:
            return None
        words = audio.get("words")
        if not words:
            log.warning("script #%s's narration has no word timings, so its video has no subtitles (turn on "
                        "\"captions\" in the audio config; the audio stage adds them on its next run)",
                        audio["script_id"])
            return None
        grouped = captions.group_words(
            words,
            min_words=self.subtitles.get("min_words", captions.DEFAULT_MIN_WORDS),
            max_words=self.subtitles.get("max_words", captions.DEFAULT_MAX_WORDS),
            max_pause_seconds=self.subtitles.get("max_pause_seconds", captions.DEFAULT_MAX_PAUSE_SECONDS),
        )
        ass = captions.to_ass(grouped, self.subtitles.get("style"), self.output["width"], self.output["height"])
        (folder / SUBTITLE_FILE).write_text(ass, encoding="utf-8")
        self._caption_count = len(grouped)
        log.info("  subtitles: %d caption(s) from %d word(s), e.g. %s", len(grouped), len(words),
                 " | ".join(caption["text"] for caption in grouped[:3]))
        return SUBTITLE_FILE

    def _planned_seconds(self, scene_file, clip_count):
        """Each scene's planned length from the scene file, or None if it can't be used."""
        try:
            scenes = json.loads(self.db.download("scenes", scene_file["file_path"]))["scenes"]
            planned = [float(scene["duration"]) for scene in sorted(scenes, key=lambda scene: scene["index"])]
        except Exception as error:
            log.warning("  couldn't read the planned scene lengths (%s); sharing time by clip length instead", error)
            return None
        if len(planned) != clip_count:
            log.warning("  the scene file has %d scene(s) but there are %d clip(s); sharing time by clip length "
                        "instead", len(planned), clip_count)
            return None
        return planned

    def _log_timing(self, timing, narration_seconds):
        log.info("  fitting %d clip(s) to %.1fs of narration:", len(timing), narration_seconds)
        for number, scene in enumerate(timing):
            if scene["slowdown"] > 1:
                how = f"slowed down x{scene['slowdown']:.2f}"
            elif scene["clip_seconds"] > scene["seconds"]:
                how = f"cut from {scene['clip_seconds']:.1f}s"
            else:
                how = "as is"
            log.info("    scene %d: %.2fs (%s)", number, scene["seconds"], how)
            if scene["slowdown"] > NOTICEABLE_SLOWDOWN:
                log.warning("scene %d is slowed down x%.2f to fill the narration, which may look unnatural",
                            number, scene["slowdown"])

    def _worth_trying(self, row):
        """True if a script with this `final_videos` row (None = never tried) should be assembled now."""
        if row is None:
            return True
        return row.get("status", "done") != "done" and (row.get("attempts") or 0) < self.max_attempts

    # --- Logging ------------------------------------------------------------

    def _log_summary(self, rows):
        done = [row for row in rows if row["status"] == "done"]
        failed = [row for row in rows if row["status"] == "failed"]
        log.info("done: assembled %d of %d script(s), %d failed", len(done), len(rows), len(failed))
        for row in done:
            log.info("  ok     script #%s -> %s (%.1fs, %s)", row["script_id"], row["file_path"], row["seconds"],
                     _dollars(row.get("cost_usd")))
        for row in failed:
            log.error("  FAILED script #%s (attempt %s/%s): %s", row["script_id"], row["attempts"], self.max_attempts,
                      row.get("error"))

    def _warn_about_given_up(self, given_up):
        for row in given_up:
            log.warning("script #%s's final video was given up on after %s attempt(s). Last error: %s. To try "
                        "again, raise max_attempts_per_script (now %s) or set attempts to 0 for final_videos row "
                        "#%s.", row["script_id"], row["attempts"], row.get("error"), self.max_attempts, row["id"])

    # --- Reading and writing Supabase ---------------------------------------

    def _scripts_ready(self):
        """Everything needed to assemble each ready script, oldest first.

        Ready means: no finished final video (and attempts left), the
        narration is done, and some scene file of the script has a finished
        clip for every scene (the newest such file is used).
        """
        finals = {row["script_id"]: row for row in self.db.select("final_videos")}
        self._warn_about_given_up([row for row in finals.values()
                                   if row.get("status", "done") != "done" and not self._worth_trying(row)])
        audio = {row["script_id"]: row for row in self.db.select("audio_files", {"status": "eq.done"})}

        clips_by_file = {}
        for clip in self.db.select("video_scenes", {"status": "eq.done"}):
            clips_by_file.setdefault(clip["scene_file_id"], {})[clip["scene_index"]] = clip
        complete_files = {}  # script id -> its newest scene file with every clip done
        for scene_file in self.db.select("scene_files", order="id.asc"):
            count = scene_file.get("scene_count")
            if count and set(clips_by_file.get(scene_file["id"], {})) >= set(range(count)):
                complete_files[scene_file["script_id"]] = scene_file

        scripts = self.db.select("scripts", order="id.asc")
        to_do = [script for script in scripts if self._worth_trying(finals.get(script["id"]))]
        ready = []
        for script in to_do:
            scene_file = complete_files.get(script["id"])
            if scene_file and script["id"] in audio:
                clips = clips_by_file[scene_file["id"]]
                ready.append({"script": script, "scene_file": scene_file, "audio": audio[script["id"]],
                              "clips": [clips[index] for index in range(scene_file["scene_count"])],
                              "previous": finals.get(script["id"])})
        waiting_for_clips = sum(1 for script in to_do if script["id"] not in complete_files)
        log.info("found %d script(s): %d assembled or given up, %d waiting for clips, %d waiting for narration, "
                 "%d ready", len(scripts), len(scripts) - len(to_do), waiting_for_clips,
                 len(to_do) - waiting_for_clips - len(ready), len(ready))
        return ready

    def _record_attempt(self, script, previous, attempt, result):
        """Write the outcome to the script's one `final_videos` row (made on its first attempt)."""
        now = datetime.now(timezone.utc).isoformat()
        values = {**result, "attempts": attempt, "updated_at": now}
        if previous:
            return self.db.update("final_videos", {"id": f"eq.{previous['id']}"}, values)[0]
        return self.db.insert("final_videos", {**values, "script_id": script["id"], "created_at": now})


# --- Timing and ffmpeg (no Supabase, easy to test) ----------------------------

def plan_timing(planned_seconds, clip_seconds, narration_seconds):
    """How long each clip plays so that together they last as long as the narration.

    Each scene gets its share of the narration in proportion to its planned
    length (or, without plans, its clip's length). A clip longer than its
    share is cut short; a shorter one is slowed down by `slowdown` to fill it.
    Returns one dict per clip: seconds (time on screen), clip_seconds
    (length of the clip as made), use_seconds (how much of the clip is
    shown) and slowdown (1 = normal speed).
    """
    weights = planned_seconds or clip_seconds
    total = sum(weights)
    timing = []
    for weight, clip in zip(weights, clip_seconds):
        share = narration_seconds * weight / total
        use = min(clip, share)
        timing.append({"seconds": round(share, 3), "clip_seconds": round(clip, 3), "use_seconds": round(use, 3),
                       "slowdown": round(share / use, 3)})
    return timing


def build_ffmpeg_cmd(clips, timing, narration, music, out, music_volume=0.15, output=None, subtitles=None):
    """The ffmpeg command that joins the fitted clips and lays the audio under them.

    Every clip is cut to its `use_seconds`, stretched by its `slowdown` and
    brought to the `output` size and frame rate (cropped or padded to the
    frame), then the clips are joined end to end; the result lasts as long
    as the narration. `subtitles` (an .ass file name, relative to where
    ffmpeg runs) is burned into the picture. Music, if any, loops quietly
    under the narration.
    """
    o = {**DEFAULT_OUTPUT, **(output or {})}
    width, height = o["width"], o["height"]
    if o["fit"] == "pad":
        frame = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                 f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")
    else:
        frame = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for clip in clips:
        cmd += ["-i", str(clip)]
    narration_input = len(clips)
    cmd += ["-i", str(narration)]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]

    parts = [f"[{i}:v]trim=duration={scene['use_seconds']},setpts={scene['slowdown']}*(PTS-STARTPTS),"
             f"{frame},fps={o['fps']},setsar=1[v{i}]" for i, scene in enumerate(timing)]
    joined = "[joined]" if subtitles else "[v]"
    parts.append("".join(f"[v{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0{joined}")
    if subtitles:
        parts.append(f"[joined]subtitles={subtitles}[v]")
    if music:
        parts.append(f"[{narration_input + 1}:a]volume={music_volume}[m];"
                     f"[{narration_input}:a][m]amix=inputs=2:duration=first:normalize=0[a]")
        audio_out = "[a]"
    else:
        audio_out = f"{narration_input}:a"

    return cmd + ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", audio_out,
                  "-c:v", "libx264", "-preset", str(o["preset"]), "-crf", str(o["crf"]), "-pix_fmt", "yuv420p",
                  "-r", str(o["fps"]), "-c:a", "aac", "-b:a", str(o["audio_bitrate"]),
                  "-ar", str(o["audio_sample_rate"]), "-movflags", "+faststart", "-shortest", str(out)]


def run_ffmpeg(cmd, cwd=None):
    """Run an ffmpeg command (in folder `cwd`); on failure the error carries ffmpeg's own message."""
    log.debug("running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=cwd)
    except FileNotFoundError:
        raise RuntimeError(f"{cmd[0]} is not installed (or not on PATH)") from None
    except subprocess.CalledProcessError as error:
        output = (error.stderr or error.stdout or "").strip()[-FFMPEG_OUTPUT_CHARS:]
        raise RuntimeError(f"{cmd[0]} failed (exit {error.returncode}): {output or 'no output'}") from None


def media_seconds(path):
    """Length of an audio or video file in seconds, read with ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffprobe is not installed (or not on PATH)") from None
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffprobe couldn't read {Path(path).name}: {error.stderr.strip()}") from None
    return round(float(result.stdout.strip()), 3)


# --- Small helpers -----------------------------------------------------------

def _total(amounts):
    """Sum of the known amounts; None if none are known."""
    known = [amount for amount in amounts if amount is not None]
    return round(sum(float(amount) for amount in known), 6) if known else None


def _dollars(amount):
    return "cost unknown" if amount is None else f"~${float(amount):.3f}"


def _size(n_bytes):
    return f"{n_bytes / 1024 / 1024:.1f} MB" if n_bytes >= 1024 * 1024 else f"{n_bytes / 1024:.0f} KB"
