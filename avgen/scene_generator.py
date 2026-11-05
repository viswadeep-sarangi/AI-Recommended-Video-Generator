"""Scene generator: splits each narration script into short video scenes.

For each script that hasn't been split yet (oldest first, at most
`max_per_run` per run), it:
  1. downloads the script text from Supabase storage,
  2. asks the LLM to break it into scenes of a few seconds each, where every
     scene has a piece of the narration and a description of what to show,
  3. checks the LLM's reply has exactly the expected shape (asking again if
     it doesn't), then keeps every scene's length within limits,
  4. saves the scenes as a JSON file in storage,
  5. records the file in the `scene_files` table so it isn't split again.

If one script fails, the error is logged and the run moves on to the next
one. The failed script has no scene file, so the next run tries it again.

The video stage then turns each scene's description into a video clip.

The LLM must reply with a JSON list like this (anything else is rejected):

    [
      {"narration": "...", "visual_prompt": "...", "duration": 3},
      {"narration": "...", "visual_prompt": "...", "duration": 4.5}
    ]

and the saved scene file looks like this:

    {
      "script_id": 7,
      "scenes": [
        {"index": 0, "narration": "...", "visual_prompt": "...", "duration": 3.0},
        {"index": 1, "narration": "...", "visual_prompt": "...", "duration": 4.5}
      ]
    }
"""
import json
import logging
import time

from .llm import extract_json, wrap_prompt
from .storage_paths import script_location

log = logging.getLogger(__name__)

# Defaults used when the "scenes" section of the config leaves a setting out.
DEFAULT_MIN_SCENE_SECONDS = 2
DEFAULT_MAX_SCENE_SECONDS = 5
DEFAULT_MAX_PER_RUN = 20
DEFAULT_LLM_ATTEMPTS = 2        # ask again once if the LLM's reply is unusable
DEFAULT_WORDS_PER_SECOND = 2.5  # speaking speed, used to sanity-check scene lengths

# Warn when the scenes' total length or total narration is further than this
# (as a fraction) from what the script itself suggests.
MISMATCH_TOLERANCE = 0.3

# How much of a long piece of text to show on one INFO log line.
PREVIEW_CHARS = 80

# The fields every scene in the LLM's reply must have.
TEXT_FIELDS = ("narration", "visual_prompt")


class InvalidSceneReply(ValueError):
    """The LLM's reply isn't the JSON list of scenes we asked for."""


class SceneGenerationError(RuntimeError):
    """Every script picked for this run failed."""


class SceneGenerator:
    def __init__(self, llm, db, cfg):
        self.llm = llm  # the language model that splits scripts into scenes
        self.db = db    # Supabase: tables and file storage
        self.cfg = cfg  # the "scenes" section of the config

        # Every scene must last between these many seconds.
        self.min_s = cfg.get("min_scene_seconds", DEFAULT_MIN_SCENE_SECONDS)
        self.max_s = cfg.get("max_scene_seconds", DEFAULT_MAX_SCENE_SECONDS)

    # --- Main entry point -------------------------------------------------

    def run_pending(self, limit=None):
        """Split the oldest scripts that don't have a scene file yet.

        A script that fails is logged and skipped; the others still run.
        Raises SceneGenerationError only if every script picked failed.
        """
        how_many = limit or self.cfg.get("max_per_run", DEFAULT_MAX_PER_RUN)
        self._log_settings(how_many)

        waiting = self._scripts_waiting_for_scenes()
        if not waiting:
            log.info("every script already has a scene file; nothing to split")
            return []

        batch = waiting[:how_many]
        log.info("%d script(s) waiting for scenes; splitting %d this run:", len(waiting), len(batch))
        for script in batch:
            log.info("  script #%s (topic #%s, written %s): %s", script["id"], script.get("topic_id"),
                     script.get("created_at", "?"), script["file_path"])
        left_over = waiting[how_many:]
        if left_over:
            log.info("%d script(s) left for later runs (max_per_run=%d): #%s", len(left_over), how_many,
                     ", #".join(str(s["id"]) for s in left_over))

        scene_files, failures = [], []
        for number, script in enumerate(batch, start=1):
            log.info("[%d/%d] splitting script #%s", number, len(batch), script["id"])
            started = time.monotonic()
            try:
                scene_files.append(self.generate(script))
            except Exception as error:
                failures.append((script, error))
                log.exception("[%d/%d] failed on script #%s after %.1fs: %s. Skipping it and moving on; "
                              "it has no scene file, so the next run will try it again.",
                              number, len(batch), script["id"], time.monotonic() - started, error)
            else:
                log.info("[%d/%d] script #%s done in %.1fs", number, len(batch), script["id"],
                         time.monotonic() - started)

        self._log_summary(batch, scene_files, failures)
        if failures and not scene_files:
            raise SceneGenerationError(f"all {len(failures)} script(s) failed; see the errors above")
        return scene_files

    def generate(self, script):
        """Split, check, save and record the scenes for one script."""
        script_text = self._download_script_text(script)
        if not script_text.strip():
            raise ValueError(f"script file {script['file_path']} is empty")

        raw_scenes = self._ask_llm_for_scenes(script, script_text)
        scenes = normalize_scenes(raw_scenes, self.min_s, self.max_s)
        self._log_scenes(script, raw_scenes, scenes)
        self._check_scenes_match_script(script, script_text, scenes)

        file_path = self._save_scene_file(script, scenes)
        log.info("  saved %d scene(s) to the scenes bucket: %s", len(scenes), file_path)
        row = self._record_scene_file(script, file_path, scenes)
        log.info("  recorded as scene file #%s", row.get("id"))
        return row

    # --- Talking to the LLM -----------------------------------------------

    def build_prompt(self, script_text):
        """The request sent to the LLM, wrapped in the configured extra notes."""
        request = (
            "Split this narration into consecutive video scenes, "
            f"each {self.min_s}-{self.max_s} seconds long. "
            "Reply ONLY with JSON: "
            '[{"narration": str, "visual_prompt": str, "duration": seconds}].'
            "\n\n"
            f"{script_text}"
        )

        notes_before = self.cfg.get("prepend_instructions", "")  # goes before the request
        notes_after = self.cfg.get("append_instructions", "")    # goes after the request
        return wrap_prompt(request, notes_before, notes_after)

    def _ask_llm_for_scenes(self, script, script_text):
        """Ask the LLM to split the script, asking again if the reply is unusable.

        Returns the checked (but not yet tidied) list of scenes.
        """
        prompt = self.build_prompt(script_text)
        log.debug("LLM prompt for script #%s:\n%s", script["id"], prompt)

        attempts = max(1, int(self.cfg.get("llm_attempts", DEFAULT_LLM_ATTEMPTS)))
        for attempt in range(1, attempts + 1):
            log.info("  asking the LLM to split it into scenes (attempt %d/%d)", attempt, attempts)
            started = time.monotonic()
            reply = self.llm.complete(prompt)
            log.info("  LLM replied in %.1fs (%d characters)", time.monotonic() - started, len(reply or ""))
            log.debug("LLM reply for script #%s (attempt %d/%d):\n%s", script["id"], attempt, attempts, reply)
            try:
                scenes = parse_scene_reply(reply)
            except InvalidSceneReply as problem:
                if attempt == attempts:
                    raise InvalidSceneReply(f"the LLM's reply was still unusable after {attempts} "
                                            f"attempt(s): {problem}") from None
                log.warning("script #%s: the LLM's reply on attempt %d/%d was unusable, asking again. "
                            "Problem: %s", script["id"], attempt, attempts, problem)
                continue
            log.info("  the LLM's reply is valid: %d scene(s)", len(scenes))
            return scenes

    # --- Logging and sanity checks ---------------------------------------

    def _log_settings(self, how_many):
        log.info("settings: max_per_run=%s scene length=%s-%ss llm_attempts=%s words_per_second=%s",
                 how_many, self.min_s, self.max_s,
                 self.cfg.get("llm_attempts", DEFAULT_LLM_ATTEMPTS),
                 self.cfg.get("words_per_second", DEFAULT_WORDS_PER_SECOND))
        log.debug("prepend_instructions: %r", self.cfg.get("prepend_instructions", ""))
        log.debug("append_instructions: %r", self.cfg.get("append_instructions", ""))

    def _log_scenes(self, script, raw_scenes, scenes):
        """One line per scene, plus a warning if any length had to be changed."""
        total = sum(scene["duration"] for scene in scenes)
        log.info("  %d scene(s), %.1fs in total:", len(scenes), total)
        changed = 0
        for raw, scene in zip(raw_scenes, scenes):
            note = ""
            if raw["duration"] != scene["duration"]:
                changed += 1
                note = f" (LLM said {raw['duration']:g}s, kept within {self.min_s}-{self.max_s}s)"
            log.info("    scene %d: %.1fs%s | %s", scene["index"], scene["duration"], note,
                     _preview(scene["narration"]))
            log.debug("    scene %d narration: %s", scene["index"], scene["narration"])
            log.debug("    scene %d visual prompt: %s", scene["index"], scene["visual_prompt"])
        if changed:
            log.warning("script #%s: %d of %d scene length(s) were outside %s-%ss and were adjusted",
                        script["id"], changed, len(scenes), self.min_s, self.max_s)

    def _check_scenes_match_script(self, script, script_text, scenes):
        """Warn if the scenes look like they don't cover the script properly.

        Two checks, both warnings only (the scenes are still saved):
          - the scenes' narration should contain about as many words as the script,
          - the scenes' total length should be about as long as reading the script aloud.
        """
        script_words = len(script_text.split())
        scene_words = sum(len(scene["narration"].split()) for scene in scenes)
        log.info("  narration: %d words in the scenes vs %d in the script", scene_words, script_words)
        if _too_different(scene_words, script_words):
            log.warning("script #%s: the scenes' narration has %d words but the script has %d; "
                        "the LLM may have dropped or added text", script["id"], scene_words, script_words)

        words_per_second = self.cfg.get("words_per_second", DEFAULT_WORDS_PER_SECOND)
        spoken_seconds = script_words / words_per_second if words_per_second else 0
        total_seconds = sum(scene["duration"] for scene in scenes)
        log.info("  length: %.1fs of scenes vs ~%.1fs to read the script aloud (at %s words/s)",
                 total_seconds, spoken_seconds, words_per_second)
        if _too_different(total_seconds, spoken_seconds):
            log.warning("script #%s: the scenes last %.1fs but the narration takes ~%.1fs to read; "
                        "video and audio may drift out of sync", script["id"], total_seconds, spoken_seconds)

    def _log_summary(self, batch, scene_files, failures):
        log.info("done: split %d of %d script(s), %d failed", len(scene_files), len(batch), len(failures))
        for row in scene_files:
            log.info("  ok     script #%s -> %s (%s scene(s))", row.get("script_id"), row.get("file_path"),
                     row.get("scene_count"))
        for script, error in failures:
            log.error("  FAILED script #%s (%s): %s: %s", script["id"], script["file_path"],
                      type(error).__name__, error)

    # --- Reading and writing Supabase ------------------------------------

    def _scripts_waiting_for_scenes(self):
        """Scripts that have no row in `scene_files` yet, oldest first."""
        already_split = {row["script_id"] for row in self.db.select("scene_files")}
        scripts = self.db.select("scripts", order="id.asc")
        log.info("found %d script(s) in total, %d already split into scenes", len(scripts), len(already_split))
        return [script for script in scripts if script["id"] not in already_split]

    def _download_script_text(self, script):
        """Fetch the narration text the script generator saved in storage."""
        log.info("  downloading script text: %s", script["file_path"])
        text = self.db.download("scripts", script["file_path"]).decode("utf-8")
        log.info("  script text: %d words, %d characters | %s", len(text.split()), len(text), _preview(text))
        log.debug("script #%s full text:\n%s", script["id"], text)
        return text

    def _save_scene_file(self, script, scenes):
        """Upload the scenes as JSON to storage; returns where they were saved.

        The file goes in the same day folder as its script, named after the
        script, e.g.
            2026-11-05/script_7_the-town-that-vanished.json
        See storage_paths.py for how the other stages name files.
        """
        day, label = script_location(self.db, script)
        file_path = f"{day}/{label}.json"
        contents = json.dumps({"script_id": script["id"], "scenes": scenes}, indent=2)
        log.debug("scene file for script #%s:\n%s", script["id"], contents)
        self.db.upload("scenes", file_path, contents, "application/json")
        return file_path

    def _record_scene_file(self, script, file_path, scenes):
        """Add a row to the `scene_files` table pointing at the saved file."""
        return self.db.insert("scene_files", {
            "script_id": script["id"],
            "file_path": file_path,
            "scene_count": len(scenes),
        })


# --- Checking and tidying the LLM's answer ---------------------------------

def parse_scene_reply(reply):
    """Read the LLM's reply and make sure it is the list of scenes we asked for.

    Expected: a non-empty JSON list where every item is an object with
    non-empty "narration" and "visual_prompt" text and a positive number
    "duration" (a number written as text, like "3", is accepted too).
    A list wrapped as {"scenes": [...]} is unwrapped, since LLMs often do that.

    Returns the scenes with each duration as a float. Raises
    InvalidSceneReply describing every problem found.
    """
    try:
        data = extract_json(reply or "")
    except ValueError as error:  # json.JSONDecodeError is a ValueError
        raise InvalidSceneReply(f"not valid JSON ({error}); reply started with: {_preview(reply)!r}") from None

    if isinstance(data, dict) and isinstance(data.get("scenes"), list):
        log.debug("the LLM wrapped the scene list in {\"scenes\": [...]}; unwrapping it")
        data = data["scenes"]
    if not isinstance(data, list):
        found = f"an object with keys {sorted(data)}" if isinstance(data, dict) else type(data).__name__
        raise InvalidSceneReply(f"expected a JSON list of scenes, got {found}")
    if not data:
        raise InvalidSceneReply("the list of scenes is empty")

    problems, scenes = [], []
    for number, scene in enumerate(data):
        if not isinstance(scene, dict):
            problems.append(f"scene {number} should be an object, got {type(scene).__name__}")
            continue
        for field in TEXT_FIELDS:
            value = scene.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"scene {number} needs non-empty text in {field!r}, got {value!r}")
        duration = _as_seconds(scene.get("duration"))
        if duration is None:
            problems.append(f"scene {number} needs a positive number in 'duration', got {scene.get('duration')!r}")
        scenes.append({**scene, "duration": duration})

    if problems:
        raise InvalidSceneReply(f"{len(problems)} problem(s): " + "; ".join(problems))
    return scenes


def normalize_scenes(scenes, min_s=DEFAULT_MIN_SCENE_SECONDS, max_s=DEFAULT_MAX_SCENE_SECONDS):
    """Turn a checked scene list into clean, numbered scenes.

    - Each scene's duration is kept between `min_s` and `max_s` seconds
      (e.g. with 2-5 s: 1 becomes 2, 9 becomes 5, 3 stays 3). A scene with
      no duration gets `min_s`.
    - Scenes are numbered 0, 1, 2, ... in order.
    - Narration and visual descriptions are trimmed; missing ones become empty text.
    """
    tidy = []
    for index, scene in enumerate(scenes):
        duration = float(scene.get("duration", min_s))
        duration = min(max(duration, min_s), max_s)
        tidy.append({
            "index": index,
            "narration": scene.get("narration", "").strip(),
            "visual_prompt": scene.get("visual_prompt", "").strip(),
            "duration": duration,
        })
    return tidy


# --- Small helpers -----------------------------------------------------------

def _as_seconds(value):
    """`value` as a positive float, or None if it isn't one."""
    if isinstance(value, bool):  # True/False are technically numbers in Python
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 and seconds != float("inf") else None


def _too_different(actual, expected):
    """True if `actual` is more than MISMATCH_TOLERANCE away from `expected`."""
    return expected > 0 and abs(actual - expected) > expected * MISMATCH_TOLERANCE


def _preview(text, limit=PREVIEW_CHARS):
    """The start of `text` on one line, for log messages."""
    one_line = " ".join(str(text or "").split())
    return one_line if len(one_line) <= limit else one_line[:limit - 3] + "..."
