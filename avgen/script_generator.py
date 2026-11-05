"""Script generator: turns trending topics into narration scripts.

For each topic waiting to be scripted, it:
  1. asks the LLM to write a short narration script,
  2. saves the script as a text file in Supabase storage,
  3. records the script in the `scripts` table,
  4. marks the topic as "scripted" so it isn't picked again.
"""
import logging
from datetime import datetime, timezone

from .llm import wrap_prompt
from .storage_paths import day_folder, title_snippet

log = logging.getLogger(__name__)

# Defaults used when the "script" section of the config leaves a setting out.
DEFAULT_TARGET_SECONDS = 30
DEFAULT_WORDS_PER_SECOND = 2.5
DEFAULT_MAX_PER_RUN = 3

# Warn when a script's length is further than this from the target word count.
WORD_COUNT_TOLERANCE = 0.3


class ScriptGenerator:
    def __init__(self, llm, db, cfg):
        self.llm = llm  # the language model that writes the scripts
        self.db = db    # Supabase: tables and file storage
        self.cfg = cfg  # the "script" section of the config

    # --- Main entry point -------------------------------------------------

    def run_pending(self, limit=None):
        """Write scripts for the best-scoring topics that don't have one yet."""
        how_many = limit or self.cfg.get("max_per_run", DEFAULT_MAX_PER_RUN)
        log.info("settings: max_per_run=%s target_seconds=%s words_per_second=%s",
                 how_many, self.cfg.get("target_seconds", DEFAULT_TARGET_SECONDS),
                 self.cfg.get("words_per_second", DEFAULT_WORDS_PER_SECOND))

        topics = self._topics_waiting_for_a_script(how_many)
        if not topics:
            log.info("no topics with status 'new'; nothing to script")
            return []
        log.info("picked %d topic(s) to script:", len(topics))
        for topic in topics:
            log.info("  topic #%s (score %s): %s", topic["id"], topic.get("score"), topic["title"])

        scripts = []
        for number, topic in enumerate(topics, start=1):
            log.info("[%d/%d] scripting topic #%s: %s", number, len(topics), topic["id"], topic["title"])
            try:
                scripts.append(self.generate(topic))
            except Exception:
                log.error("[%d/%d] failed on topic #%s; it stays 'new' and will be retried next run "
                          "(%d script(s) saved before this)", number, len(topics), topic["id"], len(scripts))
                raise
        log.info("done: wrote %d script(s)", len(scripts))
        return scripts

    def generate(self, topic, seconds=None):
        """Write, save and record a script for one topic."""
        seconds = seconds or self.cfg.get("target_seconds", DEFAULT_TARGET_SECONDS)

        prompt = self.build_prompt(topic, seconds)
        log.debug("LLM prompt for topic #%s:\n%s", topic["id"], prompt)
        script_text = self.llm.complete(prompt).strip()
        log.debug("LLM script for topic #%s:\n%s", topic["id"], script_text)
        self._check_length(topic, script_text, seconds)

        file_path = self._save_script_file(topic, script_text)
        log.info("  saved to the scripts bucket: %s", file_path)
        script_row = self._record_script(topic, file_path, seconds)
        log.info("  recorded as script #%s", script_row.get("id"))
        self._mark_topic_scripted(topic)
        log.info("  topic #%s marked 'scripted'", topic["id"])
        return script_row

    def _check_length(self, topic, script_text, seconds):
        """Log the script's length, warning if it's far from the target."""
        words = len(script_text.split())
        target = int(seconds * self.cfg.get("words_per_second", DEFAULT_WORDS_PER_SECOND))
        log.info("  script: %d words (target ~%d), %d characters", words, target, len(script_text))
        if target and abs(words - target) > target * WORD_COUNT_TOLERANCE:
            log.warning("script for topic #%s is %d words, far from the ~%d target; "
                        "its narration will run noticeably longer or shorter than %ss",
                        topic["id"], words, target, seconds)

    # --- The prompt -------------------------------------------------------

    def build_prompt(self, topic, seconds):
        """The request sent to the LLM, wrapped in the configured style notes."""
        words_per_second = self.cfg.get("words_per_second", DEFAULT_WORDS_PER_SECOND)
        target_words = int(seconds * words_per_second)

        request = (
            f"Write a narration script for a video of about {seconds} seconds "
            f"(~{target_words} words) on the topic:\n"
            f"{topic['title']}\n"
            f"{topic.get('description') or ''}\n"
            "Output only the spoken narration text."
        )

        style_notes = self.cfg.get("context", "")             # goes before the request
        extra_rules = self.cfg.get("append_instructions", "")  # goes after the request
        return wrap_prompt(request, style_notes, extra_rules)

    # --- Reading and writing Supabase ------------------------------------

    def _topics_waiting_for_a_script(self, how_many):
        """Topics still marked "new", highest score first."""
        return self.db.select("topics", {"status": "eq.new"}, order="score.desc", limit=how_many)

    def _save_script_file(self, topic, script_text):
        """Upload the script text to storage; returns where it was saved.

        Scripts are grouped into one folder per day (UTC) inside the
        `scripts` bucket, named after the topic, e.g.
            2026-11-05/topic_42_the-town-that-vanished_070000Z.txt
        The time (UTC) keeps two runs on the same day from overwriting each
        other. See storage_paths.py for how the other stages name files.
        """
        now = datetime.now(timezone.utc)
        file_name = f"topic_{topic['id']}_{title_snippet(topic['title'])}_{now.strftime('%H%M%SZ')}.txt"
        file_path = f"{day_folder(now)}/{file_name}"
        self.db.upload("scripts", file_path, script_text, "text/plain")
        return file_path

    def _record_script(self, topic, file_path, seconds):
        """Add a row to the `scripts` table pointing at the saved file."""
        return self.db.insert("scripts", {
            "topic_id": topic["id"],
            "file_path": file_path,
            "target_seconds": seconds,
        })

    def _mark_topic_scripted(self, topic):
        """Change the topic's status so the next run skips it."""
        self.db.update("topics", {"id": f"eq.{topic['id']}"}, {"status": "scripted"})
