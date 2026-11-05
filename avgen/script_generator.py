"""Script generator: turns trending topics into narration scripts.

For each topic waiting to be scripted, it:
  1. asks the LLM to write a short narration script,
  2. saves the script as a text file in Supabase storage,
  3. records the script in the `scripts` table,
  4. marks the topic as "scripted" so it isn't picked again.
"""
from datetime import datetime, timezone

from .llm import wrap_prompt

# Defaults used when the "script" section of the config leaves a setting out.
DEFAULT_TARGET_SECONDS = 30
DEFAULT_WORDS_PER_SECOND = 2.5
DEFAULT_MAX_PER_RUN = 3
DEFAULT_FOLDER = "scripts"


class ScriptGenerator:
    def __init__(self, llm, db, cfg):
        self.llm = llm  # the language model that writes the scripts
        self.db = db    # Supabase: tables and file storage
        self.cfg = cfg  # the "script" section of the config

    # --- Main entry point -------------------------------------------------

    def run_pending(self, limit=None):
        """Write scripts for the best-scoring topics that don't have one yet."""
        topics = self._topics_waiting_for_a_script(limit or self.cfg.get("max_per_run", DEFAULT_MAX_PER_RUN))
        return [self.generate(topic) for topic in topics]

    def generate(self, topic, seconds=None):
        """Write, save and record a script for one topic."""
        seconds = seconds or self.cfg.get("target_seconds", DEFAULT_TARGET_SECONDS)

        prompt = self.build_prompt(topic, seconds)
        script_text = self.llm.complete(prompt).strip()

        file_path = self._save_script_file(topic, script_text)
        script_row = self._record_script(topic, file_path, seconds)
        self._mark_topic_scripted(topic)
        return script_row

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

        Example path: scripts/topic_42_20261104T090000Z.txt
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = self.cfg.get("folder", DEFAULT_FOLDER)
        file_path = f"{folder}/topic_{topic['id']}_{timestamp}.txt"
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
