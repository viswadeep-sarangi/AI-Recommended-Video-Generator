"""Topic -> script (stored in Supabase storage, logged in a table)."""
from datetime import datetime, timezone

from .llm import wrap_prompt


class ScriptGenerator:
    def __init__(self, llm, db, cfg):
        self.llm, self.db, self.cfg = llm, db, cfg

    def build_prompt(self, topic, seconds):
        body = (
            f"Write a narration script for a video of about {seconds} seconds "
            f"(~{int(seconds * self.cfg.get('words_per_second', 2.5))} words) on the topic:\n"
            f"{topic['title']}\n{topic.get('description', '')}\n"
            "Output only the spoken narration text."
        )
        return wrap_prompt(body, self.cfg.get("context", ""), self.cfg.get("append_instructions", ""))

    def generate(self, topic, seconds=None):
        seconds = seconds or self.cfg.get("target_seconds", 30)
        text = self.llm.complete(self.build_prompt(topic, seconds)).strip()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = f"{self.cfg.get('folder', 'scripts')}/topic_{topic['id']}_{stamp}.txt"
        self.db.upload("scripts", path, text, "text/plain")
        row = self.db.insert("scripts", {"topic_id": topic["id"], "file_path": path, "target_seconds": seconds})
        self.db.update("topics", {"id": f"eq.{topic['id']}"}, {"status": "scripted"})
        return row

    def run_pending(self, limit=None):
        topics = self.db.select("topics", {"status": "eq.new"}, order="score.desc", limit=limit or self.cfg.get("max_per_run", 3))
        return [self.generate(t) for t in topics]
