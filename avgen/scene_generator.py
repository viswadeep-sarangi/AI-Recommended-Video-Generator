"""Script -> scenes (2-5 s each) JSON file in storage, logged in a table."""
import json

from .llm import extract_json, wrap_prompt


def normalize_scenes(scenes, min_s=2.0, max_s=5.0):
    """Clamp durations into [min_s, max_s], renumber."""
    out = []
    for s in scenes:
        d = min(max(float(s.get("duration", min_s)), min_s), max_s)
        out.append({"index": len(out), "narration": s.get("narration", ""),
                    "visual_prompt": s.get("visual_prompt", ""), "duration": d})
    return out


class SceneGenerator:
    def __init__(self, llm, db, cfg):
        self.llm, self.db, self.cfg = llm, db, cfg
        self.min_s = cfg.get("min_scene_seconds", 2)
        self.max_s = cfg.get("max_scene_seconds", 5)

    def build_prompt(self, script_text):
        body = (
            f"Split this narration into consecutive video scenes, each {self.min_s}-{self.max_s} seconds long. "
            'Reply ONLY with JSON: [{"narration": str, "visual_prompt": str, "duration": seconds}].\n\n' + script_text
        )
        return wrap_prompt(body, self.cfg.get("prepend_instructions", ""), self.cfg.get("append_instructions", ""))

    def generate(self, script):
        text = self.db.download("scripts", script["file_path"]).decode("utf-8")
        scenes = normalize_scenes(extract_json(self.llm.complete(self.build_prompt(text))), self.min_s, self.max_s)
        path = f"{self.cfg.get('folder', 'scenes')}/script_{script['id']}.json"
        self.db.upload("scenes", path, json.dumps({"script_id": script["id"], "scenes": scenes}, indent=2), "application/json")
        return self.db.insert("scene_files", {"script_id": script["id"], "file_path": path, "scene_count": len(scenes)})

    def run_pending(self):
        done = {r["script_id"] for r in self.db.select("scene_files")}
        return [self.generate(s) for s in self.db.select("scripts") if s["id"] not in done]
