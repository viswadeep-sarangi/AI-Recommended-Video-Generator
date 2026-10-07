"""Finds scenes that have no generated clip yet and generates them."""
import json
from datetime import datetime, timezone


class VideoPipeline:
    def __init__(self, db, generator, cfg=None):
        self.db, self.gen, self.cfg = db, generator, cfg or {}

    def pending(self):
        done = {(r["scene_file_id"], r["scene_index"]) for r in self.db.select("video_scenes", {"status": "eq.done"})}
        for sf in self.db.select("scene_files"):
            scenes = json.loads(self.db.download("scenes", sf["file_path"]))["scenes"]
            for s in scenes:
                if (sf["id"], s["index"]) not in done:
                    yield sf, s

    def run(self, limit=None):
        n, results = 0, []
        for sf, scene in self.pending():
            if limit and n >= limit:
                break
            n += 1
            category = self._category(sf)
            path = f"{self.cfg.get('folder', 'video_scenes')}/script_{sf['script_id']}/scene_{scene['index']:03d}.mp4"
            try:
                provider, data = self.gen.generate(scene["visual_prompt"], scene["duration"], category)
                self.db.upload("videos", path, data, "video/mp4")
                row = {"status": "done", "file_path": path, "provider": provider}
            except Exception as e:
                row = {"status": "failed", "file_path": None, "provider": None, "error": str(e)[:500]}
            row.update(scene_file_id=sf["id"], scene_index=scene["index"], script_id=sf["script_id"],
                       created_at=datetime.now(timezone.utc).isoformat())
            results.append(self.db.insert("video_scenes", row))
        return results

    def _category(self, sf):
        try:
            script = self.db.select("scripts", {"id": f"eq.{sf['script_id']}"})[0]
            return self.db.select("topics", {"id": f"eq.{script['topic_id']}"})[0].get("category")
        except Exception:
            return None
