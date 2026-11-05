"""Finds scenes that have no generated clip yet and generates them."""
import json
from datetime import datetime, timezone

from .storage_paths import script_location


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
        n, results, folders = 0, [], {}
        for sf, scene in self.pending():
            if limit and n >= limit:
                break
            n += 1
            category = self._category(sf)
            if sf["script_id"] not in folders:
                folders[sf["script_id"]] = self._clip_folder(sf)
            # e.g. 2026-11-05/script_7_the-town-that-vanished/scene_000.mp4
            path = f"{folders[sf['script_id']]}/scene_{scene['index']:03d}.mp4"
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

    def _clip_folder(self, sf):
        """Folder for one script's clips, e.g. 2026-11-05/script_7_the-town-that-vanished."""
        rows = self.db.select("scripts", {"id": f"eq.{sf['script_id']}"})
        script = rows[0] if rows else {"id": sf["script_id"]}
        day, label = script_location(self.db, script)
        return f"{day}/{label}"

    def _category(self, sf):
        try:
            script = self.db.select("scripts", {"id": f"eq.{sf['script_id']}"})[0]
            return self.db.select("topics", {"id": f"eq.{script['topic_id']}"})[0].get("category")
        except Exception:
            return None
