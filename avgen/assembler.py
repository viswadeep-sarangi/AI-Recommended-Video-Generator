"""Combine scene clips + narration + music into one video with ffmpeg."""
import subprocess
import tempfile
from pathlib import Path


def build_ffmpeg_cmd(list_file, narration, music, out, music_volume=0.15):
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-i", str(narration)]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music), "-filter_complex",
                f"[2:a]volume={music_volume}[m];[1:a][m]amix=inputs=2:duration=first[a]", "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    return cmd + ["-c:v", "libx264", "-c:a", "aac", "-shortest", str(out)]


class Assembler:
    def __init__(self, db, cfg=None):
        self.db, self.cfg = db, cfg or {}

    def assemble(self, script):
        clips = sorted((r for r in self.db.select("video_scenes", {"script_id": f"eq.{script['id']}", "status": "eq.done"})),
                       key=lambda r: r["scene_index"])
        audio = self.db.select("audio_files", {"script_id": f"eq.{script['id']}"})
        if not clips or not audio:
            return None
        audio = audio[0]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lines = []
            for c in clips:
                p = tmp / f"s{c['scene_index']:03d}.mp4"
                p.write_bytes(self.db.download("videos", c["file_path"]))
                lines.append(f"file '{p}'")
            (tmp / "list.txt").write_text("\n".join(lines))
            (tmp / "n.mp3").write_bytes(self.db.download("audio", audio["narration_path"]))
            music = None
            if audio.get("music_path"):
                music = tmp / "m.mp3"
                music.write_bytes(self.db.download("audio", audio["music_path"]))
            out = tmp / "final.mp4"
            subprocess.run(build_ffmpeg_cmd(tmp / "list.txt", tmp / "n.mp3", music, out,
                                            self.cfg.get("music_volume", 0.15)), check=True, capture_output=True)
            path = f"{self.cfg.get('folder', 'final')}/script_{script['id']}.mp4"
            self.db.upload("final", path, out.read_bytes(), "video/mp4")
        return self.db.insert("final_videos", {"script_id": script["id"], "file_path": path})

    def run_pending(self):
        done = {r["script_id"] for r in self.db.select("final_videos")}
        results = []
        for s in self.db.select("scripts"):
            if s["id"] in done:
                continue
            scene_files = self.db.select("scene_files", {"script_id": f"eq.{s['id']}"})
            if not scene_files:
                continue
            clips = self.db.select("video_scenes", {"script_id": f"eq.{s['id']}", "status": "eq.done"})
            if len(clips) < scene_files[0]["scene_count"]:
                continue  # wait until every scene has a clip
            r = self.assemble(s)
            if r:
                results.append(r)
        return results
