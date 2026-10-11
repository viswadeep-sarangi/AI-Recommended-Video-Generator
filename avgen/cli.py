import argparse

from .assembler import Assembler
from .audio import AudioGenerator
from .config import load_config
from .llm import create_llm
from .scene_generator import SceneGenerator
from .script_generator import ScriptGenerator
from .scout import RecommendationFinder, create_notifier
from .supabase_client import SupabaseClient
from .video import VideoGenerator
from .video_pipeline import VideoPipeline


def main(argv=None):
    p = argparse.ArgumentParser(prog="avgen")
    p.add_argument("stage", choices=["scout", "scripts", "scenes", "videos", "audio", "assemble", "all"])
    p.add_argument("--config")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    db = SupabaseClient(cfg["supabase"])
    stages = ["scout", "scripts", "scenes", "videos", "audio", "assemble"] if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "scout":
            res = RecommendationFinder(create_llm(cfg["scout"]["llm"]), db, cfg["scout"], create_notifier(cfg.get("notifications"))).run()
        elif st == "scripts":
            res = ScriptGenerator(create_llm(cfg["script"]["llm"]), db, cfg["script"]).run_pending()
        elif st == "scenes":
            res = SceneGenerator(create_llm(cfg["scenes"]["llm"]), db, cfg["scenes"]).run_pending()
        elif st == "videos":
            res = VideoPipeline(db, VideoGenerator(cfg["video"]), cfg["video"]).run(cfg["video"].get("max_per_run"))
        elif st == "audio":
            res = AudioGenerator(db, cfg["audio"]).run_pending()
        else:
            res = Assembler(db, cfg.get("assemble")).run_pending()
        print(f"{st}: {len(res)} item(s)")


if __name__ == "__main__":
    main()
