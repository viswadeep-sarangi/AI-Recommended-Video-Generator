import argparse
import logging
import time

from .assembler import Assembler
from .audio import AudioGenerator
from .config import load_config
from .llm import create_llm
from .log import log_group, setup_logging
from .scene_generator import SceneGenerator
from .script_generator import ScriptGenerator
from .scout import RecommendationFinder, create_notifier
from .supabase_client import SupabaseClient
from .video import create_video_provider
from .video_pipeline import VideoPipeline

log = logging.getLogger("avgen")


def main(argv=None):
    p = argparse.ArgumentParser(prog="avgen")
    p.add_argument("stage", choices=["scout", "scripts", "scenes", "videos", "audio", "assemble", "all"])
    p.add_argument("--config")
    p.add_argument("--log-level", help="DEBUG, INFO, WARNING or ERROR (default: $AVGEN_LOG_LEVEL or INFO)")
    p.add_argument("-v", "--verbose", action="store_true", help="shorthand for --log-level DEBUG")
    a = p.parse_args(argv)
    setup_logging("DEBUG" if a.verbose else a.log_level)
    cfg = load_config(a.config)
    db = SupabaseClient(cfg["supabase"])
    stages = ["scout", "scripts", "scenes", "videos", "audio", "assemble"] if a.stage == "all" else [a.stage]
    for st in stages:
        with log_group(f"stage: {st}"):
            started = time.monotonic()
            if st == "scout":
                res = RecommendationFinder(create_llm(cfg["scout"]["llm"]), db, cfg["scout"], create_notifier(cfg.get("notifications"))).run()
            elif st == "scripts":
                res = ScriptGenerator(create_llm(cfg["script"]["llm"]), db, cfg["script"]).run_pending()
            elif st == "scenes":
                res = SceneGenerator(create_llm(cfg["scenes"]["llm"]), db, cfg["scenes"]).run_pending()
            elif st == "videos":
                res = VideoPipeline(db, create_video_provider(cfg["video"]["provider"]), cfg["video"]).run()
            elif st == "audio":
                res = AudioGenerator(db, cfg["audio"]).run_pending()
            else:
                res = Assembler(db, cfg.get("assemble")).run_pending()
            log.info("%s: %d item(s) in %.1fs", st, len(res), time.monotonic() - started)


if __name__ == "__main__":
    main()
