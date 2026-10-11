import json

import pytest

from avgen.config import _resolve
from avgen.llm import LLM, extract_json, wrap_prompt
from avgen.scene_generator import SceneGenerator, normalize_scenes
from avgen.scout import RecommendationFinder
from avgen.video import VideoGenerator
from avgen.video_pipeline import VideoPipeline


class FakeDB:
    def __init__(self):
        self.tables, self.files = {}, {}

    def insert(self, t, row):
        rows = self.tables.setdefault(t, [])
        row = {"id": len(rows) + 1, **row}
        rows.append(row)
        return row

    def select(self, t, filters=None, order=None, limit=None):
        rows = self.tables.get(t, [])
        for k, v in (filters or {}).items():
            rows = [r for r in rows if str(r.get(k)) == v.split(".", 1)[1]]
        return rows

    def update(self, *a):
        pass

    def upload(self, b, p, data, ct=None):
        self.files[(b, p)] = data

    def download(self, b, p):
        return self.files[(b, p)]


class FakeLLM(LLM):
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, prompt, system=""):
        self.prompts.append(prompt)
        return self.reply


def test_helpers():
    assert _resolve({"a": "env:NOPE_XYZ"}) == {"a": ""}
    assert extract_json('```json\n[1]\n```') == [1]
    assert wrap_prompt("b", "a", "c") == "a\n\nb\n\nc"


def test_scout_wraps_and_stores():
    llm = FakeLLM('[{"title": "T", "score": 5}]')
    db = FakeDB()
    cfg = {"prepend_instructions": "PRE", "append_instructions": "POST", "sources": []}
    rows = RecommendationFinder(llm, db, cfg).run()
    assert rows[0]["title"] == "T" and db.tables["topics"]
    assert llm.prompts[0].startswith("PRE") and llm.prompts[0].endswith("POST")


def test_scene_durations_clamped():
    s = normalize_scenes([{"duration": 1}, {"duration": 9}, {"duration": 3}])
    assert [x["duration"] for x in s] == [2, 5, 3] and [x["index"] for x in s] == [0, 1, 2]


def test_video_pipeline_generates_only_missing_and_falls_back():
    db = FakeDB()
    db.insert("scripts", {"topic_id": 1, "file_path": "s.txt"})
    db.upload("scripts", "s.txt", b"hi")
    scenes = '[{"narration":"a","visual_prompt":"p1","duration":3},{"narration":"b","visual_prompt":"p2","duration":3}]'
    SceneGenerator(FakeLLM(scenes), db, {}).run_pending()

    class Bad:
        def generate(self, p, d):
            raise RuntimeError("out of tokens")

    class Good:
        def generate(self, p, d):
            return b"mp4"

    gen = VideoGenerator({"default": ["bad", "good"]}, {"bad": Bad(), "good": Good()})
    pipe = VideoPipeline(db, gen)
    assert len(pipe.run(limit=1)) == 1
    assert db.tables["video_scenes"][0]["provider"] == "good"
    assert len(pipe.run()) == 1  # only the remaining scene
    assert pipe.run() == []
    with pytest.raises(RuntimeError):
        VideoGenerator({"default": ["bad"]}, {"bad": Bad()}).generate("x", 3)
