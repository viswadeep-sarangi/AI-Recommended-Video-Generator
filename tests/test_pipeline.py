import json

import pytest

from avgen.config import _resolve
from avgen.llm import LLM, extract_json, wrap_prompt
from avgen.scene_generator import SceneGenerator, normalize_scenes
from avgen.scout import SOURCES, RecommendationFinder, fetch_youtube
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


def test_scout_wraps_and_stores(monkeypatch):
    monkeypatch.setitem(SOURCES, "youtube", lambda src, days: [{"video_id": "v1", "title": "Trend"}])
    llm = FakeLLM('[{"title": "T", "score": 5}]')
    db = FakeDB()
    cfg = {"prepend_instructions": "PRE", "append_instructions": "POST", "sources": [{"type": "youtube"}]}
    rows = RecommendationFinder(llm, db, cfg).run()
    assert rows[0]["title"] == "T" and db.tables["topics"]
    assert llm.prompts[0].startswith("PRE") and llm.prompts[0].endswith("POST")


def test_storage_title_snippet():
    from avgen.storage_paths import title_snippet

    assert title_snippet("The Town That Vanished Overnight!") == "the-town-that-vanished"
    assert title_snippet("Café secrets: 5 facts") == "cafe-secrets-5-facts"
    assert title_snippet("???") == "untitled"
    assert len(title_snippet("x" * 100)) == 30


def test_storage_script_location_uses_script_day_and_topic():
    from avgen.storage_paths import script_location

    db = FakeDB()
    db.insert("topics", {"title": "The Town That Vanished Overnight"})
    script = {"id": 7, "topic_id": 1, "created_at": "2026-11-05T07:00:01.123456+00:00"}
    assert script_location(db, script) == ("2026-11-05", "script_7_the-town-that-vanished")


def test_scout_stops_without_signals():
    llm = FakeLLM('[{"title": "T", "score": 5}]')
    db = FakeDB()
    with pytest.raises(RuntimeError, match="no trending signals"):
        RecommendationFinder(llm, db, {"sources": []}).run()
    assert llm.prompts == [] and "topics" not in db.tables


def test_fetch_youtube_fetches_and_normalizes_video_metadata(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    calls = []

    def get(url, params, timeout):
        calls.append((url, params, timeout))
        if "search" in url:
            return Response({"items": [{"id": {"videoId": "abc123"}, "snippet": {"title": "ignored here"}}]})
        return Response({"items": [{
            "id": "abc123",
            "snippet": {
                "title": "Full title", "description": "Full description",
                "channelId": "channel-id", "channelTitle": "Uploader",
                "publishedAt": "2025-01-02T03:04:05Z",
                "tags": ["one"], "thumbnails": {"default": {"url": "https://img"}},
            },
            "contentDetails": {"duration": "PT1M2S", "caption": "true"},
            "statistics": {"viewCount": "123", "likeCount": "10", "commentCount": "2"},
            "status": {"privacyStatus": "public"},
        }]})

    monkeypatch.setattr("avgen.scout.requests.get", get)
    videos = fetch_youtube({"api_key": "key", "queries": ["science"], "videoDuration": "short"}, 30)

    assert len(videos) == 1
    assert videos[0]["title"] == "Full title"
    assert videos[0]["description"] == "Full description"
    assert videos[0]["channel_title"] == "Uploader"
    assert videos[0]["published_at"] == "2025-01-02T03:04:05Z"
    assert videos[0]["view_count"] == 123 and videos[0]["like_count"] == 10
    assert videos[0]["dislike_count"] is None
    assert videos[0]["raw_metadata"]["id"] == "abc123"
    assert len(calls) == 2 and "abc123" in calls[1][1]["id"]
    assert calls[0][1]["videoDuration"] == "short"


def test_audio_language_filter_keeps_matching_and_undeclared():
    from avgen.scout import _filter_audio_language

    def video(vid, lang):
        snippet = {"defaultAudioLanguage": lang} if lang else {}
        return {"video_id": vid, "raw_metadata": {"snippet": snippet}}

    videos = [video("a", "en"), video("b", "en-US"), video("c", "hi"), video("d", None)]
    kept = _filter_audio_language(videos, ["en"])
    assert [v["video_id"] for v in kept] == ["a", "b", "d"]
    assert _filter_audio_language(videos, None) == videos


def test_scout_keeps_youtube_signals_when_storage_fails(monkeypatch):
    class FailingDB(FakeDB):
        def upsert(self, *a):
            raise RuntimeError("HTTP 403: permission denied")

    video = {"video_id": "v1", "title": "Trend", "view_count": 5}
    monkeypatch.setitem(SOURCES, "youtube", lambda src, days: [video])
    llm = FakeLLM('[{"title": "T", "score": 5}]')
    db = FailingDB()
    rows = RecommendationFinder(llm, db, {"sources": [{"type": "youtube"}]}).run()
    assert rows and "[youtube] Trend" in llm.prompts[0]


def test_check_response_includes_body():
    import requests

    from avgen.log import check_response

    class Response:
        status_code, text = 403, '{"code":"42501","message":"permission denied for table youtube_videos"}'

        def raise_for_status(self):
            raise requests.HTTPError("403 Client Error: Forbidden for url: https://x?key=SECRET")

    with pytest.raises(requests.HTTPError) as e:
        check_response(Response(), "Supabase upsert")
    assert "42501" in str(e.value) and "SECRET" not in str(e.value)


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
