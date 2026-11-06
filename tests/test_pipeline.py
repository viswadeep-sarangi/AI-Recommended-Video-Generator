import json

import pytest

from avgen.config import _resolve
from avgen.llm import LLM, extract_json, wrap_prompt
from avgen.scene_generator import (InvalidSceneReply, SceneGenerationError, SceneGenerator, normalize_scenes,
                                   parse_scene_reply)
from avgen.scout import SOURCES, RecommendationFinder, fetch_youtube
from avgen.video import create_video_provider
from avgen.video_pipeline import VideoGenerationError, VideoPipeline


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

    def update(self, t, filters, values):
        rows = self.select(t, filters)
        for row in rows:
            row.update(values)
        return [dict(row) for row in rows]

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


GOOD_SCENES = '[{"narration":"a b","visual_prompt":"p1","duration":3},{"narration":"c","visual_prompt":"p2","duration":"4"}]'


def test_scene_reply_accepts_expected_shape():
    scenes = parse_scene_reply("```json\n" + GOOD_SCENES + "\n```")
    assert [s["duration"] for s in scenes] == [3.0, 4.0]
    wrapped = parse_scene_reply('{"scenes": ' + GOOD_SCENES + "}")
    assert [s["visual_prompt"] for s in wrapped] == ["p1", "p2"]


@pytest.mark.parametrize("reply", [
    "Sure! Here are your scenes.",                                      # not JSON
    '{"narration": "a"}',                                               # object, not a list
    "[]",                                                               # empty list
    '["just text"]',                                                    # item isn't an object
    '[{"narration": "a", "duration": 3}]',                              # no visual_prompt
    '[{"narration": " ", "visual_prompt": "p", "duration": 3}]',        # blank narration
    '[{"narration": "a", "visual_prompt": "p", "duration": "soon"}]',   # duration not a number
    '[{"narration": "a", "visual_prompt": "p", "duration": 0}]',        # duration not positive
    '[{"narration": "a", "visual_prompt": "p", "duration": true}]',     # duration is a boolean
])
def test_scene_reply_rejects_unexpected_shapes(reply):
    with pytest.raises(InvalidSceneReply):
        parse_scene_reply(reply)


def _db_with_scripts(count, missing_file_ids=()):
    db = FakeDB()
    for n in range(1, count + 1):
        db.insert("scripts", {"topic_id": None, "file_path": f"s{n}.txt"})
        if n not in missing_file_ids:
            db.upload("scripts", f"s{n}.txt", b"a b c")
    return db


def test_scene_generator_skips_a_failing_script_and_keeps_going():
    db = _db_with_scripts(3, missing_file_ids={2})
    rows = SceneGenerator(FakeLLM(GOOD_SCENES), db, {}).run_pending()
    assert [r["script_id"] for r in rows] == [1, 3]
    assert [r["script_id"] for r in db.tables["scene_files"]] == [1, 3]  # script 2 is retried next run


def test_scene_generator_respects_max_per_run():
    db = _db_with_scripts(3)
    gen = SceneGenerator(FakeLLM(GOOD_SCENES), db, {"max_per_run": 2})
    assert [r["script_id"] for r in gen.run_pending()] == [1, 2]
    assert [r["script_id"] for r in gen.run_pending()] == [3]
    assert gen.run_pending() == []


def test_scene_generator_asks_again_after_an_unusable_reply():
    class FlakyLLM(FakeLLM):
        def complete(self, prompt, system=""):
            self.prompts.append(prompt)
            return "not json" if len(self.prompts) == 1 else GOOD_SCENES

    llm = FlakyLLM(None)
    rows = SceneGenerator(llm, _db_with_scripts(1), {}).run_pending()
    assert len(rows) == 1 and len(llm.prompts) == 2


def test_scene_generator_raises_when_every_script_fails():
    db = _db_with_scripts(2)
    with pytest.raises(SceneGenerationError):
        SceneGenerator(FakeLLM("not json"), db, {"llm_attempts": 2}).run_pending()
    assert "scene_files" not in db.tables


def _db_with_scene_files(count, scenes='[{"narration":"a","visual_prompt":"p1","duration":3},'
                                       '{"narration":"b","visual_prompt":"p2","duration":3}]'):
    db = FakeDB()
    for number in range(count):
        db.insert("scripts", {"topic_id": 1, "file_path": f"s{number}.txt"})
        db.upload("scripts", f"s{number}.txt", b"hi")
    SceneGenerator(FakeLLM(scenes), db, {}).run_pending()
    return db


class FakeProvider:
    """Makes clips, except for prompts in `fail_on` (each failing `times` times, forever if None)."""
    name = "fake"

    def __init__(self, fail_on=(), times=None):
        self.fail_on, self.times, self.failures = set(fail_on), times, {}

    def generate(self, prompt, duration):
        if prompt in self.fail_on and (self.times is None or self.failures.get(prompt, 0) < self.times):
            self.failures[prompt] = self.failures.get(prompt, 0) + 1
            raise RuntimeError("busy")
        return b"mp4"


def test_video_pipeline_does_one_scene_file_per_run_and_retries_on_the_same_row():
    db = _db_with_scene_files(2)
    pipe = VideoPipeline(db, FakeProvider(fail_on={"p2"}, times=1))

    first = pipe.run()  # only the first scene file; scene p2 fails
    assert [(r["script_id"], r["status"], r["attempts"]) for r in first] == [(1, "done", 1), (1, "failed", 1)]
    assert first[0]["provider"] == "fake" and first[0]["file_path"].endswith("/scene_000.mp4")

    second = pipe.run()  # still the first file: only the failed scene is retried
    assert [(r["script_id"], r["scene_index"], r["status"], r["attempts"]) for r in second] == [(1, 1, "done", 2)]
    assert len(db.tables["video_scenes"]) == 2  # the retry updated the scene's row, no new one

    third = pipe.run()  # now the second file
    assert [(r["script_id"], r["status"]) for r in third] == [(2, "done"), (2, "done")]
    assert VideoPipeline(db, FakeProvider(), {"max_scene_files_per_run": 5}).run() == []


def test_video_pipeline_gives_up_on_a_scene_after_max_attempts():
    db = _db_with_scene_files(2)
    pipe = VideoPipeline(db, FakeProvider(fail_on={"p2"}), {"max_attempts_per_scene": 2})

    pipe.run()  # file 1: p1 done, p2 attempt 1 fails
    with pytest.raises(VideoGenerationError):
        pipe.run()  # file 1: p2 attempt 2 fails, the only scene tried
    third = pipe.run()  # file 1 is given up on, so file 2 is next, not stuck behind it
    assert {r["script_id"] for r in third} == {2}
    rows = [r for r in db.tables["video_scenes"] if r["script_id"] == 1]
    assert sorted((r["scene_index"], r["status"], r["attempts"]) for r in rows) == [(0, "done", 1), (1, "failed", 2)]

    # raising the limit makes the given-up scene worth trying again
    retry = VideoPipeline(db, FakeProvider(), {"max_attempts_per_scene": 3}).run()
    assert [(r["script_id"], r["scene_index"], r["status"], r["attempts"]) for r in retry] == [(1, 1, "done", 3)]


def test_video_pipeline_raises_when_every_scene_fails():
    db = _db_with_scene_files(1, '[{"narration":"a","visual_prompt":"p","duration":3}]')
    with pytest.raises(VideoGenerationError):
        VideoPipeline(db, FakeProvider(fail_on={"p"})).run()
    assert db.tables["video_scenes"][0]["status"] == "failed"


class FakeReply:
    def __init__(self, data=None, content=b""):
        self.data, self.content, self.status_code, self.text = data, content, 200, ""

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def _fake_seedance(monkeypatch, statuses, final=None):
    """Point the Seedance provider at fake HTTP calls; returns what was sent."""
    from avgen.video import seedance

    sent = {}
    statuses = iter(statuses)

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, body=json, auth=headers["Authorization"])
        return FakeReply({"id": "cgt-1"})

    def fake_get(url, headers=None, timeout=None):
        if url == "https://cdn/clip.mp4":
            return FakeReply(content=b"mp4 bytes")
        assert url.endswith("/contents/generations/tasks/cgt-1")
        status = next(statuses)
        job = {"id": "cgt-1", "status": status}
        if status == "succeeded":
            job.update(content={"video_url": "https://cdn/clip.mp4"}, usage={"total_tokens": 49005})
        if status == "failed":
            job.update(final or {})
        return FakeReply(job)

    monkeypatch.setattr(seedance.requests, "post", fake_post)
    monkeypatch.setattr(seedance.requests, "get", fake_get)
    return sent


def test_seedance_sends_text_only_request_and_returns_the_clip(monkeypatch):
    sent = _fake_seedance(monkeypatch, ["queued", "running", "succeeded"])
    provider = create_video_provider({"type": "seedance", "api_key": "key", "poll_seconds": 0,
                                      "usd_per_million_tokens": 1.0})

    assert provider.name == "seedance"
    assert provider.generate("a foggy town", 3.5) == b"mp4 bytes"
    assert sent["auth"] == "Bearer key"
    assert sent["body"]["model"] == "seedance-1-0-pro-fast-251015"
    assert sent["body"]["content"] == [{"type": "text", "text": "a foggy town --ratio 9:16 --resolution 480p "
                                                                 "--duration 4 --camerafixed false"}]  # 3.5s -> 4s


def test_seedance_keeps_clip_length_within_limits_and_reports_failures(monkeypatch):
    sent = _fake_seedance(monkeypatch, ["failed"], {"error": {"code": "SensitiveContent", "message": "rejected"}})
    provider = create_video_provider({"type": "seedance", "api_key": "key", "poll_seconds": 0, "min_seconds": 4})

    with pytest.raises(RuntimeError, match="SensitiveContent rejected"):
        provider.generate("a foggy town", 2)
    assert "--duration 4 " in sent["body"]["content"][0]["text"]  # 2s raised to min_seconds
