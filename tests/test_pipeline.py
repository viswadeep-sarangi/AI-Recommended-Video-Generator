import json

import pytest

from avgen.audio import Narration, Transcript, create_transcriber, create_voice
from avgen.audio_generator import AudioGenerationError, AudioGenerator
from avgen.captions import ass_colour, group_words, to_ass, to_srt
from avgen import assembler
from avgen.assembler import Assembler, AssemblyError, build_ffmpeg_cmd, plan_timing
from avgen.config import _resolve
from avgen.llm import LLM, extract_json, wrap_prompt
from avgen.scene_generator import (InvalidSceneReply, SceneGenerationError, SceneGenerator, normalize_scenes,
                                   parse_scene_reply)
from avgen.scout import SOURCES, RecommendationFinder, fetch_youtube
from avgen.storage_paths import day_folder
from avgen.video import Clip, create_video_provider
from avgen.video_generator import VideoGenerationError, VideoGenerator


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
        return Clip(b"mp4", model="m", resolution="480p", ratio="9:16", seconds=5, tokens=49005, cost_usd=0.049)


def test_video_generator_does_one_scene_file_per_run_and_retries_on_the_same_row():
    db = _db_with_scene_files(2)
    pipe = VideoGenerator(db, FakeProvider(fail_on={"p2"}, times=1))

    first = pipe.run()  # only the first scene file; scene p2 fails
    assert [(r["script_id"], r["status"], r["attempts"]) for r in first] == [(1, "done", 1), (1, "failed", 1)]
    assert first[0]["provider"] == "fake" and first[0]["file_path"].endswith("/scene_000.mp4")
    made = first[0]
    assert (made["resolution"], made["ratio"], made["tokens"], made["cost_usd"]) == ("480p", "9:16", 49005, 0.049)
    assert first[1].get("cost_usd") is None  # a failed attempt has no clip details

    second = pipe.run()  # still the first file: only the failed scene is retried
    assert [(r["script_id"], r["scene_index"], r["status"], r["attempts"]) for r in second] == [(1, 1, "done", 2)]
    assert len(db.tables["video_scenes"]) == 2  # the retry updated the scene's row, no new one

    third = pipe.run()  # now the second file
    assert [(r["script_id"], r["status"]) for r in third] == [(2, "done"), (2, "done")]
    assert VideoGenerator(db, FakeProvider(), {"max_scene_files_per_run": 5}).run() == []


def test_video_generator_dates_clip_folders_by_first_clip_not_by_script():
    db = _db_with_scene_files(1)
    db.tables["scripts"][0]["created_at"] = "2026-01-01T07:00:00Z"  # written long before its clips
    pipe = VideoGenerator(db, FakeProvider(fail_on={"p2"}, times=1))

    first = pipe.run()  # p1 done today, p2 fails
    assert first[0]["file_path"].startswith(day_folder() + "/script_1_")

    # pretend the first clip was made on an earlier day: the retry joins it there
    db.tables["video_scenes"][0]["file_path"] = "2025-12-31/script_1_x/scene_000.mp4"
    retry = pipe.run()
    assert retry[0]["file_path"] == "2025-12-31/script_1_x/scene_001.mp4"


def test_video_generator_gives_up_on_a_scene_after_max_attempts():
    db = _db_with_scene_files(2)
    pipe = VideoGenerator(db, FakeProvider(fail_on={"p2"}), {"max_attempts_per_scene": 2})

    pipe.run()  # file 1: p1 done, p2 attempt 1 fails
    with pytest.raises(VideoGenerationError):
        pipe.run()  # file 1: p2 attempt 2 fails, the only scene tried
    third = pipe.run()  # file 1 is given up on, so file 2 is next, not stuck behind it
    assert {r["script_id"] for r in third} == {2}
    rows = [r for r in db.tables["video_scenes"] if r["script_id"] == 1]
    assert sorted((r["scene_index"], r["status"], r["attempts"]) for r in rows) == [(0, "done", 1), (1, "failed", 2)]

    # raising the limit makes the given-up scene worth trying again
    retry = VideoGenerator(db, FakeProvider(), {"max_attempts_per_scene": 3}).run()
    assert [(r["script_id"], r["scene_index"], r["status"], r["attempts"]) for r in retry] == [(1, 1, "done", 3)]


def test_video_generator_raises_when_every_scene_fails():
    db = _db_with_scene_files(1, '[{"narration":"a","visual_prompt":"p","duration":3}]')
    with pytest.raises(VideoGenerationError):
        VideoGenerator(db, FakeProvider(fail_on={"p"})).run()
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
            job.update(content={"video_url": "https://cdn/clip.mp4"}, usage={"total_tokens": 49005},
                       model="seedance-1-0-pro-fast-251015", resolution="480p", ratio="9:16", duration=4)
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
    clip = provider.generate("a foggy town", 3.5)
    assert clip.video == b"mp4 bytes" and clip.job_id == "cgt-1"
    assert (clip.resolution, clip.ratio, clip.seconds) == ("480p", "9:16", 4)
    assert (clip.tokens, clip.cost_usd) == (49005, 0.049005)  # at $1 per million tokens
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


class FakeVoice:
    """Narrates, except texts in `fail_on` (each failing `times` times, forever if None)."""
    name = "fake_voice"

    def __init__(self, fail_on=(), times=None):
        self.fail_on, self.times, self.failures, self.texts = set(fail_on), times, {}, []

    def narrate(self, text):
        self.texts.append(text)
        if text in self.fail_on and (self.times is None or self.failures.get(text, 0) < self.times):
            self.failures[text] = self.failures.get(text, 0) + 1
            raise RuntimeError("tts busy")
        return Narration(b"mp3", model="m", voice="alloy", characters=len(text), seconds=30, cost_usd=0.0075)


def _db_with_narration_ready(count, clips_done=True):
    """`count` scripts (texts "t0", "t1", ...) whose single scene has a clip (or not)."""
    db = FakeDB()
    for number in range(count):
        script = db.insert("scripts", {"topic_id": 1, "file_path": f"s{number}.txt", "created_at": "2026-01-01T07:00:00Z"})
        db.upload("scripts", f"s{number}.txt", f"t{number}".encode())
        scene_file = db.insert("scene_files", {"script_id": script["id"], "file_path": "x.json", "scene_count": 1})
        if clips_done:
            db.insert("video_scenes", {"script_id": script["id"], "scene_file_id": scene_file["id"],
                                       "scene_index": 0, "status": "done"})
    return db


def test_audio_generator_keeps_going_after_a_failure_and_retries_on_the_same_row():
    db = _db_with_narration_ready(3)
    pipe = AudioGenerator(db, FakeVoice(fail_on={"t0"}, times=1), cfg={"max_scripts_per_run": 2})

    first = pipe.run()  # script 1 fails, script 2 still gets narrated; script 3 waits (limit 2)
    assert [(r["script_id"], r["status"], r["attempts"]) for r in first] == [(1, "failed", 1), (2, "done", 1)]
    done = first[1]
    assert done["narration_path"] == f"{day_folder()}/script_2_untitled/narration.mp3"  # dated today, not by the script
    assert (done["provider"], done["voice"], done["characters"], done["cost_usd"]) == ("fake_voice", "alloy", 2, 0.0075)

    second = pipe.run()  # the failed script is retried on its own row, then script 3
    assert [(r["script_id"], r["status"], r["attempts"]) for r in second] == [(1, "done", 2), (3, "done", 1)]
    assert len(db.tables["audio_files"]) == 3
    assert pipe.run() == []


def test_audio_generator_waits_for_clips_unless_told_not_to():
    db = _db_with_narration_ready(1, clips_done=False)
    assert AudioGenerator(db, FakeVoice()).run() == []
    assert [r["status"] for r in AudioGenerator(db, FakeVoice(), cfg={"wait_for_clips": False}).run()] == ["done"]


def test_audio_generator_gives_up_after_max_attempts_and_raises_when_all_fail():
    db = _db_with_narration_ready(1)
    pipe = AudioGenerator(db, FakeVoice(fail_on={"t0"}), cfg={"max_attempts_per_script": 2})
    for _ in range(2):
        with pytest.raises(AudioGenerationError):
            pipe.run()
    assert pipe.run() == []  # given up on
    assert (db.tables["audio_files"][0]["status"], db.tables["audio_files"][0]["attempts"]) == ("failed", 2)


def test_openai_tts_sends_settings_and_estimates_cost(monkeypatch):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, json=json)
        return FakeReply(content=b"mp3 bytes")

    monkeypatch.setattr("avgen.audio.openai_tts.requests.post", fake_post)
    monkeypatch.setattr("avgen.audio.openai_tts.mp3_seconds", lambda audio: 30.0)
    voice = create_voice({"provider": "openai_tts", "api_key": "k", "instructions": "calm", "usd_per_minute": 0.015})
    narration = voice.narrate("  Hello there.  ")
    assert sent["url"].endswith("/audio/speech")
    assert sent["json"] == {"model": "gpt-4o-mini-tts", "voice": "alloy", "input": "Hello there.",
                            "response_format": "mp3", "instructions": "calm"}
    assert (narration.audio, narration.characters, narration.seconds, narration.cost_usd) == (b"mp3 bytes", 12, 30.0, 0.0075)
    with pytest.raises(ValueError):
        voice.narrate("x" * 5000)


def test_plan_timing_fits_clips_to_the_narration():
    # planned 2s + 3s; clips came out 3s + 3s; narration is 6s -> shares 2.4s and 3.6s
    timing = plan_timing([2, 3], [3, 3], 6)
    assert [(t["seconds"], t["use_seconds"], t["slowdown"]) for t in timing] == [(2.4, 2.4, 1.0), (3.6, 3.0, 1.2)]
    assert sum(t["seconds"] for t in timing) == 6
    # without plans, time is shared by clip length
    assert [t["seconds"] for t in plan_timing(None, [2, 6], 4)] == [1.0, 3.0]


def test_build_ffmpeg_cmd_trims_stretches_and_mixes_music():
    timing = plan_timing([2, 3], [3, 3], 6)
    cmd = build_ffmpeg_cmd(["a.mp4", "b.mp4"], timing, "n.mp3", "m.mp3", "out.mp4", 0.2)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "[0:v]trim=duration=2.4,setpts=1.0*(PTS-STARTPTS)" in graph
    assert "[1:v]trim=duration=3.0,setpts=1.2*(PTS-STARTPTS)" in graph
    assert "[v0][v1]concat=n=2:v=1:a=0[v]" in graph and "[3:a]volume=0.2" in graph
    plain = build_ffmpeg_cmd(["a.mp4"], timing[:1], "n.mp3", None, "out.mp4")
    assert plain[plain.index("[v]") + 2] == "1:a"


def _db_ready_to_assemble(count):
    db = _db_with_narration_ready(count)
    for row in db.tables["scene_files"]:
        db.upload("scenes", row["file_path"], b'{"scenes": [{"index": 0, "duration": 3}]}')
    for row in db.tables["video_scenes"]:
        row.update(file_path=f"clip{row['script_id']}.mp4", cost_usd=0.05)
        db.upload("videos", row["file_path"], b"mp4")
    for script in db.tables["scripts"]:
        db.insert("audio_files", {"script_id": script["id"], "status": "done", "cost_usd": 0.0075,
                                  "narration_path": f"n{script['id']}.mp3"})
        db.upload("audio", f"n{script['id']}.mp3", b"mp3")
    return db


def _fake_ffmpeg(monkeypatch, fail_first=0):
    """Fake ffmpeg/ffprobe: every file is 4s long; the first `fail_first` videos fail to build."""
    monkeypatch.setattr(assembler, "media_seconds", lambda path: 4.0)
    calls = []

    def run(cmd, cwd=None):
        calls.append(cmd)
        if len(calls) <= fail_first:
            raise RuntimeError("ffmpeg failed")
        open(cmd[-1], "wb").write(b"final")

    monkeypatch.setattr(assembler, "run_ffmpeg", run)


def test_assembler_keeps_going_after_a_failure_and_records_cost(monkeypatch):
    db = _db_ready_to_assemble(2)
    _fake_ffmpeg(monkeypatch, fail_first=1)
    first = Assembler(db).run()
    assert [(r["script_id"], r["status"]) for r in first] == [(1, "failed"), (2, "done")]
    made = first[1]
    assert made["file_path"] == f"{day_folder()}/script_2_untitled.mp4"
    assert (made["clip_count"], made["seconds"], made["cost_usd"]) == (1, 4.0, 0.0575)
    assert db.files[("final", made["file_path"])] == b"final"

    second = Assembler(db).run()  # the failed script is retried on its own row
    assert [(r["script_id"], r["status"], r["attempts"]) for r in second] == [(1, "done", 2)]
    assert len(db.tables["final_videos"]) == 2 and Assembler(db).run() == []


def test_assembler_waits_for_narration_and_gives_up_after_max_attempts(monkeypatch):
    db = _db_ready_to_assemble(1)
    db.tables["audio_files"][0]["status"] = "failed"
    _fake_ffmpeg(monkeypatch)
    assert Assembler(db).run() == []  # no finished narration yet

    db.tables["audio_files"][0]["status"] = "done"
    monkeypatch.setattr(assembler, "run_ffmpeg", lambda cmd, cwd=None: (_ for _ in ()).throw(RuntimeError("bad")))
    pipe = Assembler(db, {"max_attempts_per_script": 2})
    for _ in range(2):
        with pytest.raises(AssemblyError):
            pipe.run()
    assert pipe.run() == []


WORDS = [{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.25}
         for i, w in enumerate(["The", "town", "that", "vanished.", "Nobody", "knows", "why", "it", "happened"])]


def test_group_words_makes_short_captions_and_breaks_at_sentences_and_pauses():
    captions = group_words(WORDS, min_words=2, max_words=4)
    assert [c["text"] for c in captions] == ["The town that vanished.", "Nobody knows why it", "happened"]
    assert captions[0]["start"] == 0.0 and captions[0]["end"] == captions[1]["start"]  # no flicker between
    assert [c["text"] for c in group_words(WORDS, 1, 2)] == ["The town", "that vanished.", "Nobody knows",
                                                            "why it", "happened"]
    paused = [dict(w) for w in WORDS[:4]]
    paused[2]["start"], paused[2]["end"], paused[3]["start"], paused[3]["end"] = 2.0, 2.2, 2.3, 2.5
    assert [c["text"] for c in group_words(paused, 2, 4)] == ["The town", "that vanished."]
    with pytest.raises(ValueError):
        group_words(WORDS, 3, 2)


def test_subtitle_style_is_configurable():
    assert ass_colour("#0000FF") == "&H00FF0000"      # blue, solid (.ass is blue-green-red)
    assert ass_colour("#FFFF0080") == "&H7F00FFFF"    # yellow, half see-through
    ass = to_ass(group_words(WORDS[:2]), {"color": "#0000FF", "outline_color": "#FFFF00", "font_size": 70,
                                          "uppercase": True, "position": "top"}, 1080, 1920)
    style = next(line for line in ass.splitlines() if line.startswith("Style: "))
    assert style.startswith("Style: Caption,DejaVu Sans,70,&H00FF0000,&H00FF0000,&H0000FFFF,")
    assert ",8," in style  # top
    assert "PlayResY: 1920" in ass and "Dialogue: 0,0:00:00.00,0:00:00.55,Caption,,0,0,0,,THE TOWN" in ass
    with pytest.raises(ValueError):
        to_ass([], {"position": "left"})


def test_ffmpeg_cmd_makes_a_fixed_9_16_output_and_burns_subtitles():
    timing = plan_timing([2], [3], 2)
    cmd = build_ffmpeg_cmd(["a.mp4"], timing, "n.mp3", None, "out.mp4", subtitles="captions.ass")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30" in graph
    assert "concat=n=1:v=1:a=0[joined];[joined]subtitles=captions.ass[v]" in graph
    assert cmd[cmd.index("-crf") + 1] == "20" and cmd[cmd.index("-ar") + 1] == "48000"
    padded = build_ffmpeg_cmd(["a.mp4"], timing, "n.mp3", None, "o.mp4", output={"width": 720, "height": 1280,
                                                                                  "fit": "pad"})
    assert "pad=720:1280:(ow-iw)/2:(oh-ih)/2:color=black" in padded[padded.index("-filter_complex") + 1]


def test_assembler_burns_subtitles_from_the_narration_word_timings(monkeypatch):
    db = _db_ready_to_assemble(1)
    db.tables["audio_files"][0].update(words=WORDS, transcription_cost_usd=0.003)
    _fake_ffmpeg(monkeypatch)
    calls = []
    real = assembler.run_ffmpeg

    def run(cmd, cwd=None):
        calls.append((cmd, (cwd / "captions.ass").read_text()))
        real(cmd, cwd)

    monkeypatch.setattr(assembler, "run_ffmpeg", run)
    made = Assembler(db, {"subtitles": {"max_words": 3}}).run()[0]
    cmd, ass = calls[0]
    assert "subtitles=captions.ass" in cmd[cmd.index("-filter_complex") + 1]
    assert "Caption,,0,0,0,,The town that" in ass
    assert made["details"]["subtitles"] == 3 and made["narration_cost_usd"] == 0.0105


class FakeTranscriber:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, seconds=None, script_text=None):
        self.calls += 1
        return Transcript(words=WORDS, model="whisper-1", cost_usd=0.003)


def test_audio_adds_word_timings_and_backfills_old_narrations():
    db = _db_with_narration_ready(2)
    AudioGenerator(db, FakeVoice(), cfg={"max_scripts_per_run": 1}).run()  # captions off: no word timings
    assert db.tables["audio_files"][0].get("words") is None

    voice, transcriber = FakeVoice(), FakeTranscriber()
    rows = AudioGenerator(db, voice, cfg={}, transcriber=transcriber).run()
    # script 1 only gets word timings (not narrated again); script 2 is narrated and timed
    assert voice.texts == ["t1"] and transcriber.calls == 2
    assert [(r["script_id"], r["status"], len(r["words"])) for r in rows] == [(1, "done", 9), (2, "done", 9)]
    assert rows[0]["transcription_model"] == "whisper-1" and rows[0]["cost_usd"] == 0.0075  # narration cost kept
    assert rows[0]["attempts"] == 1 and rows[0]["caption_attempts"] == 1  # captions don't use up narration attempts
    srt = db.files[("audio", rows[0]["captions_path"])]
    assert rows[0]["captions_path"] == rows[0]["narration_path"].replace("narration.mp3", "captions.srt")
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert AudioGenerator(db, voice, cfg={}, transcriber=transcriber).run() == []


def test_openai_transcriber_asks_for_word_timings(monkeypatch):
    sent = {}

    def fake_post(url, data=None, files=None, headers=None, timeout=None):
        sent.update(url=url, data=data)
        return FakeReply({"text": "Hi there", "duration": 1.0,
                          "words": [{"word": " Hi", "start": 0.0, "end": 0.4}, {"word": "there", "start": 0.4, "end": 0.9}]})

    monkeypatch.setattr("avgen.audio.transcribe.requests.post", fake_post)
    assert create_transcriber({"enabled": False}) is None
    transcriber = create_transcriber({"enabled": True, "api_key": "k", "model": "whisper-1", "usd_per_minute": 0.006})
    transcript = transcriber.transcribe(b"mp3", seconds=30, script_text="Hi there")
    assert sent["url"].endswith("/audio/transcriptions")
    assert sent["data"]["timestamp_granularities[]"] == "word" and sent["data"]["response_format"] == "verbose_json"
    assert transcript.words[0] == {"word": "Hi", "start": 0.0, "end": 0.4} and transcript.cost_usd == 0.003


def test_srt_numbers_and_times_each_caption():
    srt = to_srt([{"text": "The town", "start": 0.0, "end": 0.55}, {"text": "vanished", "start": 61.2, "end": 62.0}])
    assert srt == "1\n00:00:00,000 --> 00:00:00,550\nThe town\n\n2\n00:01:01,200 --> 00:01:02,000\nvanished\n"


def test_audio_adds_missing_srt_without_transcribing_again():
    db = _db_with_narration_ready(1)
    AudioGenerator(db, FakeVoice(), cfg={}).run()
    db.tables["audio_files"][0]["words"] = WORDS  # word timings made by an earlier version, but no captions.srt
    transcriber = FakeTranscriber()
    rows = AudioGenerator(db, FakeVoice(), cfg={}, transcriber=transcriber).run()
    assert transcriber.calls == 0 and rows[0]["captions_path"].endswith("/captions.srt")


class BrokenTranscriber(FakeTranscriber):
    def transcribe(self, audio, seconds=None, script_text=None):
        self.calls += 1
        raise RuntimeError("transcription down")


def test_audio_keeps_the_narration_when_captions_fail_and_retries_only_the_captions():
    db = _db_with_narration_ready(1)
    voice = FakeVoice()
    gen = AudioGenerator(db, voice, cfg={"max_attempts_per_script": 2}, transcriber=BrokenTranscriber())
    with pytest.raises(AudioGenerationError):
        gen.run()  # narration made, captions failed
    row = db.tables["audio_files"][0]
    assert (row["status"], row["attempts"], row["caption_attempts"]) == ("done", 1, 1)
    with pytest.raises(AudioGenerationError):
        gen.run()  # only the captions are tried again
    assert voice.texts == ["t0"] and db.tables["audio_files"][0]["caption_attempts"] == 2
    assert gen.run() == []  # captions given up on; the narration is still there for the video
