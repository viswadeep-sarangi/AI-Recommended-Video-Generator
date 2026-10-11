import json

import requests


class OpenAITTS:
    def __init__(self, api_key, model="gpt-4o-mini-tts", voice="alloy", base_url="https://api.openai.com/v1", **_):
        self.key, self.model, self.voice, self.base = api_key, model, voice, base_url.rstrip("/")

    def narrate(self, text):
        r = requests.post(f"{self.base}/audio/speech", headers={"Authorization": "Bearer " + self.key}, timeout=300,
                          json={"model": self.model, "voice": self.voice, "input": text, "response_format": "mp3"})
        r.raise_for_status()
        return r.content


class StaticMusic:
    """Background music from a configured URL (generator APIs can be added the same way)."""

    def __init__(self, url, **_):
        self.url = url

    def music(self, prompt, seconds):
        r = requests.get(self.url, timeout=300)
        r.raise_for_status()
        return r.content


VOICE = {"openai_tts": OpenAITTS}
MUSIC = {"static": StaticMusic}


def register_voice(name, cls):
    VOICE[name] = cls


def register_music(name, cls):
    MUSIC[name] = cls


class AudioGenerator:
    def __init__(self, db, cfg, voice=None, music=None):
        self.db, self.cfg = db, cfg
        v = dict(cfg["voice"])
        self.voice = voice or VOICE[v.pop("provider")](**v)
        m = dict(cfg.get("music") or {"provider": "none"})
        self.music = music or (MUSIC[m.pop("provider")](**m) if m.get("provider") != "none" else None)

    def generate(self, script):
        text = self.db.download("scripts", script["file_path"]).decode("utf-8")
        folder = f"{self.cfg.get('folder', 'audio')}/script_{script['id']}"
        row = {"script_id": script["id"], "narration_path": f"{folder}/narration.mp3", "music_path": None}
        self.db.upload("audio", row["narration_path"], self.voice.narrate(text), "audio/mpeg")
        if self.music:
            row["music_path"] = f"{folder}/music.mp3"
            self.db.upload("audio", row["music_path"],
                           self.music.music(self.cfg.get("music_prompt", text[:200]), script.get("target_seconds", 30)), "audio/mpeg")
        return self.db.insert("audio_files", row)

    def run_pending(self):
        done = {r["script_id"] for r in self.db.select("audio_files")}
        return [self.generate(s) for s in self.db.select("scripts") if s["id"] not in done]
