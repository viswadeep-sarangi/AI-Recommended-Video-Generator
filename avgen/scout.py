"""Recommendation finder ("scout"): finds trending topics and stores them in Supabase."""
from datetime import datetime, timedelta, timezone

import requests

from .llm import extract_json, wrap_prompt

YOUTUBE_SEARCH = "https://www.googleapis.com/youtube/v3/search"


def fetch_youtube(src, days):
    after = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "part": "snippet", "type": "video", "order": src.get("order", "viewCount"),
        "publishedAfter": after, "maxResults": src.get("max_results", 15), "key": src["api_key"],
    }
    for k in ("regionCode", "videoCategoryId", "relevanceLanguage"):
        if src.get(k):
            params[k] = src[k]
    out = []
    for q in src.get("queries") or [""]:
        r = requests.get(YOUTUBE_SEARCH, params={**params, "q": q}, timeout=60)
        r.raise_for_status()
        out += [i["snippet"]["title"] for i in r.json().get("items", [])]
    return out


SOURCES = {"youtube": fetch_youtube}


class RecommendationFinder:
    def __init__(self, llm, db, cfg, notifier=None):
        self.llm, self.db, self.cfg, self.notifier = llm, db, cfg, notifier

    def gather_signals(self):
        days = self.cfg.get("lookback_days", 60)
        signals = []
        for src in self.cfg.get("sources", []):
            if not src.get("enabled", True):
                continue
            try:
                signals += [f"[{src['type']}] {t}" for t in SOURCES[src["type"]](src, days)]
            except Exception as e:  # one failing source should not stop the run
                print(f"source {src.get('type')} failed: {e}")
        return signals

    def build_prompt(self, signals):
        seeds = ", ".join(self.cfg.get("seed_topics", [])) or "any"
        n = self.cfg.get("top_n", 10)
        body = (
            f"Recommend the top {n} trending topics for short-form videos from the last "
            f"{self.cfg.get('lookback_days', 60)} days. Focus areas: {seeds}.\n"
            "Trending signals:\n" + ("\n".join(signals) or "(none available; use your own knowledge)") +
            '\n\nReply ONLY with JSON: [{"title": str, "description": str, "score": number 0-100}]'
        )
        return wrap_prompt(body, self.cfg.get("prepend_instructions", ""), self.cfg.get("append_instructions", ""))

    def run(self):
        topics = extract_json(self.llm.complete(self.build_prompt(self.gather_signals())))
        rows = []
        for t in topics:
            rows.append(self.db.insert("topics", {
                "title": t["title"], "description": t.get("description", ""),
                "score": t.get("score"), "status": "new",
            }))
        if self.notifier and rows:
            self.notifier.send("Top trending topics:\n" + "\n".join(f"- {r['title']}" for r in rows[: self.cfg.get("notify_top_x", 5)]))
        return rows


class TelegramNotifier:
    def __init__(self, bot_token, chat_id, **_):
        self.token, self.chat_id = bot_token, chat_id

    def send(self, text):
        requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                      json={"chat_id": self.chat_id, "text": text}, timeout=30).raise_for_status()


def create_notifier(cfg):
    cfg = cfg or {}
    if cfg.get("enabled") and cfg.get("type") == "telegram":
        return TelegramNotifier(**cfg)
    return None
