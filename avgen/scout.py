"""Recommendation finder ("scout"): finds trends and stores their source metadata."""
from datetime import datetime, timedelta, timezone

import requests

from .llm import extract_json, wrap_prompt

YOUTUBE_SEARCH = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_PARTS = (
    "snippet,contentDetails,statistics,status,liveStreamingDetails,"
    "topicDetails,recordingDetails,player,localizations"
)


def fetch_youtube(src, days):
    after = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "part": "snippet", "type": "video", "order": src.get("order", "viewCount"),
        "publishedAfter": after, "maxResults": src.get("max_results", 15), "key": src["api_key"],
    }
    for k in ("regionCode", "videoCategoryId", "relevanceLanguage"):
        if src.get(k):
            params[k] = src[k]
    search_results = {}
    for q in src.get("queries") or [""]:
        r = requests.get(YOUTUBE_SEARCH, params={**params, "q": q}, timeout=60)
        r.raise_for_status()
        for item in r.json().get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if video_id:
                search_results.setdefault(video_id, {"id": video_id, "queries": []})
                if q not in search_results[video_id]["queries"]:
                    search_results[video_id]["queries"].append(q)

    videos = []
    ids = list(search_results)
    for start in range(0, len(ids), 50):
        batch = ids[start:start + 50]
        r = requests.get(
            YOUTUBE_VIDEOS,
            params={"part": YOUTUBE_PARTS, "id": ",".join(batch), "key": src["api_key"]},
            timeout=60,
        )
        r.raise_for_status()
        for item in r.json().get("items", []):
            video = _normalize_youtube_video(item)
            video["queries"] = search_results[video["video_id"]]["queries"]
            videos.append(video)
    return videos


def _normalize_youtube_video(item):
    """Keep useful columns easy to query while retaining the complete API response."""
    snippet = item.get("snippet") or {}
    content = item.get("contentDetails") or {}
    stats = item.get("statistics") or {}
    status = item.get("status") or {}
    return {
        "video_id": item["id"],
        "video_url": f"https://www.youtube.com/watch?v={item['id']}",
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "channel_id": snippet.get("channelId"),
        "channel_title": snippet.get("channelTitle"),
        "published_at": snippet.get("publishedAt"),
        "thumbnails": snippet.get("thumbnails"),
        "tags": snippet.get("tags"),
        "category_id": snippet.get("categoryId"),
        "live_broadcast_content": snippet.get("liveBroadcastContent"),
        "localized": snippet.get("localized"),
        "duration": content.get("duration"),
        "dimension": content.get("dimension"),
        "definition": content.get("definition"),
        "caption": content.get("caption"),
        "licensed_content": content.get("licensedContent"),
        "projection": content.get("projection"),
        "view_count": _integer(stats.get("viewCount")),
        "like_count": _integer(stats.get("likeCount")),
        # YouTube removed public dislike counts; this remains null unless an API
        # response ever provides one again.
        "dislike_count": _integer(stats.get("dislikeCount")),
        "comment_count": _integer(stats.get("commentCount")),
        "upload_status": status.get("uploadStatus"),
        "privacy_status": status.get("privacyStatus"),
        "license": status.get("license"),
        "embeddable": status.get("embeddable"),
        "public_stats_viewable": status.get("publicStatsViewable"),
        "made_for_kids": status.get("madeForKids"),
        "self_declared_made_for_kids": status.get("selfDeclaredMadeForKids"),
        "live_streaming_details": item.get("liveStreamingDetails"),
        "topic_ids": (item.get("topicDetails") or {}).get("topicIds"),
        "recording_details": item.get("recordingDetails"),
        "player": item.get("player"),
        "raw_metadata": item,
    }


def _integer(value):
    return int(value) if value is not None else None


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
                records = SOURCES[src["type"]](src, days)
                if src["type"] == "youtube":
                    for record in records:
                        self._store_youtube_video(record)
                        signals.append(
                            f"[youtube] {record['title']} | channel: {record.get('channel_title') or 'unknown'} "
                            f"| published: {record.get('published_at') or 'unknown'} "
                            f"| views: {record.get('view_count') if record.get('view_count') is not None else 'unknown'} "
                            f"| likes: {record.get('like_count') if record.get('like_count') is not None else 'unknown'} "
                            f"| duration: {record.get('duration') or 'unknown'}"
                        )
                else:
                    signals += [f"[{src['type']}] {t}" for t in records]
            except Exception as e:  # one failing source should not stop the run
                print(f"source {src.get('type')} failed: {e}")
        return signals

    def _store_youtube_video(self, record):
        row = dict(record)
        row["fetched_at"] = datetime.now(timezone.utc).isoformat()
        if hasattr(self.db, "upsert"):
            self.db.upsert("youtube_videos", row, "video_id")
        else:
            self.db.insert("youtube_videos", row)

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
