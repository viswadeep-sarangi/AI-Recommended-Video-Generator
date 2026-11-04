"""Recommendation finder ("scout"): finds trends and stores their source metadata."""
import logging
from datetime import datetime, timedelta, timezone

import requests

from .llm import extract_json, wrap_prompt
from .log import check_response

log = logging.getLogger(__name__)

YOUTUBE_SEARCH = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_PARTS = (
    "snippet,contentDetails,statistics,status,liveStreamingDetails,"
    "topicDetails,recordingDetails,player,localizations"
)
# Optional search.list filters passed through from the source config when set.
YOUTUBE_FILTERS = ("regionCode", "videoCategoryId", "relevanceLanguage", "videoDuration")


def fetch_youtube(src, days):
    after = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "part": "snippet", "type": "video", "order": src.get("order", "viewCount"),
        "publishedAfter": after, "maxResults": src.get("max_results", 15), "key": src["api_key"],
    }
    for k in YOUTUBE_FILTERS:
        if src.get(k):
            params[k] = src[k]
    if not src.get("api_key"):
        log.error("YouTube api_key is empty; is YOUTUBE_API_KEY set?")
    log.info(
        "YouTube search: queries=%s published_after=%s order=%s max_results=%s filters=%s",
        src.get("queries") or [""], after, params["order"], params["maxResults"],
        {k: params[k] for k in YOUTUBE_FILTERS if k in params},
    )
    search_results = {}
    for q in src.get("queries") or [""]:
        r = requests.get(YOUTUBE_SEARCH, params={**params, "q": q}, timeout=60)
        check_response(r, f"YouTube search for query {q!r}")
        items = r.json().get("items", [])
        log.info("YouTube search %r: %d result(s)", q, len(items))
        for item in items:
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
        check_response(r, f"YouTube videos.list for {len(batch)} id(s)")
        for item in r.json().get("items", []):
            video = _normalize_youtube_video(item)
            video["queries"] = search_results[video["video_id"]]["queries"]
            log.debug("YouTube video %s: %r views=%s", video["video_id"], video["title"], video["view_count"])
            videos.append(video)
    log.info("YouTube: fetched details for %d of %d unique video(s)", len(videos), len(ids))
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
                log.info("source %s is disabled; skipping", src.get("type"))
                continue
            log.info("source %s: fetching (lookback %d days)", src.get("type"), days)
            try:
                records = SOURCES[src["type"]](src, days)
                if src["type"] == "youtube":
                    self._store_youtube_videos(records)
                    for record in records:
                        signals.append(
                            f"[youtube] {record['title']} | channel: {record.get('channel_title') or 'unknown'} "
                            f"| published: {record.get('published_at') or 'unknown'} "
                            f"| views: {record.get('view_count') if record.get('view_count') is not None else 'unknown'} "
                            f"| likes: {record.get('like_count') if record.get('like_count') is not None else 'unknown'} "
                            f"| duration: {record.get('duration') or 'unknown'}"
                        )
                else:
                    signals += [f"[{src['type']}] {t}" for t in records]
            except Exception:  # one failing source should not stop the run
                log.exception("source %s failed; its trends will not be used", src.get("type"))
        log.info("gathered %d trending signal(s)", len(signals))
        if not signals:
            log.warning("no trending signals were gathered; the LLM will pick topics from its own knowledge")
        return signals

    def _store_youtube_videos(self, records):
        """Store each video on its own so one bad row doesn't lose the rest.

        Storage failures are logged but don't drop the videos from the trend signals.
        """
        failed = 0
        for record in records:
            try:
                self._store_youtube_video(record)
            except Exception as e:
                failed += 1
                if failed == 1:
                    log.error("could not store YouTube video %s: %s", record.get("video_id"), e)
                else:
                    log.debug("could not store YouTube video %s: %s", record.get("video_id"), e)
        if failed:
            log.error("stored %d of %d YouTube video(s) in youtube_videos; %d failed (first error above)",
                      len(records) - failed, len(records), failed)
        else:
            log.info("stored %d YouTube video(s) in youtube_videos", len(records))

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
        prompt = self.build_prompt(self.gather_signals())
        log.info("asking LLM for topics (prompt: %d chars)", len(prompt))
        log.debug("LLM prompt:\n%s", prompt)
        reply = self.llm.complete(prompt)
        log.debug("LLM reply:\n%s", reply)
        try:
            topics = extract_json(reply)
        except ValueError:
            log.error("LLM reply is not valid JSON:\n%s", reply)
            raise
        log.info("LLM returned %d topic(s)", len(topics))
        rows = []
        for t in topics:
            rows.append(self.db.insert("topics", {
                "title": t["title"], "description": t.get("description", ""),
                "score": t.get("score"), "status": "new",
            }))
            log.info("topic #%s (score %s): %s", rows[-1].get("id"), t.get("score"), t["title"])
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
