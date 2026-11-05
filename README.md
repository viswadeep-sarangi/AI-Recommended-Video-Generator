# AI Recommended Video Generator

Pipeline (all state in the Supabase `avgen` schema plus storage buckets, see `supabase/schema.sql`):

1. `scout` – YouTube search and video details are stored in `youtube_videos`; the LLM uses those signals to create rows in `topics` (optional Telegram notification). Runs daily via `.github/workflows/daily-scout.yml`.
2. `scripts` – topics -> script file in storage + `scripts` log row.
3. `scenes` – scripts -> 2–5 s scene JSON + `scene_files` log row. The LLM's reply is checked (and retried once if unusable); a failing script is skipped and retried next run. Up to `max_per_run` (20) scripts per run. Runs daily via `.github/workflows/daily-scenes.yml`.
4. `videos` – scenes without clips -> video provider (Sora, generic HTTP such as Kling, with fallback / per-category routing) -> storage + `video_scenes` log.
5. `audio` – narration (TTS) and optional music -> storage + `audio_files` log.
6. `assemble` – ffmpeg merges clips + audio -> `final_videos`. (YouTube upload is a future extension.)

## Usage
```
pip install -r requirements.txt
cp config/config.example.json config/config.json   # values "env:NAME" are read from env vars
python -m avgen scout      # or scripts|scenes|videos|audio|assemble|all
```
LLM / video / voice backends are swappable in config (`provider`/`type`); add new ones with
`register_llm`, `register_video_provider`, `register_voice`, `register_music`.

The Supabase API must expose the `avgen` schema (Supabase Dashboard >
Settings > API > Exposed schemas) after applying `supabase/schema.sql`.

### YouTube metadata

The scout stores one up-to-date row per video in `youtube_videos`, keyed by
`video_id`. It includes the title, full description, uploader channel ID and
name, original publication timestamp, thumbnails, tags, category, duration,
captions and quality details, visibility/licensing/embed status, views, likes,
comments, live-stream details, topic IDs, recording details, player data,
search queries, fetch time, and the complete unmodified YouTube API response in
`raw_metadata`.

YouTube no longer provides public dislike counts through the Data API, so
`dislike_count` is normally `NULL`. Historical or private dislike data cannot be
reconstructed by this application.
