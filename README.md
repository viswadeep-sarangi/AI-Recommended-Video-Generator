# AI Recommended Video Generator

Pipeline (all state in the Supabase `avgen` schema plus storage buckets, see `supabase/schema.sql`):

1. `scout` – YouTube search and video details are stored in `youtube_videos`; the LLM uses those signals to create rows in `topics` (optional Telegram notification). Runs daily via `.github/workflows/daily-scout.yml`.
2. `scripts` – topics -> script file in storage + `scripts` log row.
3. `scenes` – scripts -> 2–5 s scene JSON + `scene_files` log row. The LLM's reply is checked (and retried once if unusable); a failing script is skipped and retried next run. Up to `max_per_run` (20) scripts per run. Runs daily via `.github/workflows/daily-scenes.yml`.
4. `videos` – scenes without clips -> Seedance via BytePlus ModelArk (`ARK_API_KEY`; 480p 9:16 by default, ~$0.05 per 5s clip) -> storage + one `video_scenes` row per scene. Works on up to `max_scene_files_per_run` (5) scene files per run, i.e. all the missing clips of up to 5 scripts. A failing scene is retried on later runs, up to `max_attempts_per_scene` (3) attempts, then given up on (logged as a warning every run). Several models can be listed under `models`, tried in order: when one is paused by BytePlus (free quota used up, `SetLimitExceeded`) the run switches to the next without using up an attempt, and stops cleanly once all are paused. `max_usd_per_month` ($30) stops making clips once this calendar month's paid clips (from `video_scenes`, so across runs) are estimated to cost that much, and starts again on the 1st; models marked `free_quota_only` (Safe Experience Mode on in the ModelArk console, so they never bill) don't count toward it. Newer models such as Seedance 1.5 pro, if added, need `"settings_in": "body"` and `"generate_audio": false` (sound doubles its price). Runs daily via `.github/workflows/daily-videos.yml`.
5. `audio` – scripts whose clips are all made -> narration via OpenAI text-to-speech (`gpt-4o-mini-tts`, ~$0.015 per minute; OpenAI retires it on 6 Jan 2027) and optional music -> storage + one `audio_files` row per script, with its estimated cost. With `captions` on, the narration is also sent to OpenAI transcription (`whisper-1`, ~$0.006 per minute) for word timings, used for subtitles, and `captions.srt` is saved next to the narration (for YouTube closed captions). Narrations missing either get them on the next run, without being narrated again. Up to `max_scripts_per_run` (10) scripts per run; a failing script is retried on later runs, up to `max_attempts_per_script` (3) attempts. Runs daily via `.github/workflows/daily-audio.yml`.
6. `assemble` – scripts with every clip and the narration made -> ffmpeg fits each clip to its share of the narration (cut short or slowed down) and joins them into a 1080x1920 (9:16) 30 fps mp4 (`output`; clips are cropped or padded to fit) with burned-in subtitles of 2–3 words each, on the lower-third line (600 px up from the bottom, clear of the Shorts title and buttons) (`subtitles`: grouping and the whole look, e.g. colours, font, outline, position, are set in the config) -> `final` storage, dated the day it was made, + one `final_videos` row per script with its length, size and the estimated cost of the whole video (clips + narration). Up to `max_scripts_per_run` (5) per run; a failing script is retried on later runs, up to `max_attempts_per_script` (3) attempts. Needs ffmpeg. Runs daily via `.github/workflows/daily-assemble.yml`. (YouTube upload is a future extension.)

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
