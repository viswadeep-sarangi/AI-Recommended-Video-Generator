# AI Recommended Video Generator

Pipeline (all state in Supabase tables + storage buckets, see `supabase/schema.sql`):

1. `scout` – LLM + YouTube trends -> `topics` table (optional Telegram notification). Runs daily via `.github/workflows/daily-scout.yml`.
2. `scripts` – topics -> script file in storage + `scripts` log row.
3. `scenes` – scripts -> 2–5 s scene JSON + `scene_files` log row.
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
