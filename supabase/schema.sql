create schema if not exists avgen;

create table if not exists avgen.youtube_videos (
  id bigint generated always as identity primary key,
  video_id text not null unique,
  video_url text not null,
  title text not null,
  description text,
  channel_id text, channel_title text, published_at timestamptz,
  thumbnails jsonb, tags jsonb, category_id text, live_broadcast_content text,
  localized jsonb, duration text, dimension text, definition text, caption text,
  licensed_content boolean, projection text,
  view_count bigint, like_count bigint, dislike_count bigint, comment_count bigint,
  upload_status text, privacy_status text, license text,
  embeddable boolean, public_stats_viewable boolean,
  made_for_kids boolean, self_declared_made_for_kids boolean,
  live_streaming_details jsonb, topic_ids jsonb, recording_details jsonb,
  player jsonb, queries jsonb, raw_metadata jsonb not null,
  fetched_at timestamptz default now(), created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists avgen.topics (
  id bigint generated always as identity primary key,
  title text not null, description text, category text, score numeric,
  status text not null default 'new', created_at timestamptz default now());
create table if not exists avgen.scripts (
  id bigint generated always as identity primary key,
  topic_id bigint references avgen.topics(id), file_path text not null,
  target_seconds int, created_at timestamptz default now());
create table if not exists avgen.scene_files (
  id bigint generated always as identity primary key,
  script_id bigint references avgen.scripts(id), file_path text not null,
  scene_count int, created_at timestamptz default now());
create table if not exists avgen.video_scenes (
  id bigint generated always as identity primary key,
  script_id bigint, scene_file_id bigint references avgen.scene_files(id), scene_index int not null,
  status text not null, file_path text, provider text, error text, created_at timestamptz default now());
create table if not exists avgen.audio_files (
  id bigint generated always as identity primary key,
  script_id bigint references avgen.scripts(id), narration_path text, music_path text,
  created_at timestamptz default now());
create table if not exists avgen.final_videos (
  id bigint generated always as identity primary key,
  script_id bigint references avgen.scripts(id), file_path text,
  created_at timestamptz default now());

grant usage on schema avgen to service_role;
grant all on all tables in schema avgen to service_role;
grant all on all sequences in schema avgen to service_role;

alter default privileges in schema avgen
  grant all on tables to service_role;
alter default privileges in schema avgen
  grant all on sequences to service_role;

-- Storage buckets for the files each stage produces (names match "buckets" in the config).
-- Private: only requests with a privileged key (the pipeline's service key) can read or write.
-- Safe to re-run: existing buckets are left untouched.
insert into storage.buckets (id, name, public) values
  ('scripts', 'scripts', false),
  ('scenes', 'scenes', false),
  ('video-scenes', 'video-scenes', false),
  ('audio', 'audio', false),
  ('final-videos', 'final-videos', false)
on conflict (id) do nothing;
