create table if not exists topics (
  id bigint generated always as identity primary key,
  title text not null, description text, category text, score numeric,
  status text not null default 'new', created_at timestamptz default now());
create table if not exists scripts (
  id bigint generated always as identity primary key,
  topic_id bigint references topics(id), file_path text not null,
  target_seconds int, created_at timestamptz default now());
create table if not exists scene_files (
  id bigint generated always as identity primary key,
  script_id bigint references scripts(id), file_path text not null,
  scene_count int, created_at timestamptz default now());
create table if not exists video_scenes (
  id bigint generated always as identity primary key,
  script_id bigint, scene_file_id bigint references scene_files(id), scene_index int not null,
  status text not null, file_path text, provider text, error text, created_at timestamptz default now());
create table if not exists audio_files (
  id bigint generated always as identity primary key,
  script_id bigint references scripts(id), narration_path text, music_path text,
  created_at timestamptz default now());
create table if not exists final_videos (
  id bigint generated always as identity primary key,
  script_id bigint references scripts(id), file_path text,
  created_at timestamptz default now());
-- Create storage buckets: scripts, scenes, video-scenes, audio, final-videos
