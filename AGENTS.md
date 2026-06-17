# AI YouTube Shorts Generator

- Main entrypoint: `main.py`; public Python API: `shorts_generator.generate_shorts`.
- API mode is default. It uses MuAPI for download, Whisper, highlight ranking, and autocrop.
- Local mode is opt-in via `--mode local`; it needs `ffmpeg` on PATH plus `pip install -r requirements-local.txt`.
- Local mode accepts YouTube URLs, `file://` URLs, or direct local file paths.
- Local downloads cache as `output/source_<youtube_id>.*`; local transcripts cache as `output/<video_stem>.srt`.
- The repo has no test suite or CI config. Validate changes with a focused CLI smoke run instead of inventing a broad test harness.
- `shorts_generator/highlights.py` is the core prompt/selection logic; keep its JSON contract stable (`title`, `start_time`, `end_time`, `score`, `hook_sentence`, `virality_reason`).
- `shorts_generator/local/clipper.py` is the heaviest file: podcast mode uses YOLOv8 + ByteTrack + MediaPipe/OpenCV fallbacks, so keep crop changes conservative.
- `opencode.json` points OpenCode at the configured OpenAI-compatible gateway; do not change it unless the task is about OpenCode itself.
- If you need a quick manual check, run one short clip generation with the smallest realistic input and inspect the printed result plus any written files.
