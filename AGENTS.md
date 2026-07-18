# AI YouTube Shorts Generator

- Entrypoint: `main.py`; public API: `shorts_generator.generate_shorts`.
- Modes: Default (API/MuAPI) vs `--mode local` (requires `ffmpeg` on PATH + `pip install -r requirements-local.txt`).
- Inputs: YouTube URLs, `file://` URLs, or local file paths.
- Caching: Local downloads as `output/source_<youtube_id>.*`; transcripts as `output/<video_stem>.srt`.
- Validation: No test suite, linter, typechecker, or CI. Validate changes with CLI smoke run:
  ```bash
  python main.py <URL_OR_FILE_PATH> --mode local
  ```

## Core Components

- `shorts_generator/highlights.py`: Core prompt/selection logic. JSON contract (`title`, `start_time`, `end_time`, `score`, `hook_sentence`, `virality_reason`) must remain stable. LLM usage pluggable (`call_muapi_llm` for API, `call_local_llm` for local).
- `shorts_generator/local/clipper.py`: 1565 lines. Podcast mode uses YOLOv8 + ByteTrack + MediaPipe/OpenCV fallbacks. Keep crop changes minimal.

## Architecture Gotchas

- `highlights.py` mixes Indonesian and English in Tavily context prompts — intentional.
- `config.py` loads `.env` at import via `python-dotenv`. Default `POLL_TIMEOUT_SECONDS` is 600 (trust code, not README).
- Videos >30 min auto-chunked into 20-min windows with 60s overlap. Highlights offset-adjusted and globally deduped.
- Local clipper fallback chain: MediaPipe Pose+FaceMesh → OpenCV Haar cascade → center crop. Podcast mode falls back to legacy path on YOLO failure.
- `Pipeline._sanitize_highlight_timestamps` silently drops clips where `start >= end` after clamping. Check logs if clips disappear.
- `generate_shorts()` return shape: `{mode, source_video_url, transcript, highlights, shorts}`. `clip_url` is hosted URL (API mode) or local path (local mode).
