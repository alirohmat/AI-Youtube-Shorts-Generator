"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py:
  - content-type / density detection
  - chunking for long videos with overlap
  - virality-criteria prompt
  - score-based dedupe with overlap suppression

The LLM call is pluggable via the `llm_fn` argument so the same prompts can
drive either MuAPI (default, --mode api) or a direct local LLM client
(--mode local).
"""
import json
import os
import re
from typing import Callable, Dict, List, Optional

from . import muapi
from .config import TAVILY_API_KEY, USE_TAVILY_CONTEXT


LLMFn = Callable[[str], str]


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""


VIRALITY_CRITERIA = """
Virality signals to prioritize (ranked by impact):
1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong about...")
2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement; raw unscripted reactions
3. OPINION BOMBS — strong, polarizing or counter-intuitive statements that trigger agree/disagree
4. REVELATION MOMENTS — surprising facts, stats, or confessions that reframe how the viewer thinks
5. CONFLICT/TENSION — disagreement, pushback, or a problem being confronted head-on
6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
7. STORY PEAKS — the climax or twist of an anecdote; the payoff moment
8. PRACTICAL VALUE — a concrete tip, hack, or insight the viewer can immediately apply
"""


HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}

Content type: {content_type} | Density: {density}

{external_context_block}

Your task: identify the most viral-worthy highlights from the transcript.

Rules:
- Every highlight must open with a strong HOOK — a line that grabs attention within the first 3 seconds
- Duration sweet spot: 45-90 seconds. Go shorter (20-44s) only for a perfect standalone one-liner. Go longer (91-180s) only when a story arc needs full context to land
- Never cut mid-sentence or mid-thought — each clip must feel complete and self-contained
- Clips must not overlap significantly with each other
- Score 0-100 on viral potential (not general quality)
- {num_clips_instruction}
- For each highlight, identify the single best "hook_sentence" — the opening line that would make someone stop scrolling
- Explain in one sentence why this clip is viral ("virality_reason")

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","virality_reason":"string"}}]}}"""


CHUNK_SIZE_SECONDS = 1200       # 20-min chunks for long videos
LONG_VIDEO_THRESHOLD = 1800     # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
GPT_CALL_TIMEOUT_SECONDS = 300  # cap LLM polls at 5 min — a wedged call should fail fast


def _extract_chunk_topic(transcript_text: str) -> str:
    text = re.sub(r"\[[0-9]+(?:\.[0-9]+)?s\]\s*", " ", transcript_text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        candidate = sentence.strip()
        if not candidate:
            continue
        candidate = re.sub(r"^\[[^\]]*\]\s*", "", candidate).strip()
        if not candidate:
            continue
        if re.fullmatch(r"[\W_\d]+", candidate):
            continue
        return candidate[:1000]

    return text[:1000]


def extract_topic_with_llm(chunk_text: str, llm_fn: Callable[[str], str]) -> str:
    prompt = (
        "Rangkum topik utama dari transcript berikut dalam 1 kalimat singkat "
        "(fokus pada subjek dan isu utama, maksimal 10 kata, bahasa Indonesia):\n\n"
        f"{chunk_text}"
    )
    raw = llm_fn(prompt)
    cleaned = clean_json_response(raw).strip()
    cleaned = cleaned.strip('"').strip("'")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    return cleaned


def _default_external_context_block() -> str:
    return (
        "[EXTERNAL CONTEXT: unavailable] "
        "Gunakan konteks eksternal ini untuk menilai sensitivitas, relevansi, dan potensi viral dari topik tersebut."
    )


def _build_external_context_block(chunk_topic: str) -> str:
    if not USE_TAVILY_CONTEXT or not TAVILY_API_KEY:
        return _default_external_context_block()

    try:
        from tavily import TavilyClient

        tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        response = tavily_client.search(query=chunk_topic, search_depth="basic", max_results=3)
        context = " ".join([r.get("content", "") for r in response.get("results", [])][:3]).strip()
        if not context:
            context = "unavailable"
        return (
            f"[EXTERNAL CONTEXT: {context}] "
            "Gunakan konteks eksternal ini untuk menilai sensitivitas, relevansi, dan potensi viral dari topik tersebut."
        )
    except Exception:
        print("[WARN] Tavily context failed, proceeding without external context", flush=True)
        return _default_external_context_block()


def _build_theme_locked_external_context_block(video_topic: str, chunk_topic: str) -> str:
    """Keep Tavily context anchored to a single video theme.

    We use the video-level topic as the primary search anchor so long-video
    chunks do not drift into unrelated subtopics. The chunk topic is only used
    as a light refinement hint in the prompt, not as the search target.
    """
    if not USE_TAVILY_CONTEXT or not TAVILY_API_KEY:
        return _default_external_context_block()

    combined_query = video_topic.strip()
    if chunk_topic.strip() and chunk_topic.strip().lower() != video_topic.strip().lower():
        combined_query = f"{video_topic}. Context tambahan dari bagian ini: {chunk_topic}"

    try:
        from tavily import TavilyClient

        tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        response = tavily_client.search(query=combined_query, search_depth="basic", max_results=3)
        contexts = []
        for result in response.get("results", [])[:3]:
            content = str(result.get("content", "")).strip()
            if content:
                contexts.append(content)
        context = " ".join(contexts).strip() or "unavailable"
        return (
            f"[VIDEO THEME: {video_topic}] [EXTERNAL CONTEXT: {context}] "
            "Gunakan konteks eksternal ini untuk menilai sensitivitas, relevansi, dan potensi viral, "
            "tetapi tetap pertahankan tema utama video dan jangan lompat ke topik yang tidak sejalan."
        )
    except Exception:
        print("[WARN] Tavily context failed, proceeding without external context", flush=True)
        return _default_external_context_block()


def call_muapi_llm(prompt: str) -> str:
    """Default LLM backend: MuAPI gpt-5-mini."""
    result = muapi.run(
        "gpt-5-mini",
        {"prompt": prompt},
        label="gpt-5-mini",
        timeout=GPT_CALL_TIMEOUT_SECONDS,
    )

    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str) and outputs[0].strip():
        return outputs[0]

    for key in ("output", "text", "response", "result", "content"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = v.get("text") or v.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]

    raise RuntimeError(f"Could not extract gpt-5-mini text from response: {result}")


def clean_json_response(text: str) -> str:
    """Extract the most likely JSON object from a noisy LLM response."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Normalize common LLM time suffixes like `677.1s` into valid JSON numbers.
    # This is intentionally conservative: only strip a trailing `s` when it is
    # attached to a numeric literal and appears before a JSON delimiter.
    cleaned = re.sub(r"(\d+\.?\d*)s\b", r"\1", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    # Best-effort validation that we are looking at a JSON-like object.
    if not re.match(r"^\s*\{[\s\S]*\}\s*$", cleaned):
        return cleaned
    return cleaned


def _parse_json_loose(raw: str) -> Dict:
    """Parse noisy JSON from LLMs with json5 fallback."""
    text = clean_json_response(raw)
    try:
        return json.loads(text)
    except Exception as first_error:
        try:
            import json5  # type: ignore
        except Exception:
            print(f"[json-parse] raw LLM response:\n{raw}", flush=True)
            raise first_error

        try:
            return json5.loads(text)
        except Exception:
            print(f"[json-parse] raw LLM response:\n{raw}", flush=True)
            raise


def detect_content_type(transcript: Dict, llm_fn: LLMFn = call_muapi_llm) -> Dict[str, str]:
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments)


def chunk_transcript(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks = []
    start = 0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        chunk_segs = [
            s for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            chunk = dict(transcript)
            chunk["segments"] = chunk_segs
            chunk["duration"] = end - start
            chunk["_offset"] = start
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    is_chunk: bool = False,
    external_context_block: str = "",
    llm_fn: LLMFn = call_muapi_llm,
) -> Dict:
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload (which times out gpt-5-mini).
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 90))
    min_clips = min(target, natural_max, 8)
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        num_clips_instruction=f"Generate at least {min_clips} highlights",
        external_context_block=external_context_block or _default_external_context_block(),
    )
    full_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    raw = llm_fn(full_prompt)
    try:
        return _parse_json_loose(raw)
    except Exception:
        print(f"[highlights] raw LLM response for debugging:\n{raw}", flush=True)
        raise


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-scoring one already kept."""
    highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def _normalize_highlight_bounds(highlight: Dict, duration: float) -> Optional[Dict]:
    """Clamp and repair highlight timestamps before the pipeline filters them.

    This prevents bad LLM output like zero-length clips or clips that extend
    past the transcript duration from poisoning the final crop stage.
    """
    try:
        start_time = float(highlight.get("start_time", 0.0))
        end_time = float(highlight.get("end_time", 0.0))
    except (TypeError, ValueError):
        return None

    if duration > 0:
        start_time = max(0.0, min(start_time, duration))
        end_time = max(0.0, min(end_time, duration))

    if end_time <= start_time:
        # Repair obviously broken ranges by expanding to a reasonable default.
        repaired_start = max(0.0, min(start_time, max(duration - 30.0, 0.0)))
        repaired_end = repaired_start + 30.0
        if duration > 0:
            repaired_end = min(repaired_end, duration)
        start_time, end_time = repaired_start, repaired_end

    if end_time <= start_time:
        return None

    return {**highlight, "start_time": start_time, "end_time": end_time}


def get_highlights(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[LLMFn] = None,
) -> Dict:
    """Main entry point — returns {highlights: [...]} sorted by score.

    `llm_fn` swaps the underlying LLM. Defaults to MuAPI gpt-5-mini; local
    mode passes in a local LLM-backed callable.
    """
    llm_fn = llm_fn or call_muapi_llm
    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(f"[highlights] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s", flush=True)

    transcript_text = build_transcript_text(transcript)
    video_topic = _extract_chunk_topic(transcript_text)
    if not video_topic:
        try:
            video_topic = extract_topic_with_llm(transcript_text[:4000], llm_fn)
        except Exception:
            video_topic = ""
    if video_topic:
        print(f"[highlights] video topic anchor: {video_topic}", flush=True)
    else:
        video_topic = _extract_chunk_topic(transcript_text)

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True)
        all_highlights: List[Dict] = []
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk)
            print(f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            try:
                chunk_topic = extract_topic_with_llm(text, llm_fn)
                print(f"[TAVILY] Extracted topic for search: {chunk_topic}", flush=True)
            except Exception as e:
                print(f"[WARN] LLM topic extraction failed: {e}. Fallback to first sentences.", flush=True)
                chunk_topic = _extract_chunk_topic(text)
            external_context_block = _build_theme_locked_external_context_block(video_topic, chunk_topic)
            result = call_highlight_api(
                text,
                content_info,
                chunk["duration"],
                num_clips=num_clips,
                is_chunk=True,
                external_context_block=external_context_block,
                llm_fn=llm_fn,
            )
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                normalized = _normalize_highlight_bounds(h, duration)
                if normalized is not None:
                    all_highlights.append(normalized)
        highlights = dedupe_highlights(all_highlights)
    else:
        external_context_block = _build_theme_locked_external_context_block(video_topic, video_topic)
        result = call_highlight_api(
            transcript_text,
            content_info,
            duration,
            num_clips=num_clips,
            external_context_block=external_context_block,
            llm_fn=llm_fn,
        )
        normalized_highlights = [
            item
            for item in (
                _normalize_highlight_bounds(h, duration)
                for h in result.get("highlights", [])
            )
            if item is not None
        ]
        highlights = dedupe_highlights(normalized_highlights)

    return {"highlights": highlights}
