"""Local clipping: ffmpeg subclip + vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio.

The legacy cropper remains the default safe path. A podcast-specific YOLO
pipeline can be enabled by callers and falls back to the legacy path on any
runtime failure.
"""

import os
import subprocess
from math import hypot
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from ..config import LOCAL_OUTPUT_DIR


def _ratio(aspect_ratio: str) -> float:
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _crop_size(src_w: int, src_h: int, target_ratio: float) -> Tuple[int, int]:
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))
    return crop_w, crop_h


def _fit_ratio(crop_w: int, crop_h: int, target_ratio: float) -> Tuple[int, int]:
    current_ratio = crop_w / float(max(crop_h, 1))
    if current_ratio > target_ratio:
        crop_w = int(crop_h * target_ratio)
    else:
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))
    return crop_w, crop_h


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _center_from_box(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _box_from_center(
    center_x: float,
    center_y: float,
    crop_w: int,
    crop_h: int,
    src_w: int,
    src_h: int,
) -> Tuple[int, int, int, int]:
    x0 = int(round(center_x - crop_w / 2.0))
    y0 = int(round(center_y - crop_h / 2.0))
    x0 = int(_clamp(x0, 0, max(0, src_w - crop_w)))
    y0 = int(_clamp(y0, 0, max(0, src_h - crop_h)))
    return x0, y0, x0 + crop_w, y0 + crop_h


def _box_wh(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    return max(1.0, x1 - x0), max(1.0, y1 - y0)


def _box_area(box: Tuple[float, float, float, float]) -> float:
    w, h = _box_wh(box)
    return w * h


def _smooth_value(previous: Optional[float], current: float, alpha: float) -> float:
    if previous is None:
        return current
    return previous + (current - previous) * alpha


def _smooth_box(
    previous: Optional[Tuple[int, int, int, int]],
    current: Tuple[int, int, int, int],
    alpha: float,
) -> Tuple[int, int, int, int]:
    if previous is None:
        return current
    return tuple(
        int(round(_smooth_value(float(prev), float(cur), alpha)))
        for prev, cur in zip(previous, current)
    )  # type: ignore[return-value]


def _active_segment(current_timestamp: float, srt_segments: Sequence[Dict]) -> Optional[Dict]:
    for segment in srt_segments:
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", 0.0) or 0.0)
        if start <= current_timestamp <= end:
            return segment
    return None


def _slice_segments_for_clip(
    segments: Sequence[Dict],
    clip_start: float,
    clip_end: float,
) -> List[Dict]:
    clipped: List[Dict] = []
    for segment in segments:
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", 0.0) or 0.0)
        overlap_start = max(start, clip_start)
        overlap_end = min(end, clip_end)
        if overlap_end <= overlap_start:
            continue
        clipped.append(
            {
                "start": overlap_start - clip_start,
                "end": overlap_end - clip_start,
                "text": str(segment.get("text", "")).strip(),
            }
        )
    return clipped


def _motion_mask_center(gray_prev: np.ndarray, gray_curr: np.ndarray) -> Optional[Tuple[float, float]]:
    diff = cv2.absdiff(gray_prev, gray_curr)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)
    _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=1)
    moments = cv2.moments(thresh)
    if moments["m00"] <= 0:
        return None
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def _get_yolo_model():
    """Lazy-load YOLOv8n for podcast crop detection.

    Returns None if the model cannot be loaded so callers can fall back safely.
    """
    cached = getattr(_get_yolo_model, "_cached", None)
    if cached is not None:
        return cached

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        print(f"[PODCAST WARN] ultralytics unavailable: {e}", flush=True)
        return None

    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"[PODCAST WARN] failed to load YOLOv8n: {e}", flush=True)
        return None

    setattr(_get_yolo_model, "_cached", model)
    return model


def _extract_tracks(frame: np.ndarray, model) -> List[Dict]:
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        classes=[0],
    )
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    tracks: List[Dict] = []
    for idx, box in enumerate(boxes):
        xyxy = box.xyxy[0].tolist()
        x0, y0, x1, y1 = [float(v) for v in xyxy]
        track_id = idx
        if getattr(box, "id", None) is not None:
            try:
                track_id = int(box.id.item())
            except Exception:
                try:
                    track_id = int(box.id[0])
                except Exception:
                    track_id = idx

        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        tracks.append(
            {
                "track_id": track_id,
                "bbox": (x0, y0, x1, y1),
                "center": ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                "face_area": width * height,
                "confidence": float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0,
            }
        )

    tracks.sort(key=lambda item: item["center"][0])
    return tracks


def get_stable_podcast_crop(
    frame: np.ndarray,
    current_timestamp: float,
    srt_segments: Sequence[Dict],
    prev_state: Optional[Dict],
) -> Tuple[Optional[Tuple[int, int, int, int]], Dict]:
    """Return a stable crop box and next state for podcast videos.

    The function prefers:
      1. locked_id if still present,
      2. SRT-aligned active speaker with motion in lower face area,
      3. largest tracked person,
      4. None to trigger center-crop fallback.
    """
    state = dict(prev_state or {})
    src_h, src_w = frame.shape[:2]
    target_ratio = _ratio("9:16")
    crop_w, crop_h = _fit_ratio(*_crop_size(src_w, src_h, target_ratio), target_ratio)

    model = _get_yolo_model()
    if model is None:
        return None, state

    try:
        tracks = _extract_tracks(frame, model)
    except Exception as e:
        print(f"[PODCAST WARN] YOLO tracking failed: {e}", flush=True)
        return None, state

    now_segment = _active_segment(current_timestamp, srt_segments)
    speech_active = now_segment is not None

    prev_gray = state.get("prev_gray")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    state["prev_gray"] = gray

    locked_id = state.get("locked_id")
    locked_box = state.get("last_box")
    locked_ts = float(state.get("locked_ts", -1.0) or -1.0)
    last_seen_ts = float(state.get("last_seen_ts", -1.0) or -1.0)

    def lower_face_motion_score(track: Dict) -> float:
        x0, y0, x1, y1 = [int(v) for v in track["bbox"]]
        h = max(1, y1 - y0)
        lower_start = y0 + int(h * 0.60)
        roi = gray[lower_start:y1, x0:x1]
        if roi.size == 0 or prev_gray is None:
            return 0.0
        prev_roi = prev_gray[lower_start:y1, x0:x1]
        if prev_roi.size == 0:
            return 0.0
        diff = cv2.absdiff(prev_roi, roi)
        return float(np.mean(diff))

    def mouth_motion_score(track: Dict) -> float:
        x0, y0, x1, y1 = [int(v) for v in track["bbox"]]
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        mouth_top = y0 + int(h * 0.60)
        mouth_bottom = y0 + int(h * 0.88)
        mouth_left = x0 + int(w * 0.25)
        mouth_right = x0 + int(w * 0.75)
        roi = gray[mouth_top:mouth_bottom, mouth_left:mouth_right]
        if roi.size == 0 or prev_gray is None:
            return 0.0
        prev_roi = prev_gray[mouth_top:mouth_bottom, mouth_left:mouth_right]
        if prev_roi.size == 0:
            return 0.0
        diff = cv2.absdiff(prev_roi, roi)
        return float(np.mean(diff))

    def motion_score_person(track: Dict) -> float:
        return mouth_motion_score(track)

    def normalize_box(box: Tuple[float, float, float, float]) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, src_w))
        y1 = max(0, min(y1, src_h))
        x2 = max(0, min(x2, src_w))
        y2 = max(0, min(y2, src_h))
        w = x2 - x1
        h = y2 - y1
        if w < 100 or h < 100:
            return None
        return (x1, y1, x2, y2)

    def center_distance_score(track: Dict) -> float:
        cx, cy = track["center"]
        return 1.0 - min(hypot(cx - (src_w / 2.0), cy - (src_h / 2.0)) / hypot(src_w, src_h), 1.0)

    def speaker_score(track: Dict) -> float:
        score = 0.0
        if speech_active:
            score += 0.55
        score += 0.25 * center_distance_score(track)
        score += 0.20 * min(track["face_area"] / float(src_w * src_h), 0.25) / 0.25
        score += min(mouth_motion_score(track) / 22.0, 1.0) * 0.35
        return score

    motion_scores = {int(t["track_id"]): motion_score_person(t) for t in tracks}
    print(f"[DEBUG] Timestamp: {current_timestamp}s", flush=True)
    print(f"[DEBUG] Active SRT segment: {now_segment}", flush=True)
    print(f"[DEBUG] Tracked persons: {[int(t['track_id']) for t in tracks]}", flush=True)
    print(f"[DEBUG] Locked ID: {locked_id}", flush=True)
    print(f"[DEBUG] Motion score person 1: {motion_scores.get(1, 0.0)}", flush=True)
    print(f"[DEBUG] Motion score person 2: {motion_scores.get(2, 0.0)}", flush=True)

    track_by_id = {int(t["track_id"]): t for t in tracks}

    if locked_id is not None and locked_id in track_by_id and now_segment is None:
        chosen = track_by_id[locked_id]
        state["last_seen_ts"] = current_timestamp
        box = normalize_box(tuple(chosen["bbox"]))
        if box is None:
            return None, state
        state["last_box"] = box
        state["locked_ts"] = current_timestamp
        print(f"[DEBUG] Final crop box: {box}", flush=True)
        return state["last_box"], state

    if locked_id is not None and locked_box is not None and last_seen_ts >= 0.0:
        if current_timestamp - last_seen_ts < (45.0 / 30.0):
            state["last_box"] = tuple(int(v) for v in locked_box)
            return state["last_box"], state

    candidate: Optional[Dict] = None
    if now_segment is not None and tracks:
        scored = sorted(tracks, key=speaker_score, reverse=True)
        candidate = scored[0]
        for t in scored:
            if motion_score_person(t) > 6.0:
                candidate = t
                break

    if candidate is None and tracks:
        candidate = max(tracks, key=lambda t: t["face_area"])

    if candidate is None and locked_id is not None and locked_id in track_by_id:
        candidate = track_by_id[locked_id]

    if candidate is None:
        state["locked_id"] = None
        return None, state

    box = normalize_box(tuple(candidate["bbox"]))
    if box is None:
        return None, state
    if locked_box is not None:
        box = _smooth_box(locked_box, box, 0.22)

    state["locked_id"] = int(candidate["track_id"])
    state["locked_ts"] = current_timestamp
    state["last_seen_ts"] = current_timestamp
    state["last_box"] = box
    print(f"[DEBUG] Final crop box: {box}", flush=True)
    return box, state


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        source_path,
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def extract_audio_energy_array(video_path: str, fps: float) -> List[float]:
    try:
        import librosa  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "librosa is required for local cropping. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    try:
        try:
            y, sr = librosa.load(video_path, sr=16000, mono=True, backend="soundfile")
        except TypeError:
            y, sr = librosa.load(video_path, sr=16000, mono=True)
    except Exception as e:
        raise RuntimeError(f"failed to load audio from {video_path}: {e}") from e

    if y.size == 0:
        return []

    hop_length = max(1, int(sr / max(fps, 1.0)))
    frame_length = max(hop_length * 2, 2048)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

    energy_by_timestamp = list(zip(times.tolist(), rms.tolist()))
    if not energy_by_timestamp:
        return []

    duration = len(y) / float(sr)
    total_frames = max(1, int(duration * fps) + 1)
    energy_by_frame: List[float] = []
    cursor = 0
    for frame_index in range(total_frames):
        timestamp = frame_index / float(fps)
        while cursor + 1 < len(energy_by_timestamp) and energy_by_timestamp[cursor + 1][0] <= timestamp:
            cursor += 1
        energy_by_frame.append(float(energy_by_timestamp[cursor][1]))

    max_energy = max(energy_by_frame) if energy_by_frame else 0.0
    if max_energy > 0:
        energy_by_frame = [value / max_energy for value in energy_by_frame]
    return energy_by_frame


def _load_legacy_fallback() -> None:
    """Placeholder to keep legacy path isolated in the same file.

    The legacy crop pipeline below remains unchanged in behavior.
    """


def _reframe_vertical(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    debug: bool = False,
    srt_segments: Optional[Sequence[Dict]] = None,
) -> str:
    """Legacy crop path.

    This is intentionally left as the safe fallback path.
    """
    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    audio_energy = extract_audio_energy_array(in_path, fps)

    crop_w, crop_h = _fit_ratio(*_crop_size(src_w, src_h, target_ratio), target_ratio)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    silent_path = out_path + ".silent.mp4"
    debug_path = os.path.join(os.path.dirname(out_path), "output_debug.mp4") if debug else None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))
    debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (src_w, src_h)) if debug_path else None

    last_center: Optional[Tuple[int, int]] = None
    smoothing = 0.18

    def _best_subject(frame, face_data, pose_data, audio_energy_arr, frame_index: int) -> Dict:
        frame_h, frame_w = frame.shape[:2]
        energy = 0.0
        if audio_energy_arr is not None:
            try:
                energy = float(audio_energy_arr[min(frame_index, len(audio_energy_arr) - 1)])
            except (TypeError, ValueError, IndexError):
                energy = 0.0

        audio_high = energy > 0.1
        pose_center = None
        pose_stability = 0.0
        if pose_data:
            pose_center = pose_data.get("pose_center")
            pose_stability = float(pose_data.get("pose_stability", 0.0) or 0.0)

        best_candidate: Optional[Dict] = None
        best_score = -1.0
        for face in face_data or []:
            face_center = face.get("face_center")
            if not face_center:
                continue

            face_area = int(face.get("face_area", 1) or 1)
            mouth_openness_variance = float(face.get("mouth_openness_variance", 0.0) or 0.0)
            mouth_motion_score = min(mouth_openness_variance / 0.01, 1.0)

            center_x = frame_w / 2.0
            center_y = frame_h / 2.0
            distance_from_center = hypot(face_center[0] - center_x, face_center[1] - center_y)
            face_center_score = 1.0 - min(distance_from_center / hypot(frame_w, frame_h), 1.0)

            pose_alignment_score = 0.0
            if pose_center is not None:
                pose_distance = hypot(face_center[0] - pose_center[0], face_center[1] - pose_center[1])
                pose_alignment_score = 1.0 - min(pose_distance / hypot(frame_w, frame_h), 1.0)

            face_size_score = min(face_area / float(frame_w * frame_h), 0.18) / 0.18

            if audio_high:
                activity_score = (0.70 * mouth_motion_score) + (0.30 * face_center_score)
            else:
                activity_score = (0.50 * pose_stability) + (0.30 * face_center_score) + (0.20 * pose_alignment_score)

            base_score = (0.25 * face_center_score) + (0.20 * face_size_score) + (0.20 * pose_alignment_score)
            score = base_score + activity_score

            candidate = {
                "face_id": face.get("face_id"),
                "face_center": face_center,
                "face_area": face_area,
                "mouth_openness_variance": mouth_openness_variance,
                "mouth_motion_score": mouth_motion_score,
                "face_center_score": face_center_score,
                "pose_alignment_score": pose_alignment_score,
                "pose_stability": pose_stability,
                "audio_energy": energy,
                "score": score,
            }

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            fallback_center = pose_center if pose_center is not None else (frame_w // 2, frame_h // 2)
            best_candidate = {
                "face_id": None,
                "face_center": fallback_center,
                "face_area": 1,
                "mouth_openness_variance": 0.0,
                "mouth_motion_score": 0.0,
                "face_center_score": 0.0,
                "pose_alignment_score": 0.0,
                "pose_stability": pose_stability,
                "audio_energy": energy,
                "score": 0.0,
            }

        return best_candidate

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            selected_subject = None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                import mediapipe as mp  # type: ignore

                pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                pose_result = pose.process(rgb)
                face_result = face_mesh.process(rgb)
                pose_center = None
                if pose_result.pose_landmarks:
                    landmarks = pose_result.pose_landmarks
                    h, w = frame.shape[:2]
                    candidates: List[Tuple[Tuple[int, int], float]] = []
                    for landmark, weight in (
                        (mp.solutions.pose.PoseLandmark.NOSE, 3.0),
                        (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, 2.0),
                        (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, 2.0),
                        (mp.solutions.pose.PoseLandmark.LEFT_HIP, 1.5),
                        (mp.solutions.pose.PoseLandmark.RIGHT_HIP, 1.5),
                    ):
                        lm = landmarks.landmark[landmark]
                        if lm.visibility < 0.45:
                            continue
                        candidates.append(((int(lm.x * w), int(lm.y * h)), weight))
                    if candidates:
                        total = sum(weight for _, weight in candidates)
                        pose_center = (
                            int(sum(point[0] * weight for point, weight in candidates) / total),
                            int(sum(point[1] * weight for point, weight in candidates) / total),
                        )
                face_data: List[Dict] = []
                if face_result.multi_face_landmarks:
                    for idx, face_landmarks in enumerate(face_result.multi_face_landmarks):
                        xs = [lm.x for lm in face_landmarks.landmark]
                        ys = [lm.y for lm in face_landmarks.landmark]
                        x0 = max(0, int(min(xs) * src_w))
                        y0 = max(0, int(min(ys) * src_h))
                        x1 = min(src_w, int(max(xs) * src_w))
                        y1 = min(src_h, int(max(ys) * src_h))
                        face_area = max(1, (x1 - x0) * (y1 - y0))
                        face_center = ((x0 + x1) // 2, (y0 + y1) // 2)
                        top = face_landmarks.landmark[13]
                        bottom = face_landmarks.landmark[14]
                        mouth_variance = float(hypot(top.x - bottom.x, top.y - bottom.y))
                        face_data.append(
                            {
                                "face_id": idx,
                                "face_center": face_center,
                                "face_area": face_area,
                                "mouth_openness_variance": mouth_variance,
                            }
                        )
                pose_data = None
                if pose_center is not None:
                    pose_data = {"pose_center": pose_center, "pose_stability": 1.0}
                selected_subject = _best_subject(frame, face_data, pose_data, audio_energy, frame_index)
            except Exception:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                next_center = None
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    next_center = (x + w // 2, y + h // 2)
                if next_center is None:
                    next_center = (src_w // 2, src_h // 2)

                if last_center is None:
                    last_center = next_center
                else:
                    lx, ly = last_center
                    cx, cy = next_center
                    last_center = (
                        int(lx + (cx - lx) * smoothing),
                        int(ly + (cy - ly) * smoothing),
                    )
                cx, cy = last_center
                x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
                y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
                cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
                writer.write(cropped)
                if debug and debug_writer is not None:
                    debug_writer.write(frame)
                continue

            next_center: Optional[Tuple[int, int]] = None
            if selected_subject and selected_subject.get("face_center"):
                next_center = selected_subject["face_center"]

            if next_center is not None:
                if last_center is None:
                    last_center = next_center
                else:
                    lx, ly = last_center
                    cx, cy = next_center
                    last_center = (
                        int(lx + (cx - lx) * smoothing),
                        int(ly + (cy - ly) * smoothing),
                    )
            if last_center is None:
                last_center = (src_w // 2, src_h // 2)

            cx, cy = last_center
            x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
            y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
            cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
            writer.write(cropped)

            if debug and debug_writer is not None:
                overlay = frame.copy()
                if selected_subject and selected_subject.get("face_center"):
                    fx, fy = selected_subject["face_center"]
                    box_w = 160
                    box_h = 200
                    cv2.rectangle(
                        overlay,
                        (max(0, fx - box_w // 2), max(0, fy - box_h // 2)),
                        (min(src_w - 1, fx + box_w // 2), min(src_h - 1, fy + box_h // 2)),
                        (0, 255, 0),
                        3,
                    )

                score_text = f"Score: {selected_subject.get('score', 0.0):.3f}" if selected_subject else "Score: 0.000"
                energy_text = f"Audio Energy: {selected_subject.get('audio_energy', 0.0):.3f}" if selected_subject else "Audio Energy: 0.000"
                mouth_text = f"Mouth Variance: {selected_subject.get('mouth_openness_variance', 0.0):.5f}" if selected_subject else "Mouth Variance: 0.00000"
                cv2.putText(overlay, score_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(overlay, energy_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(overlay, mouth_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                debug_writer.write(overlay)
    finally:
        cap.release()
        writer.release()
        if debug_writer is not None:
            debug_writer.release()

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        silent_path,
        "-i",
        in_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def _reframe_vertical_podcast(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    debug: bool = False,
    srt_segments: Optional[Sequence[Dict]] = None,
) -> str:
    """Podcast wrapper with YOLO + SRT-aware active speaker detection.

    Falls back to legacy crop if YOLO fails to load or throws at runtime.
    """
    try:
        from ultralytics import YOLO  # type: ignore  # noqa: F401
    except Exception as e:
        print(f"[PODCAST WARN] ultralytics unavailable ({e}). Using legacy crop.", flush=True)
        return _reframe_vertical(in_path, out_path, aspect_ratio, debug=debug, srt_segments=srt_segments)

    try:
        return _reframe_vertical_podcast_impl(in_path, out_path, aspect_ratio, debug=debug, srt_segments=srt_segments)
    except Exception as e:
        print(f"[PODCAST WARN] YOLO pipeline failed ({e}). Using legacy crop.", flush=True)
        return _reframe_vertical(in_path, out_path, aspect_ratio, debug=debug, srt_segments=srt_segments)


def _reframe_vertical_podcast_impl(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    debug: bool = False,
    srt_segments: Optional[Sequence[Dict]] = None,
) -> str:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(f"ultralytics import failed: {e}") from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    crop_w, crop_h = _crop_size(src_w, src_h, target_ratio)

    model = YOLO("yolov8n.pt")

    silent_path = out_path + ".silent.mp4"
    debug_path = os.path.join(os.path.dirname(out_path), "output_debug.mp4") if debug else None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))
    debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (src_w, src_h)) if debug_path else None

    state: Dict = {}
    prev_gray: Optional[np.ndarray] = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        box, state = get_stable_podcast_crop(frame, current_timestamp, srt_segments or [], {**state, "prev_gray": prev_gray})
        prev_gray = state.get("prev_gray")

        if box is None:
            x0, y0, x1, y1 = _box_from_center(src_w / 2.0, src_h / 2.0, crop_w, crop_h, src_w, src_h)
        else:
            x0, y0, x1, y1 = box

        cropped = frame[y0:y1, x0:x1]
        if cropped.shape[0] != crop_h or cropped.shape[1] != crop_w:
            cropped = cv2.resize(cropped, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        writer.write(cropped)

        if debug and debug_writer is not None:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 3)
            cv2.putText(
                overlay,
                f"PODCAST ID: {state.get('locked_id', 'n/a')}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            debug_writer.write(overlay)

    cap.release()
    writer.release()
    if debug_writer is not None:
        debug_writer.release()

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        silent_path,
        "-i",
        in_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    srt_segments: Optional[Sequence[Dict]] = None,
    debug: bool = False,
    podcast: bool = False,
) -> str:
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        clip_segments = _slice_segments_for_clip(srt_segments or [], start_time, end_time)
        if podcast:
            _reframe_vertical_podcast(cut_path, out_path, aspect_ratio, debug=debug, srt_segments=clip_segments)
        else:
            _reframe_vertical(cut_path, out_path, aspect_ratio, debug=debug, srt_segments=clip_segments)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    transcript: Optional[Dict] = None,
    debug: bool = False,
    podcast: bool = False,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    transcript_segments = transcript.get("segments", []) if transcript else []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                srt_segments=transcript_segments,
                debug=debug,
                podcast=podcast,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
