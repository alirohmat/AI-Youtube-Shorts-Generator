"""Local clipping: ffmpeg subclip + multimodal subject-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
   2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
      window across the frame to keep the main subject centred using a blend of
      MediaPipe Pose, Face Mesh mouth motion, and face detection fallback.
"""
import os
import subprocess
from collections import deque
from math import hypot
from typing import Dict, List, Optional, Tuple

import mediapipe as mp

from ..config import LOCAL_OUTPUT_DIR


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def extract_audio_energy_array(video_path: str, fps: float) -> List[float]:
    """Extract RMS energy and map it to video frame timestamps.

    The returned list is aligned to frame index by timestamp so the caller can
    read energy[frame_index] safely during frame processing.
    """
    try:
        import librosa  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "librosa is required for audio-aware local cropping. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    # Prefer soundfile to avoid audioread deprecation warnings when available.
    try:
        import soundfile as _soundfile  # type: ignore
        _ = _soundfile  # keep import explicit for readability
    except Exception:
        pass

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


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str, debug: bool = False) -> str:
    """Crop the cut clip to the target aspect ratio, tracking the main subject."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    try:
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
    except AttributeError:
        mp_face_mesh = mp.face_mesh
        mp_pose = mp.pose

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    audio_energy = extract_audio_energy_array(in_path, fps)

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    pose = None
    face_mesh = None
    try:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        if pose is not None:
            pose.close()
        if face_mesh is not None:
            face_mesh.close()
        raise

    silent_path = out_path + ".silent.mp4"
    debug_path = os.path.join(os.path.dirname(out_path), "output_debug.mp4") if debug else None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))
    debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (src_w, src_h)) if debug_path else None

    last_center: Optional[Tuple[int, int]] = None
    smoothing = 0.18  # how aggressively to chase a new subject position

    lip_top = 13
    lip_bottom = 14
    mouth_history: Dict[int, deque] = {}

    def _landmark_xy(frame, lm) -> Tuple[int, int]:
        h, w, _ = frame.shape
        return int(lm.x * w), int(lm.y * h)

    def _pose_center(frame, landmarks) -> Optional[Tuple[int, int]]:
        h, w, _ = frame.shape
        candidates: List[Tuple[Tuple[int, int], float]] = []
        for landmark, weight in (
            (mp_pose.PoseLandmark.NOSE, 3.0),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, 2.0),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, 2.0),
            (mp_pose.PoseLandmark.LEFT_HIP, 1.5),
            (mp_pose.PoseLandmark.RIGHT_HIP, 1.5),
        ):
            lm = landmarks.landmark[landmark]
            if lm.visibility < 0.45:
                continue
            candidates.append(((int(lm.x * w), int(lm.y * h)), weight))
        if not candidates:
            nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            if nose.visibility < 0.3:
                return None
            return int(nose.x * w), int(nose.y * h)
        total = sum(weight for _, weight in candidates)
        cx = int(sum(point[0] * weight for point, weight in candidates) / total)
        cy = int(sum(point[1] * weight for point, weight in candidates) / total)
        return cx, cy

    def _face_center(frame, face_landmarks) -> Tuple[int, int]:
        h, w, _ = frame.shape
        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]
        return int(((min(xs) + max(xs)) / 2.0) * w), int(((min(ys) + max(ys)) / 2.0) * h)

    def _mouth_motion_score(face_landmarks, face_id: int) -> float:
        top = face_landmarks.landmark[13]
        bottom = face_landmarks.landmark[14]
        dist = hypot(top.x - bottom.x, top.y - bottom.y)
        hist = mouth_history.setdefault(face_id, deque(maxlen=10))
        hist.append(dist)
        if len(hist) < 6:
            return 0.0
        mean = sum(hist) / len(hist)
        variance = sum((x - mean) ** 2 for x in hist) / len(hist)
        return variance ** 0.5

    def _score_candidate(
        pose_center: Optional[Tuple[int, int]],
        face_center: Tuple[int, int],
        mouth_score: float,
        face_area: int,
        frame_w: int,
        frame_h: int,
    ) -> float:
        center_x = frame_w / 2.0
        center_y = frame_h / 2.0
        distance_from_center = hypot(face_center[0] - center_x, face_center[1] - center_y)
        center_bonus = 1.0 - min(distance_from_center / hypot(frame_w, frame_h), 1.0)

        pose_bonus = 0.0
        if pose_center is not None:
            pose_distance = hypot(face_center[0] - pose_center[0], face_center[1] - pose_center[1])
            pose_bonus = 1.0 - min(pose_distance / hypot(frame_w, frame_h), 1.0)

        area_bonus = min(face_area / float(frame_w * frame_h), 0.15) / 0.15
        mouth_bonus = min(mouth_score / 0.01, 1.0)

        return (0.32 * center_bonus) + (0.28 * pose_bonus) + (0.22 * area_bonus) + (0.18 * mouth_bonus)

    def select_best_subject(
        frame,
        face_data,
        pose_data,
        audio_energy,
        frame_index: int,
        previous_subject: Optional[Dict] = None,
    ) -> Dict:
        """Pick the best subject for the current frame.

        face_data is expected to be an iterable of dicts with at least:
          - face_center: (x, y)
          - face_area: int
          - mouth_openness: float
          - face_id: int

        pose_data is expected to be either None or a dict with:
          - pose_center: (x, y)
          - pose_stability: float in [0, 1]

        audio_energy is a sequence aligned with frame_index.
        """

        frame_h, frame_w = frame.shape[:2]
        energy = 0.0
        if audio_energy is not None:
            try:
                energy = float(audio_energy[min(frame_index, len(audio_energy) - 1)])
            except (TypeError, ValueError, IndexError):
                energy = 0.0

        audio_threshold = 0.1
        audio_high = energy > audio_threshold

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
                # Hening: prioritaskan stabilitas pose dan posisi wajah; ignore mouth motion.
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

        if previous_subject and previous_subject.get("score") is not None:
            prev_score = float(previous_subject.get("score", 0.0) or 0.0)
            if prev_score > 0:
                delta = best_candidate["score"] - prev_score
                if delta / prev_score < 0.15:
                    return previous_subject

        return best_candidate

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)
        face_result = face_mesh.process(rgb)

        pose_center = _pose_center(frame, pose_result.pose_landmarks) if pose_result.pose_landmarks else None

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
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
                face_center = _face_center(frame, face_landmarks)
                mouth_variance = _mouth_motion_score(face_landmarks, idx)
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
            pose_data = {
                "pose_center": pose_center,
                "pose_stability": 1.0 if pose_result.pose_landmarks else 0.0,
            }

        selected_subject = select_best_subject(
            frame=frame,
            face_data=face_data,
            pose_data=pose_data,
            audio_energy=audio_energy,
            frame_index=frame_index,
            previous_subject=previous_subject,
        )
        previous_subject = selected_subject

        next_center: Optional[Tuple[int, int]] = None
        if selected_subject and selected_subject.get("face_center"):
            next_center = selected_subject["face_center"]
        elif pose_center is not None:
            next_center = pose_center
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                next_center = (x + w // 2, y + h // 2)

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
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)

        if debug and debug_writer is not None:
            overlay = frame.copy()
            if selected_subject and selected_subject.get("face_center"):
                fx, fy = selected_subject["face_center"]
                box_w = 160
                box_h = 200
                cv2.rectangle(overlay, (max(0, fx - box_w // 2), max(0, fy - box_h // 2)),
                              (min(src_w - 1, fx + box_w // 2), min(src_h - 1, fy + box_h // 2)),
                              (0, 255, 0), 3)

            score_text = f"Score: {selected_subject.get('score', 0.0):.3f}" if selected_subject else "Score: 0.000"
            energy_text = f"Audio Energy: {selected_subject.get('audio_energy', 0.0):.3f}" if selected_subject else "Audio Energy: 0.000"
            mouth_text = f"Mouth Variance: {selected_subject.get('mouth_openness_variance', 0.0):.5f}" if selected_subject else "Mouth Variance: 0.00000"
            cv2.putText(overlay, score_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(overlay, energy_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(overlay, mouth_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            debug_writer.write(overlay)

    cap.release()
    if pose is not None:
        pose.close()
    if face_mesh is not None:
        face_mesh.close()
    writer.release()
    if debug_writer is not None:
        debug_writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
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
    debug: bool = False,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio, debug=debug)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    debug: bool = False,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
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
                debug=debug,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
