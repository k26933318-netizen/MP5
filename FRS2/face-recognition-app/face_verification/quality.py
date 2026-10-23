"""Quality gates for enrollment and verification frames."""

from dataclasses import dataclass
import math
from typing import Any, Literal

import cv2
import numpy as np

from .config import DEFAULT_SETTINGS, Settings

_PUBLIC_MESSAGES = {
    "FACE_TOO_SMALL": "Face too small - move closer.",
    "LOW_CONFIDENCE": "Please look straight at the camera.",
    "BAD_POSE": "Please look straight at the camera.",
    "BLURRY": "Image too blurry - hold still.",
    "BAD_LIGHTING": "Too dark or too bright - improve the lighting.",
    "MULTIPLE_FACES": "Only one face should be in view.",
    "NO_FACE": "No face detected - center your face.",
    "EYES_CLOSED": "Please keep your eyes open.",
    "OK": "",
}


@dataclass(frozen=True)
class QualityResult:
    """Frame quality result with safe public reason and non-image metrics."""

    ok: bool
    score: float
    reason_code: str
    metrics: dict[str, float | str | int]

    @property
    def public_message(self) -> str:
        """Return the stable user-facing text for this quality result."""
        return _PUBLIC_MESSAGES[self.reason_code]


def assess(
    image: np.ndarray,
    face: Any | None,
    faces: list[Any],
    mode: Literal["enroll", "verify"],
    settings: Settings = DEFAULT_SETTINGS,
) -> QualityResult:
    """Assess exactly-one-face, size, confidence, blur, lighting and optional gates."""
    if mode not in {"enroll", "verify"}:
        raise ValueError("mode must be 'enroll' or 'verify'.")
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return _fail("NO_FACE", {})
    if not faces or face is None:
        return _fail("NO_FACE", {"face_count": len(faces)})

    primary_area = _face_area(face)
    significant_faces = [
        candidate
        for candidate in faces
        if candidate is not face and _face_area(candidate) / max(primary_area, 1.0) >= settings.second_face_max_ratio
    ]
    metrics: dict[str, float | str | int] = {"face_count": len(faces)}
    if significant_faces:
        metrics["significant_face_count"] = len(significant_faces)
        return _fail("MULTIPLE_FACES", metrics)

    left, top, right, bottom = (float(value) for value in face.bbox)
    face_size = min(right - left, bottom - top)
    min_face_size = settings.enroll_min_face_px if mode == "enroll" else settings.verify_min_face_px
    metrics["face_min_dimension_px"] = face_size
    if face_size < min_face_size:
        return _fail("FACE_TOO_SMALL", metrics)

    detection_score = float(getattr(face, "det_score", 0.0))
    metrics["detection_score"] = detection_score
    if detection_score < settings.face_min_detection_score:
        return _fail("LOW_CONFIDENCE", metrics)

    crop = _clipped_crop(image, face.bbox)
    if crop.size == 0:
        return _fail("NO_FACE", metrics)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    metrics.update({"blur_variance": blur_variance, "brightness": brightness, "contrast": contrast})
    if blur_variance < settings.blur_threshold:
        return _fail("BLURRY", metrics)
    if brightness < settings.brightness_min or brightness > settings.brightness_max or contrast < settings.contrast_min:
        return _fail("BAD_LIGHTING", metrics)

    if settings.pose_gate:
        pose = _pose_degrees(face)
        metrics.update(pose)
        if pose["pose_source"] == "unavailable":
            return _fail("BAD_POSE", metrics)
        if (
            abs(float(pose["yaw_degrees"])) > settings.pose_yaw_max
            or abs(float(pose["pitch_degrees"])) > settings.pose_pitch_max
            or abs(float(pose["roll_degrees"])) > settings.pose_roll_max
        ):
            return _fail("BAD_POSE", metrics)

    if settings.eyes_open_gate:
        from .liveness import eye_aspect_ratio_106

        landmarks = getattr(face, "landmark_2d_106", None)
        if landmarks is None or eye_aspect_ratio_106(landmarks) < settings.eyes_open_min_ear:
            return _fail("EYES_CLOSED", metrics)

    blur_score = min(1.0, blur_variance / max(settings.blur_threshold * 2.0, 1.0))
    light_score = max(0.0, 1.0 - abs(brightness - 130.0) / 130.0)
    score = float(np.clip((detection_score + blur_score + light_score) / 3.0, 0.0, 1.0))
    return QualityResult(True, score, "OK", metrics)


def _fail(reason_code: str, metrics: dict[str, float | str | int]) -> QualityResult:
    return QualityResult(False, 0.0, reason_code, metrics)


def _face_area(face: Any) -> float:
    left, top, right, bottom = (float(value) for value in face.bbox)
    return max(0.0, right - left) * max(0.0, bottom - top)


def _clipped_crop(image: np.ndarray, bbox: Any) -> np.ndarray:
    height, width = image.shape[:2]
    left, top, right, bottom = bbox
    x1 = max(0, min(width, int(math.floor(left))))
    y1 = max(0, min(height, int(math.floor(top))))
    x2 = max(0, min(width, int(math.ceil(right))))
    y2 = max(0, min(height, int(math.ceil(bottom))))
    return image[y1:y2, x1:x2]


def _pose_degrees(face: Any) -> dict[str, float | str]:
    pose = getattr(face, "pose", None)
    if pose is not None and np.asarray(pose).size >= 3:
        values = np.asarray(pose, dtype="float32").reshape(-1)[:3]
        return {
            "yaw_degrees": float(values[0]),
            "pitch_degrees": float(values[1]),
            "roll_degrees": float(values[2]),
            "pose_source": "face.pose",
        }

    # Five-point keypoints provide only an approximate yaw/roll; pitch is unavailable.
    keypoints = getattr(face, "kps", None)
    if keypoints is not None and np.asarray(keypoints).shape[0] >= 3:
        points = np.asarray(keypoints, dtype="float32")
        eye_left, eye_right, nose = points[0], points[1], points[2]
        eye_midpoint = (eye_left + eye_right) / 2.0
        eye_vector = eye_right - eye_left
        eye_distance = max(float(np.linalg.norm(eye_vector)), 1e-6)
        horizontal_offset = float(np.dot(nose - eye_midpoint, eye_vector / eye_distance))
        yaw = math.degrees(math.atan2(2.0 * horizontal_offset, eye_distance))
        roll = math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))
        return {
            "yaw_degrees": yaw,
            "pitch_degrees": 0.0,
            "roll_degrees": roll,
            "pose_source": "approximate face.kps",
        }
    return {
        "yaw_degrees": 0.0,
        "pitch_degrees": 0.0,
        "roll_degrees": 0.0,
        "pose_source": "unavailable",
    }