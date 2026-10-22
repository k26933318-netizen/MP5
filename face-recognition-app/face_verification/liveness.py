"""Passive anti-spoofing, facial action challenges, and optional planar checks."""

import logging
import hashlib
import hmac
import math
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .config import DEFAULT_SETTINGS, Settings

_logger = logging.getLogger(__name__)
REAL_CLASS_INDEX = 1
ANTISPOOF_CROP_SIZE = 80
# InsightFace 2d106det eye contours, mapped to corner/upper/lower EAR order.
LEFT_EYE_106_INDICES = (63, 64, 65, 66, 68, 70)
RIGHT_EYE_106_INDICES = (76, 77, 78, 79, 81, 83)
CHALLENGE_ACTIONS = ("blink", "turn_left", "turn_right")


class AntiSpoofNotConfigured(RuntimeError):
    """Raised when required passive anti-spoofing models are unavailable."""


def preprocess_antispoof_crop(image: np.ndarray, face: Any, scale: float) -> np.ndarray:
    """Create a MiniFASNet tensor from an expanded face crop.

    The official Silent-Face-Anti-Spoofing predict path applies ``ToTensor`` to
    the OpenCV image directly: preserve BGR channel order, divide uint8 values by
    255, then convert HWC to float32 NCHW with batch size one. Its test script
    treats output class index 1 as the real-face class.
    """
    height, width = image.shape[:2]
    left, top, right, bottom = (float(value) for value in face.bbox)
    face_width = max(right - left, 1.0)
    face_height = max(bottom - top, 1.0)
    side = max(1, min(round(max(face_width, face_height) * scale), height, width))
    center_x = float(np.clip((left + right) / 2, side / 2, width - side / 2))
    center_y = float(np.clip((top + bottom) / 2, side / 2, height - side / 2))
    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))
    crop = image[y1:y1 + side, x1:x1 + side]
    if crop.size == 0:
        raise ValueError("Face crop is empty.")
    resized = cv2.resize(crop, (ANTISPOOF_CROP_SIZE, ANTISPOOF_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.transpose(resized.astype("float32") / 255.0, (2, 0, 1))[None, ...]


class AntiSpoofEnsemble:
    """Lazy, injectable pair of ONNX MiniFASNet sessions loaded only from local paths."""

    def __init__(
        self,
        settings: Settings = DEFAULT_SETTINGS,
        session_factory: Callable[[str], Any] | None = None,
        sessions: tuple[Any, Any] | None = None,
    ) -> None:
        self.settings = settings
        self._session_factory = session_factory
        self._sessions = sessions
        self._lock = threading.RLock()
        self._load_attempted = sessions is not None

    @property
    def configured(self) -> bool:
        """Whether both local models are loaded and available."""
        self.load()
        return self._sessions is not None

    def load(self) -> None:
        """Load both explicitly configured ONNX files; never download models."""
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            paths = (
                self.settings.antispoof_model_a_path,
                self.settings.antispoof_model_b_path,
            )
            if not all(paths):
                _logger.error(
                    "Passive anti-spoofing disabled: set ANTISPOOF_MODEL_A_PATH and ANTISPOOF_MODEL_B_PATH."
                )
                return
            missing = [path for path in paths if not Path(path).is_file()]
            if missing:
                _logger.error("Passive anti-spoofing disabled: configured model file is missing: %s", missing[0])
                return
            factory = self._session_factory
            if factory is None:
                import onnxruntime

                factory = lambda path: onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
            self._sessions = (factory(paths[0]), factory(paths[1]))

    def is_live(self, image: np.ndarray, face: Any) -> float:
        """Return the mean real-class probability from scale-2.7 and scale-4 models."""
        self.load()
        if self._sessions is None:
            raise AntiSpoofNotConfigured("Passive anti-spoofing is not configured.")
        scores = []
        for session, scale in zip(self._sessions, (2.7, 4.0)):
            tensor = preprocess_antispoof_crop(image, face, scale)
            input_name = session.get_inputs()[0].name
            output = np.asarray(session.run(None, {input_name: tensor})[0], dtype="float32").reshape(-1)
            probabilities = _softmax(output)
            if probabilities.size <= REAL_CLASS_INDEX:
                raise RuntimeError("Anti-spoof model output has no real-face class.")
            scores.append(float(probabilities[REAL_CLASS_INDEX]))
        return float(sum(scores) / len(scores))

    def evaluate_frames(self, frames: Sequence[tuple[np.ndarray, Any]]) -> tuple[bool, float, list[float]]:
        """Require mean and minimum per-frame liveness scores to clear configured thresholds."""
        if not frames:
            return False, 0.0, []
        scores = [self.is_live(image, face) for image, face in frames]
        mean_score = float(np.mean(scores))
        passes = mean_score >= self.settings.antispoof_threshold and min(scores) >= self.settings.antispoof_min_frame
        return passes, mean_score, scores


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def eye_aspect_ratio_106(landmarks: np.ndarray) -> float:
    """Calculate mean EAR using documented 2d106det eye contour sample indices.

    Each eye uses six contour points ordered outer, upper-outer, upper-inner,
    inner, lower-inner, lower-outer. The mapping is a model-layout assumption;
    validate it by rendering landmarks for the exact 2d106det package in use.
    """
    points = np.asarray(landmarks, dtype="float32")
    if points.ndim != 2 or points.shape[0] < 106 or points.shape[1] < 2:
        raise ValueError("106 two-dimensional landmarks are required.")

    def eye_ratio(indices: tuple[int, ...]) -> float:
        eye = points[list(indices), :2]
        horizontal = float(np.linalg.norm(eye[0] - eye[3]))
        if horizontal <= 1e-6:
            return 0.0
        vertical = float(np.linalg.norm(eye[1] - eye[5])) + float(np.linalg.norm(eye[2] - eye[4]))
        return vertical / (2.0 * horizontal)

    return (eye_ratio(LEFT_EYE_106_INDICES) + eye_ratio(RIGHT_EYE_106_INDICES)) / 2.0


def yaw_proxy(face: Any) -> float:
    """Return image-right-positive nose offset normalized by inter-eye distance."""
    keypoints = getattr(face, "kps", None)
    if keypoints is None or np.asarray(keypoints).shape[0] < 3:
        raise ValueError("Five-point face keypoints are required for the turn challenge.")
    points = np.asarray(keypoints, dtype="float32")
    left_eye, right_eye, nose = points[:3]
    eye_axis = right_eye - left_eye
    eye_distance = max(float(np.linalg.norm(eye_axis)), 1e-6)
    eye_midpoint = (left_eye + right_eye) / 2.0
    return float(np.dot(nose - eye_midpoint, eye_axis / eye_distance) / eye_distance)


def check_blink(
    landmarks: Sequence[np.ndarray],
    ear_drop: float = DEFAULT_SETTINGS.blink_ear_drop,
    recover_ratio: float = DEFAULT_SETTINGS.blink_recover_ratio,
) -> bool:
    """Check for a blink: EAR closes below baseline ratio and later recovers."""
    if len(landmarks) < 3 or any(points is None for points in landmarks):
        return False
    try:
        ratios = [eye_aspect_ratio_106(points) for points in landmarks]
    except (IndexError, TypeError, ValueError):
        return False
    baseline = ratios[0]
    if baseline <= 1e-6:
        return False
    closed_positions = [index for index, value in enumerate(ratios[1:], start=1) if value < baseline * ear_drop]
    return any(
        any(value >= baseline * recover_ratio for value in ratios[position + 1:])
        for position in closed_positions
    )


def check_turn(faces: Sequence[Any], action: str, minimum_delta: float) -> bool:
    """Check a turn in unmirrored image coordinates: subject-left moves nose image-right."""
    if len(faces) < 2 or action not in {"turn_left", "turn_right"}:
        return False
    values = [yaw_proxy(face) for face in faces]
    delta = max(values) - values[0] if action == "turn_left" else values[0] - min(values)
    return delta >= minimum_delta


def evaluate_challenge_actions(
    actions: Sequence[str],
    faces: Sequence[Any],
    frames_per_action: int = DEFAULT_SETTINGS.frames_per_action,
    settings: Settings = DEFAULT_SETTINGS,
) -> bool:
    """Evaluate actions over one baseline frame followed by fixed-size action bursts."""
    if not actions or len(faces) < 1 + len(actions) * frames_per_action:
        return False
    baseline = faces[0]
    offset = 1
    for action in actions:
        group = list(faces[offset:offset + frames_per_action])
        if len(group) != frames_per_action:
            return False
        if action == "blink":
            if not check_blink([getattr(baseline, "landmark_2d_106", None)] + [
                getattr(face, "landmark_2d_106", None) for face in group
            ], settings.blink_ear_drop, settings.blink_recover_ratio):
                return False
        elif not check_turn([baseline, *group], action, settings.challenge_yaw_delta):
            return False
        baseline = group[-1]
        offset += frames_per_action
    return True


def planar_motion_residual(first_face: Any, second_face: Any) -> float | None:
    """Return mean RANSAC landmark reprojection error normalized by inter-eye distance."""
    first = getattr(first_face, "landmark_2d_106", None)
    second = getattr(second_face, "landmark_2d_106", None)
    if first is None or second is None:
        return None
    source = np.asarray(first, dtype="float32")[:, :2]
    destination = np.asarray(second, dtype="float32")[:, :2]
    if source.shape != destination.shape or source.shape[0] < 4:
        return None
    try:
        transform, inliers = cv2.findHomography(source, destination, cv2.RANSAC, 3.0)
    except cv2.error:
        return None
    if transform is None or inliers is None or int(inliers.sum()) < 4:
        return None
    projected = cv2.perspectiveTransform(source.reshape(1, -1, 2), transform)[0]
    residuals = np.linalg.norm(projected - destination, axis=1)[inliers.ravel() == 1]
    keypoints = getattr(first_face, "kps", None)
    if keypoints is None or np.asarray(keypoints).shape[0] < 2:
        return None
    points = np.asarray(keypoints, dtype="float32")
    interocular = float(np.linalg.norm(points[0] - points[1]))
    if interocular <= 1e-6:
        return None
    return float(np.mean(residuals) / interocular)


class ChallengeManager:
    """Create and atomically consume randomized action challenges in FaceStore."""

    def __init__(
        self,
        store: Any,
        settings: Settings = DEFAULT_SETTINGS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.settings = settings
        self.clock = clock

    def issue(self, session_id: str, client_ip: str) -> dict[str, Any]:
        """Issue two distinct random actions bound to the session and IP."""
        actions = secrets.SystemRandom().sample(CHALLENGE_ACTIONS, 2)
        challenge_id = secrets.token_urlsafe(32)
        issued_at = self.clock()
        expires_at = issued_at + self.settings.challenge_ttl_sec
        self.store.create_challenge(
            challenge_id,
            session_id,
            client_ip_binding(client_ip, self.settings),
            actions,
            issued_at,
            expires_at,
        )
        return {
            "challenge_id": challenge_id,
            "actions": actions,
            "expires_in": self.settings.challenge_ttl_sec,
            "frames_per_action": self.settings.frames_per_action,
        }

    def consume(self, challenge_id: str, session_id: str, client_ip: str) -> dict[str, Any] | None:
        """Consume a challenge if it is bound, unused, and within allowed timing."""
        now = self.clock()
        challenge = self.store.consume_challenge(
            challenge_id,
            session_id,
            client_ip_binding(client_ip, self.settings),
            now,
        )
        if challenge is None:
            return None
        elapsed = now - challenge["issued_at"]
        if elapsed < self.settings.challenge_min_sec or elapsed > self.settings.challenge_ttl_sec:
            return None
        return challenge


def client_ip_binding(client_ip: str, settings: Settings = DEFAULT_SETTINGS) -> str:
    """Return a keyed IP binding so raw client addresses are not persisted."""
    key = settings.face_token_secret or settings.face_service_api_key or "development-only-ip-binding"
    return hmac.new(key.encode("utf-8"), client_ip.encode("utf-8"), hashlib.sha256).hexdigest()


def has_liveness_movement(
    frames: Sequence[tuple[np.ndarray, Any]],
    settings: Settings = DEFAULT_SETTINGS,
) -> bool:
    """Fallback movement check over caller-vetted, quality-passing frames only."""
    if len(frames) < 2:
        return False
    first_image, first_face = frames[0]
    second_image, second_face = frames[-1]
    first_landmarks = getattr(first_face, "landmark_2d_106", None)
    second_landmarks = getattr(second_face, "landmark_2d_106", None)
    if first_landmarks is not None and second_landmarks is not None:
        first_points = np.asarray(first_landmarks, dtype="float32")
        second_points = np.asarray(second_landmarks, dtype="float32")
        if first_points.shape == second_points.shape:
            moved = float(np.mean(np.linalg.norm(first_points - second_points, axis=1))) >= settings.liveness_landmark_movement
            if moved and settings.planar_check:
                measurable_turn = abs(yaw_proxy(second_face) - yaw_proxy(first_face)) >= settings.challenge_yaw_delta
                if measurable_turn:
                    residual = planar_motion_residual(first_face, second_face)
                    _logger.info("experimental_planar_residual=%s", residual)
                    return residual is not None and residual >= settings.planar_residual_min
            return moved
    first_gray = cv2.resize(cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY), (160, 120))
    second_gray = cv2.resize(cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY), (160, 120))
    return float(cv2.absdiff(first_gray, second_gray).mean()) >= settings.liveness_pixel_difference