"""Stateful eye-aspect-ratio blink liveness detection."""

from collections import deque
from dataclasses import dataclass, field
import math
import statistics
import time
from typing import Hashable

import cv2
import numpy as np

from .config import BlinkDetectorConfig, DEFAULT_CONFIG

_LEFT_EYE_106 = (63, 64, 65, 66, 68, 70)
_RIGHT_EYE_106 = (76, 77, 78, 79, 81, 83)
_LEFT_EYE_CONTOUR_106 = tuple(range(63, 72))
_RIGHT_EYE_CONTOUR_106 = tuple(range(76, 84))


@dataclass(frozen=True)
class LivenessResult:
    """Current liveness decision and blink progress for one tracked face."""

    is_live: bool
    blink_count: int
    ear: float
    status: str


@dataclass
class _FaceState:
    last_seen: float
    ears: deque[float]
    calibration_ears: list[float] = field(default_factory=list)
    baseline: float | None = None
    threshold: float | None = None
    blink_count: int = 0
    closed_at: float | None = None
    closed_frames: int = 0
    last_blink_at: float | None = None
    min_ear: float = math.inf
    max_ear: float = -math.inf
    last_landmarks: np.ndarray | None = None
    quality_since: float | None = None
    calibration_started_at: float | None = None
    restart_at: float | None = None
    quality_score: float = 1.0
    fps: float = 0.0
    last_result: LivenessResult | None = None


class BlinkLivenessDetector:
    """Calibrate and count natural close/reopen eye events independently per face.

    ``landmarks`` accepts two six-point eye contours, twelve flattened eye points,
    or InsightFace ``landmark_2d_106`` coordinates. Call :meth:`reset_missing`
    with the currently visible face IDs to discard tracks that have disappeared.
    """

    def __init__(self, config: BlinkDetectorConfig = DEFAULT_CONFIG) -> None:
        if config.consec_frames < 1 or config.required_blinks < 1 or config.smoothing_window < 1:
            raise ValueError("Frame counts and required blink count must be positive.")
        if config.time_window_seconds <= 0 or config.timeout_seconds <= 0:
            raise ValueError("Liveness timing limits must be positive.")
        if not 0.0 < config.closed_ear_ratio < 1.0:
            raise ValueError("The closed-eye EAR ratio must be between zero and one.")
        if not 0.0 < config.landmark_smoothing_factor <= 1.0:
            raise ValueError("The landmark smoothing factor must be between zero and one.")
        self.config = config
        self._states: dict[Hashable, _FaceState] = {}

    @staticmethod
    def eye_aspect_ratio(landmarks: np.ndarray) -> float:
        """Calculate mean EAR from six points per eye or InsightFace 106 points."""
        points = np.asarray(landmarks, dtype="float32")
        if points.ndim == 3 and points.shape[:2] == (2, 6) and points.shape[2] >= 2:
            eyes = (points[0, :, :2], points[1, :, :2])
        elif points.ndim == 2 and points.shape[1] >= 2 and points.shape[0] == 12:
            eyes = (points[:6, :2], points[6:, :2])
        elif points.ndim == 2 and points.shape[1] >= 2 and points.shape[0] >= 106:
            eyes = (points[list(_LEFT_EYE_106), :2], points[list(_RIGHT_EYE_106), :2])
        else:
            raise ValueError("Twelve eye landmarks or InsightFace 106-point landmarks are required.")

        ratios = []
        for eye in eyes:
            horizontal = float(np.linalg.norm(eye[0] - eye[3]))
            if horizontal <= 1e-6:
                raise ValueError("Eye landmarks have zero horizontal span.")
            vertical = float(np.linalg.norm(eye[1] - eye[5])) + float(
                np.linalg.norm(eye[2] - eye[4])
            )
            ratios.append(vertical / (2.0 * horizontal))
        ear = float(sum(ratios) / len(ratios))
        if not math.isfinite(ear):
            raise ValueError("Eye landmarks produced an invalid EAR.")
        return ear

    def update(
        self,
        frame: np.ndarray | None,
        landmarks: np.ndarray | None,
        face_id: Hashable = "default",
        timestamp: float | None = None,
    ) -> LivenessResult:
        """Update one face track from a frame and return its current blink status."""
        now = time.monotonic() if timestamp is None else timestamp
        if landmarks is None:
            self.reset(face_id)
            return LivenessResult(False, 0, 0.0, "NO FACE")
        try:
            points = np.asarray(landmarks, dtype="float32")
            ear = self.eye_aspect_ratio(points)
        except (TypeError, ValueError, IndexError):
            self.reset(face_id)
            return LivenessResult(False, 0, 0.0, "LANDMARKS UNAVAILABLE")

        state = self._states.get(face_id)
        if state is None or (state.restart_at is not None and now >= state.restart_at):
            state = self._new_state(now)
            self._states[face_id] = state

        if now < state.last_seen:
            self.reset(face_id)
            return LivenessResult(False, 0, ear, "TIMESTAMP RESET")
        frame_gap = now - state.last_seen
        if frame_gap > 0:
            state.fps = 1.0 / frame_gap
        if frame_gap > self.config.max_frame_gap_seconds:
            state.quality_since = None
            state.closed_at = None
            state.closed_frames = 0
            state.last_landmarks = None
            state.ears.clear()
            if state.baseline is None:
                state.calibration_started_at = None
                state.calibration_ears.clear()

        points = self._smooth_landmarks(state, points)
        ear = self.eye_aspect_ratio(points)
        quality_ok, quality_score = self._assess_quality(frame, points)
        state.last_seen = now
        state.quality_score = quality_score
        if not quality_ok:
            state.quality_since = None
            if state.baseline is None:
                state.calibration_started_at = None
                state.calibration_ears.clear()
            state.closed_at = None
            state.closed_frames = 0
            state.last_result = self._result(state, False, ear, "Low quality - improve lighting / move closer")
            return state.last_result

        if ear < self.config.ear_min_valid or ear > self.config.ear_max_valid:
            state.quality_since = None
            if state.baseline is None:
                state.calibration_started_at = None
                state.calibration_ears.clear()
            state.closed_at = None
            state.closed_frames = 0
            state.last_result = self._result(state, False, ear, "Verifying...")
            return state.last_result

        if state.quality_since is None:
            state.quality_since = now
        if state.baseline is None and state.calibration_started_at is None:
            state.calibration_started_at = now

        state.ears.append(ear)
        smooth_ear = float(statistics.median(state.ears))
        state.min_ear = min(state.min_ear, smooth_ear)
        state.max_ear = max(state.max_ear, smooth_ear)
        quality_elapsed = now - state.quality_since

        if state.baseline is None:
            state.calibration_ears.append(smooth_ear)
            calibration_started_at = (
                state.calibration_started_at
                if state.calibration_started_at is not None
                else now
            )
            calibration_elapsed = now - calibration_started_at
            if calibration_elapsed < self.config.calibration_seconds:
                state.last_result = self._result(state, False, smooth_ear, "Verifying...")
                return state.last_result
            state.baseline = self._calibrated_baseline(state.calibration_ears)
            if state.baseline is None:
                state.last_result = self._result(state, False, smooth_ear, "Verifying...")
                return state.last_result
            state.threshold = state.baseline * self.config.closed_ear_ratio

        threshold = state.threshold
        if threshold is None:
            state.last_result = self._result(state, False, smooth_ear, "Verifying...")
            return state.last_result
        if smooth_ear < threshold:
            if state.closed_at is None:
                state.closed_at = now
                state.closed_frames = 1
            else:
                state.closed_frames += 1
        elif state.closed_at is not None:
            closed_duration = now - state.closed_at
            varied = state.max_ear - state.min_ear >= self.config.min_ear_variation
            gap_ok = (
                state.last_blink_at is None
                or now - state.last_blink_at >= self.config.min_blink_interval_seconds
            )
            if (
                state.closed_frames >= self.config.consec_frames
                and self.config.min_blink_seconds <= closed_duration <= self.config.max_blink_seconds + 1e-6
                and varied
                and gap_ok
            ):
                state.blink_count += 1
                state.last_blink_at = now
            state.closed_at = None
            state.closed_frames = 0

        varied = state.max_ear - state.min_ear >= self.config.min_ear_variation
        is_live = state.blink_count >= self.config.required_blinks and varied
        if is_live:
            status = "LIVE"
        elif quality_elapsed >= max(
            self.config.time_window_seconds,
            self.config.timeout_seconds,
        ):
            status = "Could not verify" if varied else "SPOOF SUSPECTED"
            if state.restart_at is None:
                state.restart_at = now + self.config.auto_restart_delay_seconds
        else:
            status = "Verifying..."
        state.last_result = self._result(state, is_live, smooth_ear, status)
        return state.last_result

    def current_result(
        self,
        face_id: Hashable = "default",
        timestamp: float | None = None,
    ) -> LivenessResult:
        """Return the last result without calling a stale or low-quality face a spoof."""
        state = self._states.get(face_id)
        if state is None or state.last_result is None:
            return LivenessResult(False, 0, 0.0, "Verifying...")
        now = time.monotonic() if timestamp is None else timestamp
        if now - state.last_seen > self.config.timeout_seconds:
            return self._result(state, False, state.last_result.ear, "Could not verify")
        return state.last_result

    def debug_info(self, face_id: Hashable = "default") -> dict[str, float | None]:
        """Return scalar diagnostics without expanding the public result constructor."""
        state = self._states.get(face_id)
        if state is None:
            return {"baseline": None, "threshold": None, "quality_score": 0.0, "fps": 0.0}
        return {
            "baseline": state.baseline,
            "threshold": state.threshold,
            "quality_score": state.quality_score,
            "fps": state.fps,
        }

    def reset(self, face_id: Hashable = "default") -> None:
        """Discard all calibration and blink history for one face."""
        self._states.pop(face_id, None)

    def reset_missing(self, active_face_ids: set[Hashable]) -> None:
        """Discard any face track not present in the latest frame."""
        for face_id in set(self._states) - active_face_ids:
            self.reset(face_id)

    def _new_state(self, now: float) -> _FaceState:
        return _FaceState(
            last_seen=now,
            ears=deque(maxlen=self.config.smoothing_window),
        )

    def _smooth_landmarks(self, state: _FaceState, points: np.ndarray) -> np.ndarray:
        if state.last_landmarks is None or state.last_landmarks.shape != points.shape:
            smoothed = points.copy()
        else:
            factor = self.config.landmark_smoothing_factor
            smoothed = state.last_landmarks + (points - state.last_landmarks) * factor
        state.last_landmarks = smoothed
        return smoothed

    @staticmethod
    def _calibrated_baseline(values: list[float]) -> float | None:
        if not values:
            return None
        sorted_values = sorted(values, reverse=True)
        upper_count = max(1, math.ceil(len(sorted_values) * 0.7))
        baseline = float(statistics.median(sorted_values[:upper_count]))
        return baseline if math.isfinite(baseline) and baseline > 0 else None

    def _assess_quality(self, frame: np.ndarray | None, points: np.ndarray) -> tuple[bool, float]:
        if frame is None or points.shape[0] < 106 or frame.size == 0:
            return True, 1.0
        height, width = frame.shape[:2]
        facial_points = points[:, :2]
        if not np.isfinite(facial_points).all():
            return False, 0.0
        left = float(np.min(facial_points[:, 0]))
        top = float(np.min(facial_points[:, 1]))
        right = float(np.max(facial_points[:, 0]))
        bottom = float(np.max(facial_points[:, 1]))
        face_width = right - left
        face_height = bottom - top
        size_score = min(1.0, min(face_width, face_height) / max(self.config.min_face_size_px, 1))
        x1 = max(0, min(width, int(math.floor(left))))
        y1 = max(0, min(height, int(math.floor(top))))
        x2 = max(0, min(width, int(math.ceil(right))))
        y2 = max(0, min(height, int(math.ceil(bottom))))
        if min(face_width, face_height) < self.config.min_face_size_px or x2 <= x1 or y2 <= y1:
            return False, float(size_score)

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        brightness = float(gray.mean())
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_score = min(1.0, brightness / max(self.config.min_brightness, 1.0))
        blur_score = min(1.0, blur_variance / max(self.config.min_blur_variance, 1.0))

        left_eye = np.mean(points[list(_LEFT_EYE_CONTOUR_106)], axis=0)
        right_eye = np.mean(points[list(_RIGHT_EYE_CONTOUR_106)], axis=0)
        nose = np.mean(points[51:63], axis=0)
        eye_axis = right_eye - left_eye
        eye_distance = max(float(np.linalg.norm(eye_axis)), 1e-6)
        eye_midpoint = (left_eye + right_eye) / 2.0
        yaw_ratio = abs(float(np.dot(nose - eye_midpoint, eye_axis / eye_distance))) / eye_distance
        pose_score = min(1.0, self.config.max_yaw_ratio / max(yaw_ratio, 1e-6))
        score = float(np.mean((size_score, brightness_score, blur_score, pose_score)))
        quality_ok = (
            brightness >= self.config.min_brightness
            and blur_variance >= self.config.min_blur_variance
            and yaw_ratio <= self.config.max_yaw_ratio
        )
        return quality_ok, score

    @staticmethod
    def _result(state: _FaceState, is_live: bool, ear: float, status: str) -> LivenessResult:
        return LivenessResult(is_live, state.blink_count, ear, status)
