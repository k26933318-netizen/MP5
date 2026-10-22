"""Low-cost, timestamp-driven facial action challenges for InsightFace landmarks."""

from collections import deque
from dataclasses import dataclass, field
import math
import logging
import random
import statistics
import time
from typing import Callable, Hashable, Sequence

import cv2
import numpy as np

from .blink_detector import BlinkLivenessDetector
from .config import CHALLENGE_ORDER, FAST_MODE, MOUTH_INNER_LANDMARKS, BlinkDetectorConfig, DEFAULT_CONFIG
from .blink_detector import LivenessResult

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChallengeResult(LivenessResult):
    """Current state, short instruction, progress, and compatibility metrics."""

    state: str
    instruction: str
    progress: int
    required_progress: int
    baseline: float | None = None
    threshold: float | None = None
    yaw: float = 0.0
    mouth: float = 0.0
    quality_score: float = 0.0
    fps: float = 0.0
    retries: int = 0


@dataclass
class _FaceState:
    last_seen: float
    metrics: deque[tuple[float, float, float]]
    calibration: list[tuple[float, float, float]] = field(default_factory=list)
    calibration_elapsed: float = 0.0
    baseline_ear: float | None = None
    baseline_yaw: float | None = None
    baseline_mouth: float | None = None
    action_order: list[str] = field(default_factory=list)
    action_index: int = 0
    passed: set[str] = field(default_factory=set)
    retries: int = 0
    action_elapsed: float = 0.0
    closed_at: float | None = None
    blink_count: int = 0
    last_blink_at: float | None = None
    quality_score: float = 0.0
    fps: float = 0.0
    ear: float = 0.0
    yaw: float = 0.0
    mouth: float = 0.0
    result: ChallengeResult | None = None


class ChallengeLivenessDetector:
    """Run a randomized 2-of-3 challenge using existing 106-point landmarks."""

    _FACE_EDGES = slice(0, 33)
    _NOSE = slice(51, 63)

    def __init__(
        self,
        config: BlinkDetectorConfig = DEFAULT_CONFIG,
        action_picker: Callable[[Sequence[str]], list[str]] | None = None,
    ) -> None:
        if not 1 <= config.challenge_success_count <= len(CHALLENGE_ORDER):
            raise ValueError("The required challenge count must fit the configured actions.")
        if config.challenge_timeout_seconds <= 0 or config.challenge_max_retries < 0:
            raise ValueError("Challenge timing and retry settings are invalid.")
        self.config = config
        self._action_picker = action_picker or self._random_order
        self._states: dict[Hashable, _FaceState] = {}

    @staticmethod
    def _random_order(actions: Sequence[str]) -> list[str]:
        result = list(actions)
        random.SystemRandom().shuffle(result)
        if "head_turn" in result:
            result[result.index("head_turn")] = random.SystemRandom().choice(
                ("head_turn_left", "head_turn_right")
            )
        return result

    @staticmethod
    def _metrics(points: np.ndarray) -> tuple[float, float, float]:
        ear = BlinkLivenessDetector.eye_aspect_ratio(points)
        face_edges = points[ChallengeLivenessDetector._FACE_EDGES, :2]
        nose = points[ChallengeLivenessDetector._NOSE, :2]
        mouth = points[list(MOUTH_INNER_LANDMARKS), :2]
        face_width = float(np.ptp(face_edges[:, 0]))
        mouth_width = float(np.ptp(mouth[:, 0]))
        if face_width <= 1e-6 or mouth_width <= 1e-6:
            raise ValueError("Face landmarks have no measurable width.")
        face_midpoint = float((np.min(face_edges[:, 0]) + np.max(face_edges[:, 0])) / 2.0)
        yaw = (float(np.mean(nose[:, 0])) - face_midpoint) / face_width
        mouth_ratio = float(np.ptp(mouth[:, 1])) / mouth_width
        return ear, yaw, mouth_ratio

    def update(
        self,
        frame: np.ndarray | None,
        landmarks: np.ndarray | None,
        face_id: Hashable = "default",
        timestamp: float | None = None,
    ) -> ChallengeResult:
        """Advance one face using timestamps; low-quality frames pause progress."""
        now = time.monotonic() if timestamp is None else timestamp
        if landmarks is None:
            self.reset(face_id)
            return self._empty("checking", "Center your face")
        try:
            points = np.asarray(landmarks, dtype="float32")
            if points.ndim != 2 or points.shape[0] < 106 or points.shape[1] < 2:
                raise ValueError("InsightFace 106-point landmarks are required.")
            if not np.isfinite(points[:, :2]).all():
                raise ValueError("Landmarks contain non-finite coordinates.")
            raw_ear, _, _ = self._metrics(points)
        except (TypeError, ValueError, IndexError):
            state = self._state(face_id, now)
            return self._result(state, "low_quality", "Move closer", status="LANDMARKS UNAVAILABLE")

        state = self._state(face_id, now)
        if now < state.last_seen:
            self.reset(face_id)
            state = self._state(face_id, now)
        gap = max(0.0, now - state.last_seen)
        if gap > 0:
            state.fps = 1.0 / gap
        if gap > self.config.max_frame_gap_seconds:
            state.metrics.clear()
            state.closed_at = None
            if state.baseline_ear is None:
                state.calibration.clear()
                state.calibration_elapsed = 0.0

        quality, quality_score, quality_message = self._assess_quality(frame, points)
        state.last_seen = now
        state.quality_score = quality_score
        if raw_ear < self.config.ear_min_valid or raw_ear > self.config.ear_max_valid:
            return self._low_quality(state, "Move closer")
        if not quality:
            return self._low_quality(state, quality_message)

        measured = self._metrics(points)
        if state.metrics:
            previous = tuple(float(statistics.median(values)) for values in zip(*state.metrics))
            if any(abs(value - median) > self.config.metric_outlier_limit for value, median in zip(measured, previous)):
                return self._result(state, "checking", self._instruction(state))
        state.metrics.append(measured)
        smooth_ear, smooth_yaw, smooth_mouth = (
            float(statistics.median(values)) for values in zip(*state.metrics)
        )
        state.ear = smooth_ear
        state.yaw = smooth_yaw
        state.mouth = smooth_mouth

        if state.baseline_ear is None:
            state.calibration.append((smooth_ear, smooth_yaw, smooth_mouth))
            state.calibration_elapsed += min(gap, self.config.max_frame_gap_seconds)
            if state.calibration_elapsed < self.config.calibration_seconds:
                return self._result(state, "checking", "Look at the camera")
            state.baseline_ear = self._upper_median([item[0] for item in state.calibration])
            state.baseline_yaw = float(statistics.median(item[1] for item in state.calibration))
            state.baseline_mouth = float(statistics.median(item[2] for item in state.calibration))
            state.action_order = self._action_picker(CHALLENGE_ORDER)
            return self._result(state, "checking", self._instruction(state))

        if state.result and state.result.state in {"live", "failed"}:
            return state.result

        state.action_elapsed += gap
        if self._action_passed(state, raw_ear, now):
            state.passed.add(state.action_order[state.action_index])
            state.action_index += 1
            state.action_elapsed = 0.0
            state.closed_at = None
            if len(state.passed) >= self.config.challenge_success_count:
                return self._result(state, "live", "Verified")
            return self._result(state, "checking", self._instruction(state))

        if state.action_elapsed >= self.config.challenge_timeout_seconds:
            state.action_index += 1
            state.action_elapsed = 0.0
            state.closed_at = None
            if state.action_index >= len(state.action_order):
                if state.retries >= self.config.challenge_max_retries:
                    return self._result(state, "failed", "Try again")
                state.retries += 1
                state.action_order = self._action_picker(CHALLENGE_ORDER)
                state.action_index = 0
                state.passed.clear()
                state.blink_count = 0
            return self._result(state, "checking", self._instruction(state))
        return self._result(state, "checking", self._instruction(state))

    def current_result(self, face_id: Hashable = "default") -> ChallengeResult:
        """Return the last state for a face, or an initial checking result."""
        state = self._states.get(face_id)
        return state.result if state and state.result else self._empty("checking", "Center your face")

    def debug_info(self, face_id: Hashable = "default") -> dict[str, float | None]:
        """Return numeric values for a separate opt-in debug panel."""
        state = self._states.get(face_id)
        if state is None:
            return {"baseline": None, "threshold": None, "quality_score": 0.0, "fps": 0.0, "yaw": 0.0, "mouth": 0.0}
        return {
            "baseline": state.baseline_ear,
            "threshold": state.baseline_ear * self.config.closed_ear_ratio if state.baseline_ear else None,
            "quality_score": state.quality_score,
            "fps": state.fps,
            "yaw": state.yaw,
            "mouth": state.mouth,
        }

    def reset(self, face_id: Hashable = "default") -> None:
        """Discard all challenge progress and calibration for one face."""
        self._states.pop(face_id, None)

    def reset_missing(self, active_face_ids: set[Hashable]) -> None:
        """Discard tracks for faces no longer present in the camera view."""
        for face_id in set(self._states) - active_face_ids:
            self.reset(face_id)

    def _state(self, face_id: Hashable, now: float) -> _FaceState:
        state = self._states.get(face_id)
        if state is None:
            state = _FaceState(now, deque(maxlen=self.config.smoothing_window))
            self._states[face_id] = state
        return state

    def _action_passed(self, state: _FaceState, raw_ear: float, now: float) -> bool:
        action = state.action_order[state.action_index]
        blink_action = action == "blink" or FAST_MODE
        if blink_action:
            baseline = state.baseline_ear or 0.0
            if raw_ear < baseline * self.config.closed_ear_ratio:
                if state.closed_at is None:
                    state.closed_at = now
            elif state.closed_at is not None:
                duration = now - state.closed_at
                interval_ok = state.last_blink_at is None or now - state.last_blink_at >= self.config.min_blink_interval_seconds
                if self.config.min_blink_seconds <= duration <= self.config.max_blink_seconds and interval_ok:
                    state.blink_count += 1
                    state.last_blink_at = now
                state.closed_at = None
            blink_passed = state.blink_count >= self.config.blink_action_required_count
            if FAST_MODE:
                turned = abs(state.yaw - (state.baseline_yaw or 0.0)) >= self.config.head_turn_min_delta
                mouth_open = state.mouth - (state.baseline_mouth or 0.0) >= self.config.mouth_open_min_delta
                return blink_passed or turned or mouth_open
            return blink_passed
        if action == "head_turn_left":
            return state.yaw - (state.baseline_yaw or 0.0) >= self.config.head_turn_min_delta
        if action == "head_turn_right":
            return (state.baseline_yaw or 0.0) - state.yaw >= self.config.head_turn_min_delta
        return state.mouth - (state.baseline_mouth or 0.0) >= self.config.mouth_open_min_delta

    def _instruction(self, state: _FaceState) -> str:
        if not state.action_order:
            return "Look at the camera"
        action = state.action_order[min(state.action_index, len(state.action_order) - 1)]
        return {
            "blink": "Blink twice",
            "head_turn_left": "Turn your head slightly left",
            "head_turn_right": "Turn your head slightly right",
            "mouth_open": "Open your mouth",
        }[action]

    def _assess_quality(self, frame: np.ndarray | None, points: np.ndarray) -> tuple[bool, float, str]:
        face_points = points[:, :2]
        face_width = float(np.ptp(face_points[:, 0]))
        face_height = float(np.ptp(face_points[:, 1]))
        if min(face_width, face_height) < self.config.min_face_size_px:
            return False, min(1.0, min(face_width, face_height) / self.config.min_face_size_px), "Move closer"
        if frame is None or frame.size == 0:
            return True, 1.0, ""
        height, width = frame.shape[:2]
        x1 = max(0, int(np.floor(np.min(face_points[:, 0]))))
        y1 = max(0, int(np.floor(np.min(face_points[:, 1]))))
        x2 = min(width, int(np.ceil(np.max(face_points[:, 0]))) )
        y2 = min(height, int(np.ceil(np.max(face_points[:, 1]))) )
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, 0.0, "Move closer"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        score = min(1.0, brightness / max(self.config.quality_min_brightness, 1.0), blur / max(self.config.quality_min_blur_variance, 1.0))
        if brightness < self.config.quality_min_brightness:
            return False, score, "More light needed"
        if blur < self.config.quality_min_blur_variance:
            return False, score, "Try again"
        return True, score, ""

    @staticmethod
    def _upper_median(values: Sequence[float]) -> float:
        ordered = sorted(values, reverse=True)
        return float(statistics.median(ordered[:max(1, math.ceil(len(ordered) * 0.7))]))

    def _low_quality(self, state: _FaceState, instruction: str) -> ChallengeResult:
        state.closed_at = None
        return self._result(state, "low_quality", instruction)

    def _result(
        self,
        state: _FaceState,
        state_name: str,
        instruction: str,
        status: str | None = None,
    ) -> ChallengeResult:
        if state_name in {"live", "failed"} and (state.result is None or state.result.state != state_name):
            _logger.info(
                "liveness challenge completed state=%s progress=%d retries=%d",
                state_name,
                len(state.passed),
                state.retries,
            )
        baseline = state.baseline_ear
        result = ChallengeResult(
            is_live=state_name == "live",
            blink_count=state.blink_count,
            ear=state.ear,
            status=status or state_name,
            state=state_name,
            instruction=instruction,
            progress=len(state.passed),
            required_progress=self.config.challenge_success_count,
            baseline=baseline,
            threshold=baseline * self.config.closed_ear_ratio if baseline else None,
            yaw=state.yaw,
            mouth=state.mouth,
            quality_score=state.quality_score,
            fps=state.fps,
            retries=state.retries,
        )
        state.result = result
        return result

    @staticmethod
    def _empty(state: str, instruction: str) -> ChallengeResult:
        return ChallengeResult(
            False,
            0,
            0.0,
            state,
            state,
            instruction,
            0,
            DEFAULT_CONFIG.challenge_success_count,
        )