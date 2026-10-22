"""Configuration for the optional eye-blink liveness gate."""

from dataclasses import dataclass
import os


# --- speed-up (added) ---
FAST_MODE = True
PERFORMANCE_DEBUG = True
PERFORMANCE_REPORT_INTERVAL_SECONDS = 3.0
DETECTOR_MAX_WIDTH = 480 if FAST_MODE else 640
DETECT_EVERY_N_FRAMES = 2 if FAST_MODE else 1
RECOGNITION_FRAME_COUNT = 3 if FAST_MODE else 7
FAST_CAPTURE_GAP_MS = 100 if FAST_MODE else 700
FACE_BOX_DETECTION_INTERVAL_MS = 200 if FAST_MODE else 100

EAR_THRESHOLD = 0.21
CLOSED_EAR_RATIO = 0.72
CONSEC_FRAMES = 1
REQUIRED_BLINKS = 1
TIME_WINDOW_SECONDS = 9.0
TIMEOUT_SECONDS = 9.0
CALIBRATION_SECONDS = 1.0 if FAST_MODE else 1.5
SMOOTHING_WINDOW = 3
MIN_BLINK_SECONDS = 0.08
MAX_BLINK_SECONDS = 0.4
MIN_EAR_VARIATION = 0.025
MIN_BLINK_INTERVAL_SECONDS = 0.25
PREVIEW_INTERVAL_SECONDS = 0.1
MIN_FACE_SIZE_PX = 60
MIN_BRIGHTNESS = 35.0
MIN_BLUR_VARIANCE = 5.0
MAX_YAW_RATIO = 0.4
EAR_MIN_VALID = 0.05
EAR_MAX_VALID = 0.6
LANDMARK_SMOOTHING_FACTOR = 0.8
LANDMARK_CROP_PADDING_RATIO = 0.2
LANDMARK_MIN_SIZE_PX = 200
LANDMARK_GAMMA_THRESHOLD = 90.0
LANDMARK_GAMMA = 0.8
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = 4
AUTO_RESTART_DELAY_SECONDS = 0.75
MAX_FRAME_GAP_SECONDS = 0.5
CHALLENGE_ORDER = ("blink", "head_turn", "mouth_open")
CHALLENGE_SUCCESS_COUNT = 1 if FAST_MODE else 2
BLINK_ACTION_REQUIRED_COUNT = 1 if FAST_MODE else 2
CHALLENGE_TIMEOUT_SECONDS = 4.0 if FAST_MODE else 10.0
CHALLENGE_MAX_RETRIES = 2
HEAD_TURN_MIN_DELTA = 0.06
MOUTH_OPEN_MIN_DELTA = 0.08
METRIC_OUTLIER_LIMIT = 0.2
QUALITY_MIN_BRIGHTNESS = 25.0
QUALITY_MIN_BLUR_VARIANCE = 3.0
MOUTH_INNER_LANDMARKS = tuple(range(96, 104))
SHOW_DEBUG_INFO = False
LOW_QUALITY_SCORE_THRESHOLD = 0.5
CHECK_WHITE_BACKGROUND = False
WHITE_BACKGROUND_CHECK_INTERVAL_MS = 500
WHITE_BACKGROUND_MIN_BRIGHTNESS = 200.0
WHITE_BACKGROUND_MAX_VARIATION = 35.0


def _environment_flag(name: str, default: bool) -> bool:
    """Read a conventional boolean environment value."""
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _environment_float(name: str, default: float) -> float:
    """Read and validate a floating-point environment override."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number.") from error
    if not 0.0 < parsed < 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return parsed


ENABLE_LIVENESS = _environment_flag("ENABLE_LIVENESS", False)
DEBUG_EAR = _environment_flag("LIVENESS_DEBUG_EAR", False)
CLOSED_EAR_RATIO = _environment_float("LIVENESS_CLOSED_EAR_RATIO", CLOSED_EAR_RATIO)


@dataclass(frozen=True)
class BlinkDetectorConfig:
    """Thresholds and timing limits used by :class:`BlinkLivenessDetector`."""

    ear_threshold: float = EAR_THRESHOLD
    closed_ear_ratio: float = CLOSED_EAR_RATIO
    consec_frames: int = CONSEC_FRAMES
    required_blinks: int = REQUIRED_BLINKS
    time_window_seconds: float = TIME_WINDOW_SECONDS
    timeout_seconds: float = TIMEOUT_SECONDS
    calibration_seconds: float = CALIBRATION_SECONDS
    smoothing_window: int = SMOOTHING_WINDOW
    min_blink_seconds: float = MIN_BLINK_SECONDS
    max_blink_seconds: float = MAX_BLINK_SECONDS
    min_ear_variation: float = MIN_EAR_VARIATION
    min_blink_interval_seconds: float = MIN_BLINK_INTERVAL_SECONDS
    min_face_size_px: int = MIN_FACE_SIZE_PX
    min_brightness: float = MIN_BRIGHTNESS
    min_blur_variance: float = MIN_BLUR_VARIANCE
    max_yaw_ratio: float = MAX_YAW_RATIO
    ear_min_valid: float = EAR_MIN_VALID
    ear_max_valid: float = EAR_MAX_VALID
    landmark_smoothing_factor: float = LANDMARK_SMOOTHING_FACTOR
    landmark_crop_padding_ratio: float = LANDMARK_CROP_PADDING_RATIO
    landmark_min_size_px: int = LANDMARK_MIN_SIZE_PX
    landmark_gamma_threshold: float = LANDMARK_GAMMA_THRESHOLD
    landmark_gamma: float = LANDMARK_GAMMA
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT
    clahe_grid_size: int = CLAHE_GRID_SIZE
    auto_restart_delay_seconds: float = AUTO_RESTART_DELAY_SECONDS
    max_frame_gap_seconds: float = MAX_FRAME_GAP_SECONDS
    challenge_success_count: int = CHALLENGE_SUCCESS_COUNT
    blink_action_required_count: int = BLINK_ACTION_REQUIRED_COUNT
    challenge_timeout_seconds: float = CHALLENGE_TIMEOUT_SECONDS
    challenge_max_retries: int = CHALLENGE_MAX_RETRIES
    head_turn_min_delta: float = HEAD_TURN_MIN_DELTA
    mouth_open_min_delta: float = MOUTH_OPEN_MIN_DELTA
    metric_outlier_limit: float = METRIC_OUTLIER_LIMIT
    quality_min_brightness: float = QUALITY_MIN_BRIGHTNESS
    quality_min_blur_variance: float = QUALITY_MIN_BLUR_VARIANCE


DEFAULT_CONFIG = BlinkDetectorConfig()
