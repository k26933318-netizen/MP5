"""Optional face liveness checks."""

from .blink_detector import BlinkLivenessDetector, LivenessResult
from .challenge import ChallengeLivenessDetector, ChallengeResult
from .config import (
    BlinkDetectorConfig,
    CLOSED_EAR_RATIO,
    DEFAULT_CONFIG,
    DEBUG_EAR,
    ENABLE_LIVENESS,
    LOW_QUALITY_SCORE_THRESHOLD,
    PREVIEW_INTERVAL_SECONDS,
    SHOW_DEBUG_INFO,
    CHECK_WHITE_BACKGROUND,
    WHITE_BACKGROUND_CHECK_INTERVAL_MS,
    WHITE_BACKGROUND_MAX_VARIATION,
    WHITE_BACKGROUND_MIN_BRIGHTNESS,
)

__all__ = [
    "BlinkDetectorConfig",
    "BlinkLivenessDetector",
    "ChallengeLivenessDetector",
    "ChallengeResult",
    "CLOSED_EAR_RATIO",
    "DEFAULT_CONFIG",
    "DEBUG_EAR",
    "ENABLE_LIVENESS",
    "LOW_QUALITY_SCORE_THRESHOLD",
    "LivenessResult",
    "PREVIEW_INTERVAL_SECONDS",
    "SHOW_DEBUG_INFO",
    "CHECK_WHITE_BACKGROUND",
    "WHITE_BACKGROUND_CHECK_INTERVAL_MS",
    "WHITE_BACKGROUND_MAX_VARIATION",
    "WHITE_BACKGROUND_MIN_BRIGHTNESS",
]
