"""Configuration for the live face shape overlay."""

from liveness.config import FACE_BOX_DETECTION_INTERVAL_MS


ENABLE_SHAPE_OVERLAY = True
TRACK_FACE_BOX = True
TRACK_MODE = "fast"  # Use "legacy" for the previous server-rendered tracer.
TRACER_STYLE = "modern"  # Use "classic" to restore the previous ellipse tracer.
FACE_BOX_SMOOTHING_FACTOR = 0.25
FACE_BOX_DEAD_ZONE_PX = 2
FACE_BOX_PADDING_PERCENT = 18.0
FACE_BOX_HOLD_MS = 400
FACE_BOX_FADE_MS = 180
TRACKING_CONFIG = {
    "mode": TRACK_MODE,
    "enabled": TRACK_FACE_BOX and ENABLE_SHAPE_OVERLAY,
    "detector_width": 320,
    "detection_interval_ms": FACE_BOX_DETECTION_INTERVAL_MS,
    "recognition_interval_frames": 0,
    "min_cutoff": 1.0,
    "beta": 0.05,
    "derivative_cutoff": 1.0,
    "max_prediction_ms": 140,
    "debug": False,
    "smoothing_factor": FACE_BOX_SMOOTHING_FACTOR,
    "dead_zone_px": FACE_BOX_DEAD_ZONE_PX,
    "padding_percent": FACE_BOX_PADDING_PERCENT,
    "hold_ms": FACE_BOX_HOLD_MS,
    "fade_ms": FACE_BOX_FADE_MS,
}
TRACER_CONFIG = {
    "style": TRACER_STYLE,
    "track_face_box": TRACKING_CONFIG["enabled"],
    "track_mode": TRACKING_CONFIG["mode"],
    "tracking": TRACKING_CONFIG,
    "face_box_smoothing_factor": TRACKING_CONFIG["smoothing_factor"],
    "face_box_dead_zone_px": TRACKING_CONFIG["dead_zone_px"],
    "face_box_padding_percent": TRACKING_CONFIG["padding_percent"],
    "face_box_hold_ms": TRACKING_CONFIG["hold_ms"],
    "face_box_fade_ms": TRACKING_CONFIG["fade_ms"],
    "colors": {
        "searching": "#f1f4f8",
        "hold": "#82b3df",
        "verified": "#7ebf9a",
        "failed": "#cf7b81",
    },
    "bracket_size": 0.18,
    "thickness": 2,
    "easing_speed": 0.35,
    "state_transition_ms": 200,
}
SHAPE_TYPE = "ellipse"
SHAPE_COLOR = (255, 216, 162)
SHAPE_THICKNESS = 2
SHAPE_PADDING = 12.0
SHAPE_FILLED = False
SHAPE_ALPHA = 0.25
SMOOTHING_FACTOR = 0.35
TRACK_TIMEOUT_SECONDS = 60.0
