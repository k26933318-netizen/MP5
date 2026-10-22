"""Environment-backed settings for the face verification service."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


def _limit_blas_threads() -> None:
    """Keep OpenBLAS/MKL thread counts low on constrained machines."""
    if any(name in os.environ for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        return
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def _load_local_dotenv() -> None:
    """Load the repo-local .env without requiring python-dotenv at runtime."""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if not candidate.is_file():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_dotenv()
_limit_blas_threads()


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    return float(env.get(name, default))


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    return int(env.get(name, default))


@dataclass(frozen=True)
class Settings:
    """Typed service configuration. Thresholds are initial values; tune with tools/."""

    face_profile: str = "default"
    face_model: str = "buffalo_l"
    face_det_size: int = 640
    face_data_dir: Path = Path("face_data")
    face_db_key: str = ""
    allow_model_mismatch: bool = False
    max_image_pixels: int = 25_000_000
    max_image_side: int = 1600
    max_content_length: int = 16 * 1024 * 1024
    pose_gate: bool = False
    use_tta: bool = False
    enroll_max_frames: int = 12
    enroll_min_frames: int = 5
    enroll_keep: int = 6
    verify_max_frames: int = 8
    verify_keep: int = 3
    vote_min_frames: int = 3
    vote_min_agree: int = 2
    enroll_min_face_px: int = 150
    verify_min_face_px: int = 100
    face_min_detection_score: float = 0.7
    second_face_max_ratio: float = 0.3
    blur_threshold: float = 100.0
    brightness_min: float = 60.0
    brightness_max: float = 200.0
    contrast_min: float = 25.0
    pose_yaw_max: float = 30.0
    pose_pitch_max: float = 25.0
    pose_roll_max: float = 30.0
    eyes_open_gate: bool = False
    eyes_open_min_ear: float = 0.16
    same_person_min: float = 0.5
    outlier_min_sim: float = 0.4
    verify_threshold: float = 0.5
    fusion_alpha: float = 0.5
    duplicate_threshold: float = 0.6
    impersonation_margin: float = 0.05
    max_attempts: int = 3
    antispoof_model_a_path: str = ""
    antispoof_model_b_path: str = ""
    antispoof_threshold: float = 0.7
    antispoof_min_frame: float = 0.5
    require_antispoof: bool = True
    require_challenge: bool = True
    require_enroll_challenge: bool = False
    challenge_ttl_sec: int = 20
    challenge_min_sec: float = 1.5
    frames_per_action: int = 3
    blink_ear_drop: float = 0.6
    blink_recover_ratio: float = 0.85
    challenge_yaw_delta: float = 0.15
    planar_check: bool = False
    planar_residual_min: float = 0.02
    session_ttl_sec: int = 300
    session_max_submits: int = 3
    token_ttl_sec: int = 120
    face_service_api_key: str = ""
    auth_disabled_for_dev: bool = False
    face_token_secret: str = ""
    kiosk_mode: bool = False
    kiosk_allowed_ips: tuple[str, ...] = ()
    face_cors_origins: tuple[str, ...] = ()
    host: str = "127.0.0.1"
    port: int = 5000
    waitress_threads: int = 4
    flask_debug: bool = False
    preview_interval_sec: float = 0.275
    rate_limit_ip_per_minute: int = 120
    rate_limit_subject_per_minute: int = 30
    attempt_backoff_base_sec: float = 0.1
    https_terminated: bool = False
    db_busy_retries: int = 3
    db_retry_delay_sec: float = 0.05
    detect_min_score: float = 0.5
    detect_min_width_ratio: float = 0.12
    detect_min_height_ratio: float = 0.12
    detect_min_area_ratio: float = 0.03
    detect_center_x_min: float = 0.2
    detect_center_x_max: float = 0.8
    detect_center_y_min: float = 0.15
    detect_center_y_max: float = 0.85
    liveness_landmark_movement: float = 2.0
    liveness_pixel_difference: float = 2.0

    @property
    def match_threshold(self) -> float:
        """Legacy alias for verify_threshold."""
        return self.verify_threshold

    @property
    def liveness_identity_min(self) -> float:
        """Legacy alias for same_person_min."""
        return self.same_person_min

    @property
    def max_register_images(self) -> int:
        """Legacy alias for enroll_max_frames."""
        return self.enroll_max_frames

    @property
    def max_verify_images(self) -> int:
        """Legacy alias for verify_max_frames."""
        return self.verify_max_frames

    @property
    def face_min_size(self) -> int:
        """Legacy alias for verify_min_face_px."""
        return self.verify_min_face_px


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Load settings from an environment mapping, useful for deterministic tests."""
    if environ is None:
        _load_local_dotenv()
        _limit_blas_threads()
        env = os.environ
    else:
        env = environ
    face_profile = env.get("FACE_PROFILE", "default").strip().lower()
    if face_profile not in {"default", "max_accuracy"}:
        raise ValueError("FACE_PROFILE must be 'default' or 'max_accuracy'.")
    max_accuracy = face_profile == "max_accuracy"
    verify_threshold = _float(
        env,
        "VERIFY_THRESHOLD",
        _float(env, "MATCH_THRESHOLD", 0.5),
    )
    same_person_min = _float(
        env,
        "SAME_PERSON_MIN",
        _float(env, "LIVENESS_IDENTITY_MIN", 0.5),
    )
    enroll_max_frames = _int(
        env,
        "ENROLL_MAX_FRAMES",
        _int(env, "MAX_REGISTER_IMAGES", 12),
    )
    verify_max_frames = int(
        env.get("VERIFY_MAX_FRAMES", env.get("MAX_FRAMES_VERIFY", env.get("MAX_VERIFY_IMAGES", 8)))
    )
    verify_min_face_px = _int(
        env,
        "VERIFY_MIN_FACE_PX",
        _int(env, "FACE_MIN_SIZE", 100),
    )
    return Settings(
        face_profile=face_profile,
        face_model=env.get("FACE_MODEL", "buffalo_l"),
        face_det_size=_int(env, "FACE_DET_SIZE", 640),
        face_data_dir=Path(env.get("FACE_DATA_DIR", "face_data")),
        face_db_key=env.get("FACE_DB_KEY", ""),
        allow_model_mismatch=_bool(env.get("ALLOW_MODEL_MISMATCH"), False),
        max_image_pixels=_int(env, "MAX_IMAGE_PIXELS", 25_000_000),
        max_image_side=_int(env, "MAX_IMAGE_SIDE", 1600),
        max_content_length=_int(env, "MAX_CONTENT_LENGTH", 16 * 1024 * 1024),
        pose_gate=_bool(env.get("POSE_GATE"), max_accuracy),
        use_tta=_bool(env.get("USE_TTA"), max_accuracy),
        enroll_max_frames=enroll_max_frames,
        enroll_min_frames=_int(env, "ENROLL_MIN_FRAMES", 5),
        enroll_keep=_int(env, "ENROLL_KEEP", 6),
        verify_max_frames=int(
            env.get(
                "VERIFY_MAX_FRAMES",
                env.get("MAX_FRAMES_VERIFY", env.get("MAX_VERIFY_IMAGES", 5 if max_accuracy else verify_max_frames)),
            )
        ),
        verify_keep=_int(env, "VERIFY_KEEP", 3),
        vote_min_frames=_int(env, "VOTE_MIN_FRAMES", 3),
        vote_min_agree=_int(env, "VOTE_MIN_AGREE", 3 if max_accuracy else 2),
        enroll_min_face_px=_int(env, "ENROLL_MIN_FACE_PX", 150),
        verify_min_face_px=_int(
            env,
            "VERIFY_MIN_FACE_PX",
            150 if max_accuracy and "FACE_MIN_SIZE" not in env else verify_min_face_px,
        ),
        face_min_detection_score=_float(env, "FACE_MIN_DETECTION_SCORE", 0.7),
        second_face_max_ratio=_float(env, "SECOND_FACE_MAX_RATIO", 0.3),
        blur_threshold=_float(env, "BLUR_THRESHOLD", 100.0),
        brightness_min=_float(env, "BRIGHTNESS_MIN", 60.0),
        brightness_max=_float(env, "BRIGHTNESS_MAX", 200.0),
        contrast_min=_float(env, "CONTRAST_MIN", 25.0),
        pose_yaw_max=_float(env, "POSE_YAW_MAX", 30.0),
        pose_pitch_max=_float(env, "POSE_PITCH_MAX", 25.0),
        pose_roll_max=_float(env, "POSE_ROLL_MAX", 30.0),
        eyes_open_gate=_bool(env.get("EYES_OPEN_GATE"), max_accuracy),
        eyes_open_min_ear=_float(env, "EYES_OPEN_MIN_EAR", 0.16),
        same_person_min=same_person_min,
        outlier_min_sim=_float(env, "OUTLIER_MIN_SIM", 0.4),
        verify_threshold=verify_threshold,
        fusion_alpha=_float(env, "FUSION_ALPHA", 0.5),
        duplicate_threshold=_float(env, "DUPLICATE_THRESHOLD", 0.6),
        impersonation_margin=_float(env, "IMPERSONATION_MARGIN", 0.08 if max_accuracy else 0.05),
        max_attempts=_int(env, "MAX_ATTEMPTS", 3),
        antispoof_model_a_path=env.get("ANTISPOOF_MODEL_A_PATH", ""),
        antispoof_model_b_path=env.get("ANTISPOOF_MODEL_B_PATH", ""),
        antispoof_threshold=_float(env, "ANTISPOOF_THRESHOLD", 0.7),
        antispoof_min_frame=_float(env, "ANTISPOOF_MIN_FRAME", 0.5),
        require_antispoof=_bool(env.get("REQUIRE_ANTISPOOF"), True),
        require_challenge=_bool(env.get("REQUIRE_CHALLENGE"), True),
        require_enroll_challenge=_bool(env.get("REQUIRE_ENROLL_CHALLENGE"), False),
        challenge_ttl_sec=_int(env, "CHALLENGE_TTL_SEC", 20),
        challenge_min_sec=_float(env, "CHALLENGE_MIN_SEC", 1.5),
        frames_per_action=_int(env, "FRAMES_PER_ACTION", 3),
        blink_ear_drop=_float(env, "BLINK_EAR_DROP", 0.6),
        blink_recover_ratio=_float(env, "BLINK_RECOVER_RATIO", 0.85),
        challenge_yaw_delta=_float(env, "CHALLENGE_YAW_DELTA", 0.15),
        planar_check=_bool(env.get("PLANAR_CHECK"), False),
        planar_residual_min=_float(env, "PLANAR_RESIDUAL_MIN", 0.02),
        session_ttl_sec=_int(env, "SESSION_TTL_SEC", 300),
        session_max_submits=_int(env, "SESSION_MAX_SUBMITS", 3),
        token_ttl_sec=_int(env, "TOKEN_TTL_SEC", 120),
        face_service_api_key=env.get("FACE_SERVICE_API_KEY", ""),
        auth_disabled_for_dev=_bool(env.get("AUTH_DISABLED_FOR_DEV"), False),
        face_token_secret=env.get("FACE_TOKEN_SECRET", ""),
        kiosk_mode=_bool(env.get("KIOSK_MODE"), False),
        kiosk_allowed_ips=tuple(filter(None, env.get("KIOSK_ALLOWED_IPS", "").split(","))),
        face_cors_origins=tuple(filter(None, env.get("FACE_CORS_ORIGINS", "").split(","))),
        host=env.get("HOST", "127.0.0.1"),
        port=_int(env, "PORT", 5000),
        waitress_threads=_int(env, "WAITRESS_THREADS", 4),
        flask_debug=_bool(env.get("FLASK_DEBUG"), False),
        preview_interval_sec=_float(env, "PREVIEW_INTERVAL_SEC", 0.275),
        rate_limit_ip_per_minute=_int(env, "RATE_LIMIT_IP_PER_MINUTE", 120),
        rate_limit_subject_per_minute=_int(env, "RATE_LIMIT_SUBJECT_PER_MINUTE", 30),
        attempt_backoff_base_sec=_float(env, "ATTEMPT_BACKOFF_BASE_SEC", 0.1),
        https_terminated=_bool(env.get("HTTPS_TERMINATED"), False),
        db_busy_retries=_int(env, "DB_BUSY_RETRIES", 3),
        db_retry_delay_sec=_float(env, "DB_RETRY_DELAY_SEC", 0.05),
        detect_min_score=_float(env, "DETECT_MIN_SCORE", 0.5),
        detect_min_width_ratio=_float(env, "DETECT_MIN_WIDTH_RATIO", 0.12),
        detect_min_height_ratio=_float(env, "DETECT_MIN_HEIGHT_RATIO", 0.12),
        detect_min_area_ratio=_float(env, "DETECT_MIN_AREA_RATIO", 0.03),
        detect_center_x_min=_float(env, "DETECT_CENTER_X_MIN", 0.2),
        detect_center_x_max=_float(env, "DETECT_CENTER_X_MAX", 0.8),
        detect_center_y_min=_float(env, "DETECT_CENTER_Y_MIN", 0.15),
        detect_center_y_max=_float(env, "DETECT_CENTER_Y_MAX", 0.85),
        liveness_landmark_movement=_float(env, "LIVENESS_LANDMARK_MOVEMENT", 2.0),
        liveness_pixel_difference=_float(env, "LIVENESS_PIXEL_DIFFERENCE", 2.0),
    )


DEFAULT_SETTINGS = load_settings()
# Public aliases retained for callers configuring the previous face app.
MATCH_THRESHOLD = DEFAULT_SETTINGS.verify_threshold
LIVENESS_IDENTITY_MIN = DEFAULT_SETTINGS.same_person_min
MAX_REGISTER_IMAGES = DEFAULT_SETTINGS.enroll_max_frames
MAX_VERIFY_IMAGES = DEFAULT_SETTINGS.verify_max_frames
FACE_MIN_SIZE = DEFAULT_SETTINGS.verify_min_face_px