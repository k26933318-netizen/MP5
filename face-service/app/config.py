import os
from dataclasses import dataclass
from pathlib import Path


def _load_project_env() -> None:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    insightface_model_name: str
    insightface_model_root: Path
    detection_size: int
    min_detection_score: float
    service_api_key: str
    similarity_threshold: float
    min_face_size: int
    min_image_size: int
    max_upload_bytes: int
    background_min_brightness: float
    background_max_stddev: float
    background_min_white_fraction: float
    face_diagnostics_enabled: bool
    face_center_x_tolerance: float
    face_center_y_tolerance: float
    face_min_width_fraction: float
    face_max_width_fraction: float
    face_max_roll_degrees: float
    face_max_pitch_deviation: float
    face_max_yaw_proxy: float
    face_stability_tolerance: float
    challenge_yaw_delta: float
    challenge_return_tolerance: float
    challenge_frames_per_stage: int
    challenge_ttl_seconds: int
    antispoof_model_a_path: Path
    antispoof_model_b_path: Path
    antispoof_threshold: float
    antispoof_min_frame: float
    require_antispoof: bool


def load_settings() -> Settings:
    threshold = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.50"))
    if not 0.0 < threshold <= 1.0:
        raise ValueError("FACE_SIMILARITY_THRESHOLD must be between 0 and 1")
    min_detection_score = float(os.getenv("FACE_MIN_DETECTION_SCORE", "0.70"))
    if not 0.0 < min_detection_score <= 1.0:
        raise ValueError("FACE_MIN_DETECTION_SCORE must be between 0 and 1")

    def bounded_float(name: str, default: str, minimum: float, maximum: float) -> float:
        value = float(os.getenv(name, default))
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        return value

    def positive_int(name: str, default: str, maximum: int) -> int:
        value = int(os.getenv(name, default))
        if not 1 <= value <= maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}")
        return value

    project_root = Path(__file__).resolve().parents[2]
    return Settings(
        insightface_model_name=os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l"),
        insightface_model_root=Path(os.getenv("INSIGHTFACE_MODEL_ROOT", str(Path.home() / ".insightface"))).expanduser(),
        detection_size=int(os.getenv("FACE_DETECTION_SIZE", "640")),
        min_detection_score=min_detection_score,
        service_api_key=os.getenv("FACE_SERVICE_API_KEY", ""),
        similarity_threshold=threshold,
        min_face_size=int(os.getenv("FACE_MIN_SIZE", "80")),
        min_image_size=int(os.getenv("FACE_MIN_IMAGE_SIZE", "320")),
        max_upload_bytes=int(os.getenv("FACE_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))),
        background_min_brightness=bounded_float("FACE_BACKGROUND_MIN_BRIGHTNESS", "190", 0, 255),
        background_max_stddev=bounded_float("FACE_BACKGROUND_MAX_STDDEV", "24", 0, 100),
        background_min_white_fraction=bounded_float("FACE_BACKGROUND_MIN_WHITE_FRACTION", "0.60", 0.50, 0.75),
        face_diagnostics_enabled=os.getenv("FACE_DIAGNOSTICS_ENABLED", "false").lower() in {"1", "true", "yes"},
        face_center_x_tolerance=bounded_float("FACE_CENTER_X_TOLERANCE", "0.14", 0.01, 0.5),
        face_center_y_tolerance=bounded_float("FACE_CENTER_Y_TOLERANCE", "0.18", 0.01, 0.5),
        face_min_width_fraction=bounded_float("FACE_MIN_WIDTH_FRACTION", "0.24", 0.01, 1),
        face_max_width_fraction=bounded_float("FACE_MAX_WIDTH_FRACTION", "0.62", 0.01, 1),
        face_max_roll_degrees=bounded_float("FACE_MAX_ROLL_DEGREES", "12", 0, 45),
        face_max_pitch_deviation=bounded_float("FACE_MAX_PITCH_DEVIATION", "0.22", 0.01, 0.5),
        face_max_yaw_proxy=bounded_float("FACE_MAX_YAW_PROXY", "0.18", 0.01, 1),
        face_stability_tolerance=bounded_float("FACE_STABILITY_TOLERANCE", "0.045", 0.001, 0.25),
        challenge_yaw_delta=bounded_float("FACE_CHALLENGE_YAW_DELTA", "0.14", 0.01, 1),
        challenge_return_tolerance=bounded_float("FACE_CHALLENGE_RETURN_TOLERANCE", "0.08", 0.01, 0.5),
        challenge_frames_per_stage=positive_int("FACE_CHALLENGE_FRAMES_PER_STAGE", "3", 5),
        challenge_ttl_seconds=positive_int("FACE_CHALLENGE_TTL_SECONDS", "90", 600),
        antispoof_model_a_path=Path(os.getenv(
            "FACE_ANTISPOOF_MODEL_A_PATH",
            str(project_root / "FRS2" / "face-recognition-app" / "models" / "minifasnet_a.onnx"),
        )).expanduser(),
        antispoof_model_b_path=Path(os.getenv(
            "FACE_ANTISPOOF_MODEL_B_PATH",
            str(project_root / "FRS2" / "face-recognition-app" / "models" / "minifasnet_b.onnx"),
        )).expanduser(),
        antispoof_threshold=bounded_float("FACE_ANTISPOOF_THRESHOLD", "0.70", 0, 1),
        antispoof_min_frame=bounded_float("FACE_ANTISPOOF_MIN_FRAME", "0.50", 0, 1),
        require_antispoof=os.getenv("FACE_REQUIRE_ANTISPOOF", "true").lower() in {"1", "true", "yes"},
    )


_load_project_env()
settings = load_settings()
