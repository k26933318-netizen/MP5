"""Flask application factory and versioned service/browser API."""

import base64
from collections import defaultdict, deque
from functools import wraps
import hmac
import logging
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
from flask import Blueprint, Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from .config import DEFAULT_SETTINGS, Settings
from .audit import AuditLogger
from .engine import FaceEngine, decode_image
from .liveness import AntiSpoofEnsemble, ChallengeManager, client_ip_binding
from .service import FaceService
from .store import FaceStore
from .tokens import TokenManager
from liveness import DEFAULT_CONFIG, ENABLE_LIVENESS, PREVIEW_INTERVAL_SECONDS
from liveness.config import FAST_CAPTURE_GAP_MS, FAST_MODE, PERFORMANCE_DEBUG, RECOGNITION_FRAME_COUNT
from overlay.config import (
    ENABLE_SHAPE_OVERLAY,
    SHAPE_TYPE,
    TRACK_FACE_BOX,
    TRACKING_CONFIG,
    TRACER_CONFIG,
)
from overlay.shape_overlay import draw_face_shapes

_logger = logging.getLogger(__name__)
_TERMINAL_DECISIONS = {"enrolled", "verified", "identified", "no_match", "locked", "duplicate_face", "already_enrolled"}
_IDENTIFY_SESSION_PREFIX = "__faceverify_identify__:"
_SAFE_METADATA_KEYS = {"integration", "request_tag"}
_SAFE_METADATA_VALUE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class RateLimiter:
    """Thread-safe fixed-window IP/subject rate limiter with exponential backoff."""

    def __init__(self, settings: Settings, clock: Callable[[], float] = time.monotonic) -> None:
        self.settings = settings
        self.clock = clock
        self._lock = threading.RLock()
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._violations: dict[str, int] = defaultdict(int)

    def check(self, client_ip: str, subject_id: str | None = None) -> tuple[bool, float]:
        """Admit a request or apply a short exponential delay and reject it."""
        now = self.clock()
        keys = [(f"ip:{client_ip}", self.settings.rate_limit_ip_per_minute)]
        if subject_id:
            keys.append((f"subject:{subject_id}", self.settings.rate_limit_subject_per_minute))
        denied_delay = 0.0
        with self._lock:
            for key, limit in keys:
                events = self._events[key]
                while events and events[0] <= now - 60.0:
                    events.popleft()
                if len(events) >= limit:
                    self._violations[key] += 1
                    denied_delay = min(self.settings.attempt_backoff_base_sec * (2 ** (self._violations[key] - 1)), 5.0)
                    break
            if denied_delay == 0.0:
                for key, _ in keys:
                    self._events[key].append(now)
                    self._violations[key] = 0
        if denied_delay:
            time.sleep(denied_delay)
            return False, denied_delay
        return True, 0.0


def create_blueprint(
    service: FaceService,
    token_manager: TokenManager,
    settings: Settings = DEFAULT_SETTINGS,
    limiter: RateLimiter | None = None,
    clock: Callable[[], float] = time.time,
) -> Blueprint:
    """Create the versioned API blueprint around an injected service."""
    blueprint = Blueprint("face_verification", __name__, url_prefix="/api/v1")
    rate_limiter = limiter or RateLimiter(settings)
    audit_logger = AuditLogger(service.store)
    preview_lock = threading.RLock()
    last_preview: dict[str, float] = {}

    def error_response(status: int, code: str, message: str):
        return jsonify(error=code, message=message, request_id=g.request_id), status

    def require_api_key(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            expected = settings.face_service_api_key
            supplied = request.headers.get("X-API-Key", "")
            if not expected and settings.auth_disabled_for_dev:
                return function(*args, **kwargs)
            if not expected or not hmac.compare_digest(supplied, expected):
                return error_response(401, "UNAUTHORIZED", "Service authentication required.")
            return function(*args, **kwargs)
        return wrapped

    def browser_session(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if settings.kiosk_mode and request.remote_addr not in settings.kiosk_allowed_ips:
                return error_response(403, "FORBIDDEN", "This kiosk is not authorized.")
            session_id = request.headers.get("X-Session-Id", "")
            session = service.store.get_session(session_id) if session_id else None
            if session is None:
                return error_response(401, "INVALID_SESSION", "Session authentication required.")
            if session["expires_at"] < clock():
                service.store.update_session(session_id, {"decision": "expired", "reason_code": "SESSION_EXPIRED"}, terminal=True)
                return jsonify(decision="expired", public_message="Session expired.", retry_allowed=False), 410
            if session["terminal"]:
                return error_response(409, "SESSION_FINISHED", "Session has already finished.")
            g.face_session = session
            g.face_session_id = session_id
            return function(*args, **kwargs)
        return wrapped

    @blueprint.post("/sessions")
    @require_api_key
    def create_session():
        payload = request.get_json(silent=True) or {}
        subject_id = payload.get("subject_id")
        purpose = payload.get("purpose")
        identifying = purpose == "identify"
        if purpose not in {"enroll", "verify", "identify"}:
            return error_response(400, "INVALID_REQUEST", "A valid purpose is required.")
        if not identifying and (not isinstance(subject_id, str) or not subject_id):
            return error_response(400, "INVALID_REQUEST", "subject_id is required for this purpose.")
        allowed, delay = rate_limiter.check(request.remote_addr or "", None if identifying else subject_id)
        if not allowed:
            response = error_response(429, "RATE_LIMITED", "Too many requests.")
            response[0].headers["Retry-After"] = str(max(1, int(delay)))
            return response
        requested_ttl = payload.get("ttl_sec", settings.session_ttl_sec)
        if not isinstance(requested_ttl, int) or requested_ttl < 1:
            return error_response(400, "INVALID_REQUEST", "ttl_sec must be a positive integer.")
        ttl = min(requested_ttl, settings.session_ttl_sec)
        session_id = secrets.token_urlsafe(32)
        stored_subject_id = f"{_IDENTIFY_SESSION_PREFIX}{session_id}" if identifying else subject_id
        expires_at = clock() + ttl
        raw_metadata = payload.get("metadata") or {}
        if not isinstance(raw_metadata, dict):
            return error_response(400, "INVALID_REQUEST", "metadata must be an object.")
        metadata = {
            key: value
            for key, value in raw_metadata.items()
            if key in _SAFE_METADATA_KEYS
            and isinstance(value, str)
            and _SAFE_METADATA_VALUE.fullmatch(value)
        }
        try:
            service.store.create_session(
                session_id,
                stored_subject_id,
                "verify" if identifying else purpose,
                expires_at,
                client_ip_binding(request.remote_addr or "", settings),
                bool(payload.get("allow_replace", False)) if purpose == "enroll" else False,
                metadata,
            )
        except ValueError as error:
            _logger.info("session creation rejected request_id=%s error_type=%s", g.request_id, type(error).__name__)
            return error_response(400, "INVALID_REQUEST", str(error))
        except Exception:
            _logger.exception("session creation failed request_id=%s", g.request_id)
            return error_response(500, "INTERNAL_ERROR", "Could not create the requested session. Please try again.")
        return jsonify(
            session_id=session_id,
            expires_at=expires_at,
            challenge_required=settings.require_challenge if purpose in {"verify", "identify"} else settings.require_enroll_challenge,
        ), 201

    @blueprint.get("/sessions/<session_id>/result")
    @require_api_key
    def get_session_result(session_id: str):
        session = service.store.get_session(session_id)
        if session is None or not session["terminal"] or session["result"] is None:
            return error_response(404, "NOT_FOUND", "Session result is not available.")
        return jsonify(session["result"])

    @blueprint.get("/session/challenge")
    @browser_session
    def get_challenge():
        session = g.face_session
        if session["purpose"] != "verify" and not settings.require_enroll_challenge:
            return error_response(409, "CHALLENGE_DISABLED", "This session does not require a challenge.")
        try:
            result = service.challenge_manager.issue(g.face_session_id, request.remote_addr or "")
        except Exception as error:
            _logger.exception("challenge issue failed request_id=%s error_type=%s", g.request_id, type(error).__name__)
            return error_response(500, "INTERNAL_ERROR", "Could not create a challenge.")
        return jsonify(result)

    def _preview_response(include_shape_overlay: bool = False):
        ip = request.remote_addr or ""
        blink_preview = ENABLE_LIVENESS and (
            g.face_session["purpose"] == "verify"
            or g.face_session["subject_id"].startswith(_IDENTIFY_SESSION_PREFIX)
        )
        preview_interval = (
            min(settings.preview_interval_sec, PREVIEW_INTERVAL_SECONDS)
            if blink_preview
            else settings.preview_interval_sec
        )
        now = clock()
        with preview_lock:
            previous = last_preview.get(ip, 0.0)
            if now - previous < preview_interval:
                return error_response(429, "RATE_LIMITED", "Please wait before the next preview.")
            last_preview[ip] = now
        payload = request.get_json(silent=True) or {}
        try:
            image = decode_image(payload.get("frame", ""), settings)
            detections = []
            if blink_preview and include_shape_overlay:
                found, detections = service.engine.detect_face_details(image)
                blink_result = service.update_blink_liveness(g.face_session_id, image, bool(found))
            elif blink_preview:
                blink_result = service.update_blink_liveness(g.face_session_id, image, True)
                found = blink_result.state != "low_quality"
            elif include_shape_overlay:
                found, detections = service.engine.detect_face_details(image)
            else:
                found = service.engine.detect_face(image)
            if blink_preview:
                guidance = (
                    "Verified"
                    if blink_result.is_live
                    else blink_result.instruction
                )
                liveness_data = {
                    "is_live": blink_result.is_live,
                    "blink_count": blink_result.blink_count,
                    "required_blinks": DEFAULT_CONFIG.blink_action_required_count,
                    "status": blink_result.status,
                    "state": blink_result.state,
                    "instruction": blink_result.instruction,
                    "progress": blink_result.progress,
                    "required_progress": blink_result.required_progress,
                    "retries": blink_result.retries,
                }
                liveness_data["ear"] = blink_result.ear
                liveness_data["yaw"] = blink_result.yaw
                liveness_data["mouth"] = blink_result.mouth
                liveness_data.update(service.get_blink_liveness_debug(g.face_session_id))
                response = {
                    "guidance": guidance,
                    "face_ready": bool(found),
                    "liveness_required": True,
                    "fast_mode": FAST_MODE,
                    "recognition_frame_count": RECOGNITION_FRAME_COUNT,
                    "capture_gap_ms": FAST_CAPTURE_GAP_MS,
                    "performance_debug": PERFORMANCE_DEBUG,
                    "server_performance": service.get_performance_debug(),
                    "performance_metadata": service.get_performance_metadata(),
                    "liveness": liveness_data,
                }
            else:
                response = {
                    "guidance": "Face centered. Hold still." if found else "Please center your face.",
                    "face_ready": bool(found),
                }

            if include_shape_overlay and ENABLE_SHAPE_OVERLAY:
                overlay_frame = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
                try:
                    draw_started = time.perf_counter()
                    # --- shape overlay (added) ---
                    tracer_shape = (
                        "corner_brackets"
                        if TRACER_CONFIG["style"] == "modern"
                        else SHAPE_TYPE
                    )
                    draw_face_shapes(
                        overlay_frame,
                        detections,
                        track_key=g.face_session_id,
                        shape=tracer_shape,
                    )
                    encoded, png = cv2.imencode(".png", overlay_frame)
                    if not encoded:
                        raise RuntimeError("OpenCV failed to encode the shape overlay.")
                    response["shape_overlay"] = (
                        "data:image/png;base64," + base64.b64encode(png).decode("ascii")
                    )
                    service.record_performance_stage("drawing", draw_started)
                except Exception:
                    _logger.exception("shape overlay failed request_id=%s", g.request_id)
                    raise
            http_response = jsonify(response)
            # --- face-follow box (added) ---
            if include_shape_overlay and TRACK_FACE_BOX:
                if detections:
                    height, width = image.shape[:2]
                    face_box = max(
                        (box for box, _ in detections),
                        key=lambda box: max(0.0, float(box[2] - box[0]))
                        * max(0.0, float(box[3] - box[1])),
                    )
                    normalized_box = np.clip(
                        np.asarray(face_box, dtype=np.float32)
                        / np.asarray([width, height, width, height], dtype=np.float32),
                        0.0,
                        1.0,
                    )
                    http_response.headers["X-Face-Box"] = ",".join(
                        f"{coordinate:.6f}" for coordinate in normalized_box
                    )
                else:
                    http_response.headers["X-Face-Box"] = "none"
            return http_response
        except Exception as error:
            _logger.warning("preview failed request_id=%s error_type=%s", g.request_id, type(error).__name__)
            return jsonify(guidance="Please center your face and try again.", face_ready=False), 400

    @blueprint.post("/session/preview")
    @browser_session
    def preview():
        return _preview_response()

    @blueprint.post("/session/preview-overlay")
    @browser_session
    def preview_with_shape_overlay():
        return _preview_response(include_shape_overlay=True)

    @blueprint.post("/session/liveness/reset")
    @browser_session
    def reset_liveness():
        session = g.face_session
        if not ENABLE_LIVENESS or session["purpose"] != "verify":
            return error_response(409, "LIVENESS_DISABLED", "A liveness retry is not available for this session.")
        service.reset_blink_liveness(g.face_session_id)
        return jsonify(reset=True)

    # --- fast smooth tracking (added) ---
    @blueprint.post("/session/track-box")
    @browser_session
    def track_face_box():
        started = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        try:
            image = decode_image(payload.get("frame", ""), settings)
            height, width = image.shape[:2]
            detector_width = int(TRACKING_CONFIG["detector_width"])
            if width > detector_width:
                scale = detector_width / width
                image = cv2.resize(
                    image,
                    (detector_width, max(1, round(height * scale))),
                    interpolation=cv2.INTER_LINEAR,
                )
            _, detections = service.engine.detect_face_details(image)
            face_box = None
            if detections:
                face_box = max(
                    (box for box, _ in detections),
                    key=lambda box: max(0.0, float(box[2] - box[0]))
                    * max(0.0, float(box[3] - box[1])),
                )
                image_height, image_width = image.shape[:2]
                face_box = np.clip(
                    np.asarray(face_box, dtype=np.float32)
                    / np.asarray(
                        [image_width, image_height, image_width, image_height],
                        dtype=np.float32,
                    ),
                    0.0,
                    1.0,
                ).tolist()
            return jsonify(
                face_box=face_box,
                detector_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
        except Exception as error:
            _logger.warning(
                "face box tracking failed request_id=%s error_type=%s",
                g.request_id,
                type(error).__name__,
            )
            return error_response(400, "TRACKING_FAILED", "Face tracking could not process this frame.")

    @blueprint.post("/session/submit")
    @browser_session
    def submit():
        payload = request.get_json(silent=True) or {}
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            return jsonify(decision="retry", public_message="Please capture more frames and try again.", retry_allowed=True), 400
        if service.store.increment_session_submits(g.face_session_id) > settings.session_max_submits:
            service.store.update_session(g.face_session_id, {"decision": "expired"}, terminal=True)
            return jsonify(decision="expired", public_message="Session expired.", retry_allowed=False), 410
        try:
            decoded = [decode_image(frame, settings) for frame in frames]
        except Exception as error:
            _logger.warning("frame decode failed request_id=%s error_type=%s", g.request_id, type(error).__name__)
            return jsonify(decision="retry", public_message="Image could not be processed. Please try again.", retry_allowed=True), 400

        identifying = g.face_session["subject_id"].startswith(_IDENTIFY_SESSION_PREFIX)
        allowed, delay = rate_limiter.check(
            request.remote_addr or "",
            None if identifying else g.face_session["subject_id"],
        )
        if not allowed:
            response = jsonify(decision="retry", public_message="Please wait and try again.", retry_allowed=True)
            response.status_code = 429
            response.headers["Retry-After"] = str(max(1, int(delay)))
            return response

        session = g.face_session
        if session["purpose"] == "enroll":
            result = service.enroll(
                session["subject_id"],
                decoded,
                consent=bool(payload.get("consent", False)),
                allow_replace=session["allow_replace"],
                session_id=g.face_session_id,
                client_ip=request.remote_addr or "",
                challenge_id=payload.get("challenge_id"),
            )
        elif session["subject_id"].startswith(_IDENTIFY_SESSION_PREFIX):
            result = service.identify(
                decoded,
                payload.get("challenge_id"),
                g.face_session_id,
                request.remote_addr or "",
            )
        else:
            result = service.verify(
                session["subject_id"],
                decoded,
                payload.get("challenge_id"),
                g.face_session_id,
                request.remote_addr or "",
            )

        if result["decision"] == "verified":
            try:
                result["token"] = token_manager.issue(session["subject_id"], g.face_session_id)
            except Exception as error:
                _logger.exception("token issue failed request_id=%s error_type=%s", g.request_id, type(error).__name__)
                result = {"decision": "retry", "reason_code": "TOKEN_ERROR", "public_message": "Verification failed. Please try again.", "retry_allowed": True}
        terminal = result["decision"] in _TERMINAL_DECISIONS
        service.store.update_session(g.face_session_id, result, terminal=terminal)
        browser_result = {
            "decision": result["decision"],
            "public_message": result.get("public_message", "Verification failed. Please try again."),
            "retry_allowed": bool(result.get("retry_allowed", False)),
        }
        if result.get("identified_subject_id"):
            browser_result["identified_subject_id"] = result["identified_subject_id"]
        return jsonify(browser_result), result.get("status", 200)

    @blueprint.get("/subjects")
    @require_api_key
    def list_subjects():
        return jsonify(subjects=service.store.list_subjects(
            request.args.get("limit", 100, type=int),
            request.args.get("offset", 0, type=int),
        ))

    @blueprint.get("/subjects/<subject_id>")
    @require_api_key
    def get_subject(subject_id: str):
        subject = service.store.get_subject(subject_id)
        if subject is None:
            return error_response(404, "NOT_FOUND", "Subject not found.")
        return jsonify(subject)

    @blueprint.delete("/subjects/<subject_id>")
    @require_api_key
    def delete_subject(subject_id: str):
        if not service.store.delete_subject(subject_id):
            return error_response(404, "NOT_FOUND", "Subject not found.")
        return jsonify(deleted=True)

    @blueprint.post("/subjects/<subject_id>/unlock")
    @require_api_key
    def unlock_subject(subject_id: str):
        payload = request.get_json(silent=True) or {}
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return error_response(400, "INVALID_REQUEST", "An unlock reason is required.")
        if not service.unlock_subject(subject_id, reason):
            return error_response(404, "NOT_FOUND", "Subject not found.")
        return jsonify(unlocked=True)

    @blueprint.post("/tokens/introspect")
    @require_api_key
    def introspect_token():
        payload = request.get_json(silent=True) or {}
        token = payload.get("token")
        if not isinstance(token, str):
            return error_response(400, "INVALID_REQUEST", "A token is required.")
        result = token_manager.introspect(token, bool(payload.get("consume", False)))
        if result is None:
            audit_logger.record("token_consumed", None, {"valid": False})
            return jsonify(active=False), 200
        if payload.get("consume"):
            audit_logger.record("token_consumed", result["subject_id"], {"valid": True})
        return jsonify(result)

    @blueprint.get("/health")
    @require_api_key
    def health():
        return jsonify(status="ok", model=settings.face_model)

    @blueprint.get("/config")
    @require_api_key
    def get_config():
        return jsonify(
            face_model=settings.face_model,
            face_profile=settings.face_profile,
            challenge_required=settings.require_challenge,
            antispoof_required=settings.require_antispoof,
            verify_threshold=service.verify_threshold,
        )

    return blueprint


def create_app(
    settings: Settings = DEFAULT_SETTINGS,
    service: FaceService | None = None,
    engine: Any | None = None,
    store: FaceStore | None = None,
    antispoof: Any | None = None,
    clock: Callable[[], float] = time.time,
) -> Flask:
    """Build the API without loading models unless the caller creates the engine."""
    if not settings.face_service_api_key and not settings.auth_disabled_for_dev:
        raise RuntimeError("FACE_SERVICE_API_KEY is required unless AUTH_DISABLED_FOR_DEV=true.")
    if not settings.face_service_api_key and settings.auth_disabled_for_dev:
        _logger.critical("AUTH_DISABLED_FOR_DEV is enabled. Service API key authentication is disabled.")
    if not settings.face_token_secret:
        raise RuntimeError("FACE_TOKEN_SECRET is required.")
    if settings.kiosk_mode and not settings.https_terminated:
        _logger.warning("KIOSK_MODE is enabled without HTTPS_TERMINATED=true; terminate HTTPS before production use.")
    if settings.face_model != "buffalo_l":
        _logger.warning("FACE_MODEL=%s invalidates all embeddings created with other models.", settings.face_model)

    if service is None:
        if settings.require_antispoof:
            model_paths = (
                ("ANTISPOOF_MODEL_A_PATH", settings.antispoof_model_a_path),
                ("ANTISPOOF_MODEL_B_PATH", settings.antispoof_model_b_path),
            )
            missing_models = []
            for setting_name, configured_path in model_paths:
                if not configured_path:
                    missing_models.append(f"{setting_name} is unset (configure a local .onnx file)")
                elif not Path(configured_path).is_file():
                    missing_models.append(
                        f"{setting_name}: {Path(configured_path).resolve()} (place the model file there)"
                    )
            if missing_models:
                raise RuntimeError(
                    "Required anti-spoof model files are missing:\n- "
                    + "\n- ".join(missing_models)
                    + "\nConfigure both paths in .env before starting the service."
                )
        store = store or FaceStore(settings)
        engine = engine or FaceEngine(settings)
        antispoof = antispoof or AntiSpoofEnsemble(settings)
        antispoof.load()
        service = FaceService(engine, store, antispoof, ChallengeManager(store, settings, clock), settings, clock)
    store = store or service.store
    token_manager = TokenManager(store, settings, clock)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length
    app.extensions["face_service"] = service
    app.extensions["face_store"] = store
    app.register_blueprint(create_blueprint(service, token_manager, settings, RateLimiter(settings), clock))

    @app.before_request
    def begin_request():
        g.request_started = time.perf_counter()
        g.request_id = secrets.token_hex(12)

    @app.after_request
    def finish_request(response):
        duration_ms = (time.perf_counter() - getattr(g, "request_started", time.perf_counter())) * 1000
        _logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            getattr(g, "request_id", "unknown"), request.method, request.path, response.status_code, duration_ms,
        )
        response.headers["X-Request-Id"] = getattr(g, "request_id", "unknown")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        origin = request.headers.get("Origin")
        if request.path.startswith("/api/v1/session/") and origin in settings.face_cors_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-Id"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        return jsonify(
            error="HTTP_ERROR",
            message="The requested route or method is unavailable.",
            request_id=getattr(g, "request_id", "unknown"),
        ), error.code or 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        request_id = getattr(g, "request_id", "unknown")
        _logger.exception("request_id=%s unhandled exception_type=%s", request_id, type(error).__name__)
        return jsonify(error="INTERNAL_ERROR", message="The request could not be processed.", request_id=request_id), 500

    return app