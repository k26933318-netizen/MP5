"""Browser application and authenticated face-service gateway."""

import logging
import os
from pathlib import Path
import sys

from flask import Flask, jsonify, render_template, request, send_from_directory
from waitress import serve
from werkzeug.middleware.dispatcher import DispatcherMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
    if candidate.is_file():
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from client.python_client import ServiceClient
from client.python_client import ServiceError

from face_verification.api import create_app as create_face_service_app
from face_verification.config import load_settings
from liveness import (
    CHECK_WHITE_BACKGROUND,
    LOW_QUALITY_SCORE_THRESHOLD,
    SHOW_DEBUG_INFO,
    WHITE_BACKGROUND_CHECK_INTERVAL_MS,
    WHITE_BACKGROUND_MAX_VARIATION,
    WHITE_BACKGROUND_MIN_BRIGHTNESS,
)
from overlay.config import TRACER_CONFIG
from demo.ui_text import (
    WHITE_BACKGROUND_CHECK_HINT,
    WHITE_BACKGROUND_NOTICE_BUTTON,
    WHITE_BACKGROUND_NOTICE_LINE_1,
    WHITE_BACKGROUND_NOTICE_LINE_2,
    WHITE_BACKGROUND_NOTICE_REOPEN,
    WHITE_BACKGROUND_NOTICE_TITLE,
    WHITE_BACKGROUND_RUNNING_HINT,
)

app = Flask(__name__, template_folder=".", static_folder="static")
DEFAULT_SERVICE_URL = "http://127.0.0.1:5050/service"
SERVICE_URL = os.getenv("FACE_SERVICE_URL", DEFAULT_SERVICE_URL)
SERVICE_API_KEY = os.getenv("FACE_SERVICE_API_KEY", "")
_logger = logging.getLogger(__name__)


def _client() -> ServiceClient:
    if not SERVICE_API_KEY:
        raise RuntimeError("FACE_SERVICE_API_KEY is not configured on the server.")
    return ServiceClient(app.config.get("FACE_SERVICE_URL", SERVICE_URL), SERVICE_API_KEY)


@app.get("/client/face-capture.js")
def face_capture_script():
    return send_from_directory(Path(__file__).resolve().parents[1] / "client", "face-capture.js")


@app.get("/")
def home_page():
    return render_template("home.html", active_page="home")


@app.get("/enroll")
def enroll_page():
    return render_template("enroll.html", active_page="enroll", tracer_config=TRACER_CONFIG)


@app.get("/verify")
def verify_page():
    return render_template(
        "verify.html",
        active_page="verify",
        tracer_config=TRACER_CONFIG,
        show_debug_info=SHOW_DEBUG_INFO,
        low_quality_score_threshold=LOW_QUALITY_SCORE_THRESHOLD,
        check_white_background=CHECK_WHITE_BACKGROUND,
        white_background_check_interval_ms=WHITE_BACKGROUND_CHECK_INTERVAL_MS,
        white_background_min_brightness=WHITE_BACKGROUND_MIN_BRIGHTNESS,
        white_background_max_variation=WHITE_BACKGROUND_MAX_VARIATION,
        white_background_text={
            "notice_title": WHITE_BACKGROUND_NOTICE_TITLE,
            "notice_line_1": WHITE_BACKGROUND_NOTICE_LINE_1,
            "notice_line_2": WHITE_BACKGROUND_NOTICE_LINE_2,
            "notice_button": WHITE_BACKGROUND_NOTICE_BUTTON,
            "notice_reopen": WHITE_BACKGROUND_NOTICE_REOPEN,
            "running_hint": WHITE_BACKGROUND_RUNNING_HINT,
            "check_hint": WHITE_BACKGROUND_CHECK_HINT,
        },
    )


def _session_error(error: Exception, purpose: str) -> tuple[str, int]:
    action = "enrollment" if purpose == "enroll" else "verification"
    if isinstance(error, ServiceError):
        if error.status_code == 0:
            return "The face service is unavailable. Start the application again and retry.", 503
        if error.status_code == 401:
            return "The face service rejected its server credentials. Check the service configuration.", 502
        if error.status_code == 400:
            return "Enter an ID using letters, numbers, periods, underscores, colons, or hyphens.", 400
        if error.status_code == 429:
            return "Too many requests. Wait a moment, then try again.", 429
        return f"Could not start {action}. Check the server log and try again.", 502
    if isinstance(error, RuntimeError) and "FACE_SERVICE_API_KEY" in str(error):
        return "The face service is not configured. Contact the application administrator.", 503
    return f"Could not start {action}. Check the server log and try again.", 500


@app.post("/service-api/sessions")
def create_session():
    payload = request.get_json(silent=True) or {}
    purpose = payload.get("purpose", "")
    try:
        result = _client().create_session(
            payload.get("subject_id", ""),
            purpose,
            allow_replace=bool(payload.get("allow_replace", False)),
        )
    except Exception as error:
        app.logger.exception("%s session creation failed", purpose or "Unknown")
        message, status = _session_error(error, purpose)
        return jsonify(message=message), status
    return jsonify({**result, "service_url": app.config.get("FACE_SERVICE_URL", SERVICE_URL)})


@app.get("/service-api/sessions/<session_id>/result")
def get_result(session_id: str):
    try:
        result = _client().get_result(session_id)
    except Exception as error:
        app.logger.exception("Session result read failed")
        message, status = _session_error(error, "verify")
        return jsonify(message=message), status
    if result is None:
        return jsonify(message="Session result is not ready."), 404
    return jsonify(result)


@app.post("/service-api/tokens/introspect")
def introspect_token():
    payload = request.get_json(silent=True) or {}
    try:
        result = _client().introspect_token(payload.get("token", ""), consume=True)
    except Exception as error:
        app.logger.exception("Verification token inspection failed")
        message, status = _session_error(error, "verify")
        return jsonify(message=message), status
    return jsonify(result)


def main() -> None:
    """Run the browser app and face API together on one local listener."""
    settings = load_settings()
    face_service_app = create_face_service_app(settings)
    face_service = face_service_app.extensions["face_service"]
    _logger.info("Warming face detection, landmark, and recognition models.")
    face_service.engine.warm_up()
    host = os.getenv("DEMO_HOST", "127.0.0.1")
    port = int(os.getenv("DEMO_PORT", "5050"))
    service_url = f"http://127.0.0.1:{port}/service"
    app.config["FACE_SERVICE_URL"] = service_url
    application = DispatcherMiddleware(app, {"/service": face_service_app})
    _logger.info("FaceVerify is available at http://%s:%s", host, port)
    serve(application, host=host, port=port, threads=settings.waitress_threads)


if __name__ == "__main__":
    main()