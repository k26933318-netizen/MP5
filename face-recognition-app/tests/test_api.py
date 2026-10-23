from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest

from face_verification.api import create_app
from face_verification.config import Settings, load_settings
from face_verification.service import FaceService
from face_verification.store import FaceStore
from face_verification.tokens import TokenManager


def _frame_data_url():
    checker = (np.indices((200, 200)).sum(axis=0) // 8 % 2 * 160 + 40).astype("uint8")
    image = np.repeat(checker[:, :, None], 3, axis=2)
    encoded, payload = cv2.imencode(".jpg", image)
    assert encoded
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


class ApiFakeEngine:
    def analyze(self, image):
        face = SimpleNamespace(
            bbox=np.asarray([20, 20, 180, 180], dtype="float32"),
            det_score=0.99,
            embedding=np.eye(1, 512, 0, dtype="float32")[0],
            kps=np.asarray([[40, 60], [160, 60], [100, 100], [60, 140], [140, 140]], dtype="float32"),
            landmark_2d_106=np.zeros((106, 2), dtype="float32"),
        )
        return [face]

    def embed(self, face, image):
        return face.embedding

    def detect_face(self, image):
        return True

    def detect_face_details(self, image):
        return True, [
            (
                np.asarray([20, 20, 180, 180], dtype="float32"),
                np.asarray([[60, 70], [140, 70]], dtype="float32"),
            )
        ]


class ApiFakeAntiSpoof:
    configured = True

    def is_live(self, image, face):
        return 0.99


def _api(tmp_path):
    settings = Settings(
        face_data_dir=tmp_path,
        face_service_api_key="service-secret",
        face_token_secret="token-secret-for-tests-use-a-long-random-string",
        auth_disabled_for_dev=False,
        enroll_min_frames=3,
        enroll_keep=3,
        require_challenge=False,
        require_enroll_challenge=False,
        rate_limit_ip_per_minute=100,
        rate_limit_subject_per_minute=100,
    )
    store = FaceStore(settings)
    service = FaceService(ApiFakeEngine(), store, ApiFakeAntiSpoof(), settings=settings)
    app = create_app(settings, service=service)
    app.testing = True
    return app.test_client(), service, store, settings


def test_api_key_session_auth_and_browser_response_privacy(tmp_path):
    client, service, store, settings = _api(tmp_path)
    denied = client.post("/api/v1/sessions", json={"subject_id": "subject-1", "purpose": "enroll"})
    assert denied.status_code == 401
    created = client.post(
        "/api/v1/sessions",
        json={"subject_id": "subject-1", "purpose": "enroll", "metadata": {"integration": "demo"}},
        headers={"X-API-Key": "service-secret"},
    )
    assert created.status_code == 201
    session_id = created.json["session_id"]
    assert len(session_id) >= 40
    assert client.get("/api/v1/session/challenge").status_code == 401

    response = client.post(
        "/api/v1/session/submit",
        json={"frames": [_frame_data_url()] * 3, "consent": True},
        headers={"X-Session-Id": session_id},
    )
    assert response.status_code == 200
    assert response.json == {
        "decision": "enrolled",
        "public_message": "",
        "retry_allowed": False,
    }
    assert not ({"score", "match_score", "runner_up", "subject_id", "reason_code"} & set(response.json))

    result = client.get(
        f"/api/v1/sessions/{session_id}/result",
        headers={"X-API-Key": "service-secret"},
    )
    assert result.status_code == 200
    assert result.json["decision"] == "enrolled"
    assert result.headers["Cache-Control"] == "no-store"
    assert store.get_subject("subject-1") is not None
    store.close()


def test_browser_session_id_auth_and_challenge_ip_binding(tmp_path):
    client, service, store, settings = _api(tmp_path)
    settings = Settings(
        **{
            **settings.__dict__,
            "require_challenge": True,
            "challenge_min_sec": 0.0,
        }
    )
    from face_verification.api import create_app

    service.settings = settings
    service.challenge_manager.settings = settings
    app = create_app(settings, service=service)
    app.testing = True
    client = app.test_client()
    created = client.post(
        "/api/v1/sessions",
        json={"subject_id": "subject-2", "purpose": "verify"},
        headers={"X-API-Key": "service-secret"},
    )
    session_id = created.json["session_id"]
    challenge_response = client.get(
        "/api/v1/session/challenge",
        headers={"X-Session-Id": session_id},
        environ_base={"REMOTE_ADDR": "127.0.0.2"},
    )
    assert challenge_response.status_code == 200
    challenge = challenge_response.json
    assert len(challenge["actions"]) == 2
    assert len(set(challenge["actions"])) == 2
    assert service.challenge_manager.consume(challenge["challenge_id"], session_id, "127.0.0.1") is None
    assert service.challenge_manager.consume(challenge["challenge_id"], session_id, "127.0.0.2") is not None
    store.close()


def test_enabled_blink_preview_accepts_ten_frames_per_second(tmp_path, monkeypatch):
    import face_verification.api as api_module

    monkeypatch.setattr(api_module, "ENABLE_LIVENESS", True)
    client, service, store, settings = _api(tmp_path)
    now = [100.0]
    app = create_app(settings, service=service, clock=lambda: now[0])
    app.testing = True
    client = app.test_client()
    created = client.post(
        "/api/v1/sessions",
        json={"subject_id": "subject-blink", "purpose": "verify"},
        headers={"X-API-Key": "service-secret"},
    )
    session_id = created.json["session_id"]
    headers = {"X-Session-Id": session_id}

    first = client.post(
        "/api/v1/session/preview",
        json={"frame": _frame_data_url()},
        headers=headers,
    )
    now[0] += 0.11
    second = client.post(
        "/api/v1/session/preview",
        json={"frame": _frame_data_url()},
        headers=headers,
    )

    assert first.status_code == second.status_code == 200
    assert first.json["liveness_required"] is True
    assert second.json["liveness"]["status"] == "LANDMARKS UNAVAILABLE"
    store.close()


def test_enabled_liveness_can_reset_a_verification_challenge(tmp_path, monkeypatch):
    import face_verification.api as api_module

    monkeypatch.setattr(api_module, "ENABLE_LIVENESS", True)
    client, service, store, settings = _api(tmp_path)
    created = client.post(
        "/api/v1/sessions",
        json={"subject_id": "subject-retry", "purpose": "verify"},
        headers={"X-API-Key": "service-secret"},
    )

    response = client.post(
        "/api/v1/session/liveness/reset",
        headers={"X-Session-Id": created.json["session_id"]},
    )

    assert response.status_code == 200
    assert response.json == {"reset": True}
    store.close()


def test_shape_overlay_preview_is_additive_and_keeps_existing_preview_fields(tmp_path):
    import base64

    client, service, store, settings = _api(tmp_path)
    created = client.post(
        "/api/v1/sessions",
        json={"subject_id": "subject-overlay", "purpose": "enroll"},
        headers={"X-API-Key": "service-secret"},
    )
    session_id = created.json["session_id"]
    response = client.post(
        "/api/v1/session/preview-overlay",
        json={"frame": _frame_data_url()},
        headers={"X-Session-Id": session_id},
    )

    assert response.status_code == 200
    assert response.json["guidance"] == "Face centered. Hold still."
    assert response.json["face_ready"] is True
    assert response.json["shape_overlay"].startswith("data:image/png;base64,")
    assert "face_box" not in response.json
    assert response.headers["X-Face-Box"] == "0.100000,0.100000,0.900000,0.900000"
    png_data = base64.b64decode(response.json["shape_overlay"].split(",", 1)[1])
    overlay = cv2.imdecode(np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert overlay.shape == (200, 200, 4)
    assert np.any(overlay[:, :, 3] > 0)
    assert overlay[100, 100, 3] == 0
    store.close()


def test_shape_overlay_disabled_keeps_preview_payload_unchanged(tmp_path, monkeypatch):
    import face_verification.api as api_module

    monkeypatch.setattr(api_module, "ENABLE_SHAPE_OVERLAY", False)
    client, service, store, settings = _api(tmp_path)
    created = client.post(
        "/api/v1/sessions",
        json={"subject_id": "subject-overlay-disabled", "purpose": "enroll"},
        headers={"X-API-Key": "service-secret"},
    )

    response = client.post(
        "/api/v1/session/preview-overlay",
        json={"frame": _frame_data_url()},
        headers={"X-Session-Id": created.json["session_id"]},
    )

    assert response.status_code == 200
    assert response.json == {
        "guidance": "Face centered. Hold still.",
        "face_ready": True,
    }
    store.close()


def test_face_box_tracking_disabled_keeps_raster_overlay_fallback(tmp_path, monkeypatch):
    import face_verification.api as api_module

    monkeypatch.setattr(api_module, "TRACK_FACE_BOX", False)
    client, service, store, settings = _api(tmp_path)
    created = client.post(
        "/api/v1/sessions",
        json={"subject_id": "subject-static-overlay", "purpose": "enroll"},
        headers={"X-API-Key": "service-secret"},
    )
    response = client.post(
        "/api/v1/session/preview-overlay",
        json={"frame": _frame_data_url()},
        headers={"X-Session-Id": created.json["session_id"]},
    )

    assert response.status_code == 200
    assert response.json["shape_overlay"].startswith("data:image/png;base64,")
    assert "X-Face-Box" not in response.headers
    store.close()


def test_fast_face_box_endpoint_downscales_and_skips_liveness(tmp_path, monkeypatch):
    client, service, store, settings = _api(tmp_path)
    created = client.post(
        "/api/v1/sessions",
        json={"subject_id": "subject-fast-tracking", "purpose": "verify"},
        headers={"X-API-Key": "service-secret"},
    )
    session_id = created.json["session_id"]
    detection_shapes = []

    def detect_face_details(image):
        detection_shapes.append(image.shape[:2])
        return True, [(np.asarray([32, 24, 288, 216], dtype="float32"), None)]

    def unexpected_liveness(*args, **kwargs):
        raise AssertionError("The face-box endpoint must not run blink liveness.")

    monkeypatch.setattr(service.engine, "detect_face_details", detect_face_details)
    monkeypatch.setattr(service, "update_blink_liveness", unexpected_liveness)
    monkeypatch.setattr(
        "face_verification.api.decode_image",
        lambda *_args, **_kwargs: np.zeros((480, 640, 3), dtype=np.uint8),
    )
    response = client.post(
        "/api/v1/session/track-box",
        json={"frame": _frame_data_url()},
        headers={"X-Session-Id": session_id},
    )

    assert response.status_code == 200
    np.testing.assert_allclose(response.json["face_box"], [0.1, 0.1, 0.9, 0.9])
    assert response.json["detector_ms"] >= 0
    assert detection_shapes == [(240, 320)]
    store.close()


def test_identification_session_needs_no_subject_id_and_returns_matched_id(tmp_path, monkeypatch):
    client, service, store, settings = _api(tmp_path)
    created = client.post(
        "/api/v1/sessions",
        json={"purpose": "identify"},
        headers={"X-API-Key": "service-secret"},
    )

    assert created.status_code == 201
    session_id = created.json["session_id"]
    stored_session = store.get_session(session_id)
    assert stored_session["purpose"] == "verify"
    assert stored_session["subject_id"].startswith("__faceverify_identify__:")
    assert store.get_subject(stored_session["subject_id"]) is None

    monkeypatch.setattr(
        service,
        "identify",
        lambda *args: {
            "decision": "identified",
            "identified_subject_id": "person-123",
            "public_message": "Face matched.",
            "retry_allowed": False,
            "status": 200,
        },
    )
    response = client.post(
        "/api/v1/session/submit",
        json={"frames": [_frame_data_url()]},
        headers={"X-Session-Id": session_id},
    )

    assert response.status_code == 200
    assert response.json["decision"] == "identified"
    assert response.json["identified_subject_id"] == "person-123"
    assert store.get_session(session_id)["terminal"]
    store.close()


def test_token_introspection_is_single_use_under_concurrency(tmp_path):
    client, service, store, settings = _api(tmp_path)
    store.add_subject_with_embeddings(
        "subject-token",
        [np.eye(1, 512, 0, dtype="float32")[0]],
        [0.9],
        consent=True,
    )
    manager = TokenManager(store, settings)
    token = manager.issue("subject-token", "session-token")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: manager.introspect(token, consume=True), range(2)))

    assert sum(result is not None for result in results) == 1
    assert manager.introspect(token, consume=False) is None
    store.close()


def test_token_expiry_uses_injected_clock(tmp_path):
    settings = Settings(
        face_data_dir=tmp_path,
        face_service_api_key="service-secret",
        face_token_secret="token-secret-for-tests-use-a-long-random-string",
        token_ttl_sec=3,
    )
    store = FaceStore(settings, clock=lambda: 100.0)
    manager = TokenManager(store, settings, clock=lambda: 100.0)
    token = manager.issue("subject-clock", "session-clock")
    assert manager.introspect(token) is not None

    expired = TokenManager(store, settings, clock=lambda: 103.0)
    assert expired.introspect(token) is None
    store.close()


def test_load_settings_reads_project_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("FACE_SERVICE_API_KEY=dotenv-key\nFACE_TOKEN_SECRET=dotenv-secret\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FACE_SERVICE_API_KEY", raising=False)
    monkeypatch.delenv("FACE_TOKEN_SECRET", raising=False)

    settings = load_settings()

    assert settings.face_service_api_key == "dotenv-key"
    assert settings.face_token_secret == "dotenv-secret"


def test_import_does_not_load_models_runtime_or_faiss():
    code = (
        "import sys; import face_verification.api; "
        "assert not {'insightface', 'onnxruntime', 'faiss'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_http_error_keeps_status_and_request_id(tmp_path):
    client, service, store, settings = _api(tmp_path)

    response = client.get("/missing-route")

    assert response.status_code == 404
    assert response.json["error"] == "HTTP_ERROR"
    assert response.json["request_id"]
    store.close()


def test_create_app_names_missing_required_antispoof_models(tmp_path):
    settings = Settings(
        face_data_dir=tmp_path / "data",
        face_service_api_key="service-secret",
        face_token_secret="token-secret-for-tests-use-a-long-random-string",
        antispoof_model_a_path=str(tmp_path / "missing-a.onnx"),
        antispoof_model_b_path=str(tmp_path / "missing-b.onnx"),
    )

    with pytest.raises(RuntimeError, match="ANTISPOOF_MODEL_A_PATH") as error:
        create_app(settings, engine=ApiFakeEngine())

    assert str(tmp_path / "missing-a.onnx") in str(error.value)
    assert str(tmp_path / "missing-b.onnx") in str(error.value)