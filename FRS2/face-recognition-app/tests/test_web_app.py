import logging

import demo.demo_backend as web_app
from client.python_client import ServiceError


def test_browser_pages_are_branded_camera_only_and_styled():
    client = web_app.app.test_client()

    for path in ("/", "/enroll", "/verify"):
        response = client.get(path)
        assert response.status_code == 200
        assert b"FaceVerify" in response.data
        assert b"DEMO ONLY" not in response.data
        assert b"image-upload" not in response.data

    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200

    verify_page = client.get("/verify").data
    assert b'data-mode="identify"' in verify_page
    assert b'id="subject"' not in verify_page


def test_session_proxy_logs_failure_and_returns_actionable_auth_message(monkeypatch, caplog):
    class RejectedClient:
        def create_session(self, *args, **kwargs):
            raise ServiceError(401, "Service authentication required.")

    monkeypatch.setattr(web_app, "_client", lambda: RejectedClient())
    caplog.set_level(logging.ERROR, logger=web_app.app.logger.name)

    response = web_app.app.test_client().post(
        "/service-api/sessions",
        json={"subject_id": "person-1", "purpose": "enroll"},
    )

    assert response.status_code == 502
    assert "rejected its server credentials" in response.json["message"]
    assert "session creation failed" in caplog.text


def test_session_proxy_explains_unavailable_service(monkeypatch):
    class UnavailableClient:
        def create_session(self, *args, **kwargs):
            raise ServiceError(0, "Face verification service is unavailable.")

    monkeypatch.setattr(web_app, "_client", lambda: UnavailableClient())

    response = web_app.app.test_client().post(
        "/service-api/sessions",
        json={"subject_id": "person-1", "purpose": "verify"},
    )

    assert response.status_code == 503
    assert "service is unavailable" in response.json["message"]