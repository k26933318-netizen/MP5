import base64
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_verification.config import Settings, load_settings
from face_verification.engine import FaceEngine, _create_face_analysis, decode_image


class FakeAnalysis:
    def __init__(self, faces):
        self.faces = faces
        self.calls = 0

    def get(self, image):
        self.calls += 1
        return self.faces


def _face(area, embedding):
    side = area**0.5
    return SimpleNamespace(bbox=np.array([0, 0, side, side]), embedding=np.asarray(embedding))


def test_analyze_returns_all_faces_by_area_in_one_model_pass():
    smaller = _face(100, [1.0, 0.0])
    larger = _face(400, [0.0, 1.0])
    analysis = FakeAnalysis([smaller, larger])
    engine = FaceEngine(analysis=analysis)

    result = engine.analyze(np.zeros((20, 20, 3), dtype="uint8"))

    assert result == [larger, smaller]
    assert analysis.calls == 1


def test_embed_normalizes_and_tta_uses_one_flipped_analysis():
    flipped_face = _face(400, [0.0, 1.0])
    analysis = FakeAnalysis([flipped_face])
    engine = FaceEngine(analysis=analysis, settings=Settings(use_tta=True))
    image = np.zeros((20, 20, 3), dtype="uint8")

    embedding = engine.embed(_face(400, [1.0, 0.0]), image)

    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    assert embedding == pytest.approx([2**-0.5, 2**-0.5])
    assert analysis.calls == 1


def test_decode_downscales_and_enforces_pixel_limit():
    image = np.zeros((40, 80, 3), dtype="uint8")
    _, encoded = cv2.imencode(".jpg", image)
    data = base64.b64encode(encoded).decode("ascii")

    decoded = decode_image(data, Settings(max_image_side=20))
    assert decoded.shape[:2] == (10, 20)
    with pytest.raises(ValueError, match="pixel limit"):
        decode_image(data, Settings(max_image_pixels=100))


def test_environment_aliases_and_new_names():
    settings = load_settings({
        "MATCH_THRESHOLD": "0.61",
        "LIVENESS_IDENTITY_MIN": "0.72",
        "MAX_REGISTER_IMAGES": "14",
        "VERIFY_THRESHOLD": "0.67",
    })

    assert settings.verify_threshold == 0.67
    assert settings.match_threshold == 0.67
    assert settings.same_person_min == settings.liveness_identity_min == 0.72
    assert settings.enroll_max_frames == settings.max_register_images == 14


def test_max_accuracy_profile_defaults_and_explicit_overrides():
    profile = load_settings({"FACE_PROFILE": "max_accuracy"})
    assert profile.enroll_min_frames == 5
    assert profile.enroll_keep == 6
    assert profile.use_tta
    assert profile.fusion_alpha == 0.5
    assert profile.verify_max_frames == 5
    assert profile.verify_keep == 3
    assert profile.vote_min_frames == 3
    assert profile.vote_min_agree == 3
    assert profile.verify_min_face_px == 150
    assert profile.pose_gate
    assert profile.eyes_open_gate
    assert profile.impersonation_margin == 0.08

    overridden = load_settings({"FACE_PROFILE": "max_accuracy", "USE_TTA": "false", "VOTE_MIN_AGREE": "2"})
    assert not overridden.use_tta
    assert overridden.vote_min_agree == 2
    assert load_settings({"FACE_PROFILE": "max_accuracy", "MAX_FRAMES_VERIFY": "6"}).verify_max_frames == 6
    assert load_settings({"VERIFY_MAX_FRAMES": "7", "MAX_FRAMES_VERIFY": "6"}).verify_max_frames == 7


def test_face_analysis_adds_3d_landmarks_only_when_pose_gate_enabled(monkeypatch):
    import face_verification.engine as engine_module

    monkeypatch.setattr(engine_module, "_runtime_options", lambda: (["CPUExecutionProvider"], -1))
    calls = []

    class FakeAnalysis:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def prepare(self, **kwargs):
            calls.append(kwargs)

    _create_face_analysis(Settings(pose_gate=False), FakeAnalysis)
    _create_face_analysis(Settings(pose_gate=True), FakeAnalysis)

    assert calls[0]["allowed_modules"] == ["detection", "landmark_2d_106", "recognition"]
    assert calls[2]["allowed_modules"] == [
        "detection", "landmark_2d_106", "recognition", "landmark_3d_68"
    ]
    assert calls[0]["providers"] == ["CPUExecutionProvider"]
    assert calls[1]["ctx_id"] == -1
    assert calls[3]["det_size"] == (640, 640)