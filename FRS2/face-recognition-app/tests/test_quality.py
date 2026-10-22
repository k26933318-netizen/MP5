from types import SimpleNamespace

import numpy as np
import pytest

from face_verification.config import Settings
from face_verification.quality import assess


def _sharp_image():
    checker = (np.indices((200, 200)).sum(axis=0) // 8 % 2 * 180 + 35).astype("uint8")
    return np.repeat(checker[:, :, None], 3, axis=2)


def _face(bbox=(20, 20, 180, 180), score=0.99, kps=None, pose=None):
    return SimpleNamespace(
        bbox=np.asarray(bbox, dtype="float32"),
        det_score=score,
        kps=kps,
        pose=pose,
    )


def test_quality_accepts_sharp_well_lit_single_face():
    face = _face()

    result = assess(_sharp_image(), face, [face], "enroll")

    assert result.ok
    assert result.reason_code == "OK"
    assert 0.0 <= result.score <= 1.0


def test_quality_rejects_significant_second_face():
    face = _face()
    second = _face((20, 20, 120, 120))

    result = assess(_sharp_image(), face, [face, second], "verify")

    assert result.reason_code == "MULTIPLE_FACES"
    assert result.public_message == "Only one face should be in view."


def test_quality_size_confidence_blur_lighting_reason_codes():
    image = _sharp_image()
    small = _face((20, 20, 100, 100))
    assert assess(image, small, [small], "enroll").reason_code == "FACE_TOO_SMALL"
    low_confidence = _face(score=0.2)
    assert assess(image, low_confidence, [low_confidence], "verify").reason_code == "LOW_CONFIDENCE"
    blurry_image = np.full((200, 200, 3), 127, dtype="uint8")
    face = _face()
    assert assess(blurry_image, face, [face], "verify").reason_code == "BLURRY"
    dark_pattern = (np.indices((200, 200)).sum(axis=0) // 8 % 2 * 50).astype("uint8")
    dark_image = np.repeat(dark_pattern[:, :, None], 3, axis=2)
    assert assess(dark_image, face, [face], "verify").reason_code == "BAD_LIGHTING"


def test_pose_gate_uses_pose_or_approximate_keypoints():
    image = _sharp_image()
    face = _face(pose=[45.0, 0.0, 0.0])
    result = assess(image, face, [face], "verify", Settings(pose_gate=True))
    assert result.reason_code == "BAD_POSE"

    kps = np.asarray([[10, 10], [100, 10], [90, 60], [40, 100], [80, 100]], dtype="float32")
    face = _face(kps=kps)
    result = assess(image, face, [face], "verify", Settings(pose_gate=True))
    assert result.metrics["pose_source"] == "approximate face.kps"
    assert result.reason_code == "BAD_POSE"

    missing_pose = _face()
    result = assess(image, missing_pose, [missing_pose], "verify", Settings(pose_gate=True))
    assert result.reason_code == "BAD_POSE"
    assert result.metrics["pose_source"] == "unavailable"


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        assess(_sharp_image(), _face(), [_face()], "registration")