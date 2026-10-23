from types import SimpleNamespace

import numpy as np
import pytest

from face_verification.config import Settings
from face_verification.liveness import (
    AntiSpoofEnsemble,
    AntiSpoofNotConfigured,
    ChallengeManager,
    check_blink,
    check_turn,
    evaluate_challenge_actions,
    eye_aspect_ratio_106,
    preprocess_antispoof_crop,
    yaw_proxy,
)


def _eye_landmarks(ear):
    points = np.zeros((106, 2), dtype="float32")
    for indices, start in (((63, 64, 65, 66, 68, 70), 10), ((76, 77, 78, 79, 81, 83), 40)):
        points[indices[0]] = [start, 10]
        points[indices[1]] = [start + 2, 10 - ear * 5]
        points[indices[2]] = [start + 8, 10 - ear * 5]
        points[indices[3]] = [start + 10, 10]
        points[indices[4]] = [start + 8, 10 + ear * 5]
        points[indices[5]] = [start + 2, 10 + ear * 5]
    return points


def _face(yaw=0.0, ear=0.3, bbox=(20, 10, 60, 50)):
    return SimpleNamespace(
        bbox=np.asarray(bbox, dtype="float32"),
        kps=np.asarray([[20, 20], [60, 20], [40 + yaw * 40, 30], [25, 40], [55, 40]], dtype="float32"),
        landmark_2d_106=_eye_landmarks(ear),
    )


class FakeInput:
    name = "input"


class FakeSession:
    def __init__(self, logits):
        self.logits = np.asarray([logits], dtype="float32")
        self.input_tensor = None

    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, inputs):
        self.input_tensor = inputs["input"]
        return [self.logits]


def test_preprocess_fixed_bgr_tensor_is_upstream_to_tensor_convention():
    image = np.zeros((100, 100, 3), dtype="uint8")
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30
    face = _face(bbox=(40, 40, 60, 60))

    tensor = preprocess_antispoof_crop(image, face, 2.7)

    assert tensor.shape == (1, 3, 80, 80)
    assert tensor[0, :, 0, 0] == pytest.approx([10 / 255, 20 / 255, 30 / 255])


def test_two_model_ensemble_scores_real_class_and_requires_both_models():
    first = FakeSession([0.0, 3.0, 0.0])
    second = FakeSession([0.0, 2.0, 0.0])
    ensemble = AntiSpoofEnsemble(sessions=(first, second))
    score = ensemble.is_live(np.zeros((100, 100, 3), dtype="uint8"), _face())
    assert score > 0.7
    assert first.input_tensor.shape == (1, 3, 80, 80)

    unconfigured = AntiSpoofEnsemble(settings=Settings())
    with pytest.raises(AntiSpoofNotConfigured):
        unconfigured.is_live(np.zeros((100, 100, 3), dtype="uint8"), _face())


def test_missing_anti_spoof_models_log_only_once(caplog):
    ensemble = AntiSpoofEnsemble(settings=Settings())
    assert not ensemble.configured
    assert not ensemble.configured
    assert sum("Passive anti-spoofing disabled" in record.message for record in caplog.records) == 1


def test_frame_level_anti_spoof_floor_applies_even_when_mean_passes():
    class SequenceSession(FakeSession):
        def __init__(self, logits):
            self.logits_sequence = [np.asarray([entry], dtype="float32") for entry in logits]
            self.input_tensor = None
        def run(self, output_names, inputs):
            self.input_tensor = inputs["input"]
            return [self.logits_sequence.pop(0)]

    strong = [0.0, 4.0, 0.0]
    weak = [4.0, 0.0, 0.0]
    first = SequenceSession([strong, strong, strong, strong, weak])
    second = SequenceSession([strong, strong, strong, strong, weak])
    ensemble = AntiSpoofEnsemble(
        settings=Settings(antispoof_threshold=0.7, antispoof_min_frame=0.5),
        sessions=(first, second),
    )

    passes, mean_score, scores = ensemble.evaluate_frames([
        (np.zeros((100, 100, 3), dtype="uint8"), _face()) for _ in range(5)
    ])

    assert mean_score >= 0.7
    assert min(scores) < 0.5
    assert not passes


def test_eye_aspect_ratio_and_blink_use_insightface_eye_landmark_indices():
    open_eye = _eye_landmarks(0.3)
    closed_eye = _eye_landmarks(0.08)
    assert eye_aspect_ratio_106(open_eye) == pytest.approx(0.3)
    assert check_blink([open_eye, closed_eye, open_eye])
    assert not check_blink([open_eye, closed_eye, closed_eye])
    assert not check_blink([open_eye, None, open_eye])
    assert not check_blink([open_eye, np.zeros((5, 2)), open_eye])


def test_yaw_proxy_and_turn_direction_use_unmirrored_image_coordinates():
    baseline = _face(0.0)
    moved_right = _face(0.2)
    assert yaw_proxy(moved_right) > yaw_proxy(baseline)
    assert check_turn([baseline, moved_right], "turn_left", 0.15)
    assert not check_turn([baseline, moved_right], "turn_right", 0.15)


def test_challenge_actions_use_baseline_then_action_groups():
    settings = Settings(challenge_yaw_delta=0.15)
    faces = [_face(0.0, 0.3), _face(0.0, 0.3), _face(0.0, 0.08), _face(0.0, 0.3),
             _face(0.05, 0.3), _face(0.15, 0.3), _face(0.2, 0.3)]
    assert evaluate_challenge_actions(["blink", "turn_left"], faces, settings=settings)
    assert not evaluate_challenge_actions(["blink", "turn_left"], faces[:-1], settings=settings)


def test_challenge_is_single_use_bound_to_session_ip_and_timing(tmp_path):
    now = [100.0]
    from face_verification.store import FaceStore

    store = FaceStore(Settings(face_data_dir=tmp_path), clock=lambda: now[0])
    store.create_session("s1", "sub1", "verify", 200, "127.0.0.1")
    manager = ChallengeManager(store, Settings(challenge_min_sec=1.5), clock=lambda: now[0])
    issued = manager.issue("s1", "127.0.0.1")
    assert len(issued["actions"]) == 2 and len(set(issued["actions"])) == 2
    now[0] += 1.0
    assert manager.consume(issued["challenge_id"], "s1", "127.0.0.1") is None

    issued = manager.issue("s1", "127.0.0.1")
    now[0] += 1.6
    assert manager.consume(issued["challenge_id"], "s1", "127.0.0.2") is None
    assert manager.consume(issued["challenge_id"], "s1", "127.0.0.1") is not None
    assert manager.consume(issued["challenge_id"], "s1", "127.0.0.1") is None
    store.close()