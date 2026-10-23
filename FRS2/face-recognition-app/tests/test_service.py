from types import SimpleNamespace

import numpy as np
import pytest

from face_verification.config import Settings
from face_verification.liveness import ChallengeManager
from face_verification.service import FaceService
from face_verification.store import FaceStore


def _image(seed):
    rng = np.random.default_rng(seed)
    checker = (np.indices((200, 200)).sum(axis=0) // 8 % 2 * 160 + 40).astype("uint8")
    texture = rng.integers(0, 4, (200, 200), dtype="uint8")
    gray = np.clip(checker + texture, 0, 255).astype("uint8")
    gray[0, 0] = seed * 5 % 200 + 30
    return np.repeat(gray[:, :, None], 3, axis=2)


def _fake_face(embedding, seed):
    return SimpleNamespace(
        bbox=np.asarray([20, 20, 180, 180], dtype="float32"),
        det_score=0.99,
        embedding=np.asarray(embedding, dtype="float32"),
        kps=np.asarray([[40, 60], [160, 60], [100, 100], [60, 140], [140, 140]], dtype="float32"),
        landmark_2d_106=np.full((106, 2), seed, dtype="float32"),
    )


class FakeEngine:
    def __init__(self, embedding_factory=None):
        self.embedding_factory = embedding_factory or (lambda image: np.eye(1, 512, 0, dtype="float32")[0])

    def analyze(self, image):
        return [_fake_face(self.embedding_factory(image), int(image[0, 0, 0]))]

    def embed(self, face, image):
        return face.embedding / np.linalg.norm(face.embedding)


class FakeAntiSpoof:
    configured = True

    def __init__(self, scores=None):
        self.scores = iter(scores or [])

    def is_live(self, image, face):
        return next(self.scores, 0.99)


def _service(tmp_path, settings=None, engine=None, anti=None):
    settings = settings or Settings(
        face_data_dir=tmp_path,
        enroll_min_frames=3,
        enroll_keep=4,
        verify_max_frames=8,
        verify_keep=3,
        vote_min_frames=3,
        vote_min_agree=2,
        require_challenge=False,
        same_person_min=0.5,
        duplicate_threshold=0.9,
        impersonation_margin=0.05,
    )
    store = FaceStore(settings)
    service = FaceService(engine or FakeEngine(), store, anti or FakeAntiSpoof(), settings=settings)
    return service, store


def test_enroll_happy_path_and_duplicate_face(tmp_path):
    service, store = _service(tmp_path)
    frames = [_image(10 + index) for index in range(3)]
    enrolled = service.enroll("person.one", frames, consent=True)
    assert enrolled["decision"] == "enrolled"
    assert store.count_subjects() == 1
    assert service.enroll("second", frames, consent=True)["decision"] == "duplicate_face"
    assert service.enroll("person.one", frames, consent=True)["decision"] == "already_enrolled"
    store.close()


def test_enroll_rejects_inconsistent_vectors(tmp_path):
    def vector_for(image):
        return np.eye(1, 512, int(image[0, 0, 0]) % 2, dtype="float32")[0]
    service, store = _service(tmp_path, engine=FakeEngine(vector_for))
    result = service.enroll("mixed", [_image(10), _image(11), _image(12)], consent=True)
    assert result["decision"] == "retry"
    assert result["reason_code"] == "INCONSISTENT_FRAMES"
    store.close()


def test_verify_is_1_to_1_votes_and_rejects_other_face(tmp_path):
    settings = Settings(
        face_data_dir=tmp_path,
        enroll_min_frames=3,
        verify_keep=3,
        vote_min_frames=3,
        vote_min_agree=2,
        require_challenge=False,
        same_person_min=0.5,
        duplicate_threshold=0.99,
        impersonation_margin=0.99,
        verify_threshold=0.5,
    )
    service, store = _service(tmp_path, settings=settings)
    enrol_frames = [_image(50 + index) for index in range(3)]
    assert service.enroll("claimed", enrol_frames, consent=True)["decision"] == "enrolled"
    right = service.verify("claimed", [_image(60 + index) for index in range(3)], None, "sid", "127.0.0.1")
    assert right["decision"] == "verified"

    wrong_engine = FakeEngine(lambda image: np.eye(1, 512, 1, dtype="float32")[0])
    wrong_service = FaceService(wrong_engine, store, FakeAntiSpoof(), settings=settings)
    wrong = wrong_service.verify("claimed", [_image(70 + index) for index in range(3)], None, "sid", "127.0.0.1")
    assert wrong["decision"] == "rejected"
    assert wrong["reason_code"] == "MISMATCH"
    store.close()


def test_verify_mixed_people_and_single_frame_cannot_pass(tmp_path):
    def alternating(image):
        return np.eye(1, 512, int(image[0, 0, 0]) % 2, dtype="float32")[0]
    settings = Settings(
        face_data_dir=tmp_path,
        require_challenge=False,
        verify_max_frames=8,
        verify_keep=4,
        vote_min_frames=2,
        vote_min_agree=1,
        same_person_min=0.5,
        duplicate_threshold=0.99,
        impersonation_margin=0.99,
    )
    service, store = _service(tmp_path, settings=settings, engine=FakeEngine(alternating))
    store.add_subject_with_embeddings("subject", [np.eye(1, 512, 0, dtype="float32")[0]], [0.9], consent=True)
    single = service.verify("subject", [_image(1)], None, "sid", "127.0.0.1")
    assert single["decision"] == "retry"
    mixed = service.verify("subject", [_image(2), _image(3)], None, "sid", "127.0.0.1")
    assert mixed["decision"] == "rejected"
    assert mixed["reason_code"] == "INCONSISTENT_FRAMES"
    store.close()


def test_missing_antispoof_fails_closed_for_enroll_and_verify(tmp_path):
    settings = Settings(face_data_dir=tmp_path, enroll_min_frames=1, require_antispoof=True)
    service, store = _service(tmp_path, settings=settings, anti=SimpleNamespace(configured=False))
    assert service.enroll("subject", [_image(1)], consent=True)["status"] == 503
    store.add_subject_with_embeddings("subject", [np.eye(1, 512, 0, dtype="float32")[0]], [0.9], consent=True)
    result = service.verify("subject", [_image(1), _image(2)], None, "sid", "127.0.0.1")
    assert result["status"] == 503
    assert result["reason_code"] == "ANTISPOOF_NOT_CONFIGURED"
    store.close()


def test_failed_attempts_lock_and_unlock(tmp_path):
    settings = Settings(
        face_data_dir=tmp_path,
        require_challenge=False,
        verify_keep=3,
        vote_min_frames=3,
        vote_min_agree=3,
        same_person_min=0.5,
        duplicate_threshold=0.99,
        impersonation_margin=0.99,
        max_attempts=3,
    )
    service, store = _service(tmp_path, settings=settings, engine=FakeEngine(lambda image: np.eye(1, 512, 1, dtype="float32")[0]))
    store.add_subject_with_embeddings("subject", [np.eye(1, 512, 0, dtype="float32")[0]], [0.9], consent=True)
    for attempt in range(3):
        result = service.verify("subject", [_image(80 + index) for index in range(3)], None, "sid", "127.0.0.1")
    assert result["decision"] == "locked"
    assert service.unlock_subject("subject", "OPERATOR_REVIEW")
    assert store.get_subject("subject")["status"] == "active"
    assert store.get_subject("subject")["failed_attempts"] == 0
    store.close()


def test_quality_retry_does_not_increment_failed_attempts(tmp_path):
    settings = Settings(
        face_data_dir=tmp_path,
        require_challenge=False,
        verify_keep=3,
        vote_min_frames=3,
        same_person_min=0.5,
    )
    service, store = _service(tmp_path, settings=settings)
    store.add_subject_with_embeddings("subject", [np.eye(1, 512, 0, dtype="float32")[0]], [0.9], consent=True)
    tiny = np.zeros((200, 200, 3), dtype="uint8")
    faces = [_fake_face(np.eye(1, 512, 0, dtype="float32")[0], 1) for _ in range(3)]

    class SmallFaceEngine(FakeEngine):
        def analyze(self, image):
            face = _fake_face(np.eye(1, 512, 0, dtype="float32")[0], 1)
            face.bbox = np.asarray([10, 10, 50, 50], dtype="float32")
            return [face]

    service.engine = SmallFaceEngine()
    result = service.verify("subject", [tiny, tiny.copy(), tiny.copy()], None, "sid", "127.0.0.1")

    assert result["decision"] == "retry"
    assert result["reason_code"] == "FACE_TOO_SMALL"
    assert store.get_subject("subject")["failed_attempts"] == 0
    store.close()


def test_verify_accepts_live_frames_that_complete_issued_challenge(tmp_path):
    from face_verification.liveness import CHALLENGE_ACTIONS

    settings = Settings(
        face_data_dir=tmp_path,
        require_challenge=True,
        frames_per_action=3,
        challenge_min_sec=1.5,
        verify_max_frames=8,
        verify_keep=3,
        vote_min_frames=3,
        vote_min_agree=2,
        same_person_min=0.5,
        duplicate_threshold=0.99,
        impersonation_margin=0.99,
    )
    store = FaceStore(settings)
    store.add_subject_with_embeddings("live-subject", [np.eye(1, 512, 0, dtype="float32")[0]], [0.9], consent=True)
    store.create_session("live-session", "live-subject", "verify", 200, "backend-ip")
    now = [100.0]
    challenges = ChallengeManager(store, settings, clock=lambda: now[0])
    issued = challenges.issue("live-session", "browser-ip")

    def landmarks(ear):
        points = np.zeros((106, 2), dtype="float32")
        for indices, start in (((33, 35, 37, 39, 41, 43), 10), ((87, 89, 91, 93, 95, 97), 40)):
            points[indices[0]] = [start, 10]
            points[indices[1]] = [start + 2, 10 - ear * 5]
            points[indices[2]] = [start + 8, 10 - ear * 5]
            points[indices[3]] = [start + 10, 10]
            points[indices[4]] = [start + 8, 10 + ear * 5]
            points[indices[5]] = [start + 2, 10 + ear * 5]
        return points

    frame_faces = []
    base_yaw = 0.0
    action_sequences = []
    for action in issued["actions"]:
        if action == "blink":
            sequence = [(base_yaw, 0.08), (base_yaw, 0.08), (base_yaw, 0.3)]
        elif action == "turn_left":
            sequence = [(base_yaw + 0.1, 0.3), (base_yaw + 0.2, 0.3), (base_yaw + 0.2, 0.3)]
        else:
            sequence = [(base_yaw - 0.1, 0.3), (base_yaw - 0.2, 0.3), (base_yaw - 0.2, 0.3)]
        action_sequences.append(sequence)
        base_yaw = sequence[-1][0]

    all_specs = [(0.0, 0.3)] + [spec for sequence in action_sequences for spec in sequence]
    for index, (yaw, ear) in enumerate(all_specs, start=1):
        face = _fake_face(np.eye(1, 512, 0, dtype="float32")[0], index)
        face.kps = np.asarray([[20, 20], [120, 20], [70 + yaw * 100, 50], [35, 90], [105, 90]], dtype="float32")
        face.landmark_2d_106 = landmarks(ear)
        frame_faces.append(face)

    class ChallengeEngine:
        def analyze(self, image):
            return [frame_faces[int(image[0, 0, 1]) - 1]]
        def embed(self, face, image):
            return face.embedding

    service = FaceService(
        ChallengeEngine(), store, FakeAntiSpoof(), challenges, settings, clock=lambda: now[0]
    )
    now[0] += 2.0
    frames = []
    for index in range(len(frame_faces)):
        image = _image(index + 10)
        image[0, 0, 1] = index + 1
        frames.append(image)

    result = service.verify(
        "live-subject", frames, issued["challenge_id"], "live-session", "browser-ip"
    )

    assert result["decision"] == "verified"
    assert result["frames_used"] == 3
    assert result["liveness_score"] >= settings.antispoof_threshold
    store.close()


def test_identify_finds_active_subject_without_claimed_id(tmp_path, monkeypatch):
    settings = Settings(
        face_data_dir=tmp_path,
        require_challenge=False,
        verify_keep=3,
        vote_min_frames=3,
        vote_min_agree=2,
        same_person_min=0.5,
        verify_threshold=0.5,
        impersonation_margin=0.05,
    )
    engine = FakeEngine()
    service, store = _service(tmp_path, settings=settings, engine=engine)
    store.add_subject_with_embeddings(
        "person-123",
        [np.eye(1, 512, 0, dtype="float32")[0]],
        [0.9],
        consent=True,
    )
    store.add_subject_with_embeddings(
        "person-456",
        [np.eye(1, 512, 1, dtype="float32")[0]],
        [0.9],
        consent=True,
    )
    monkeypatch.setattr("face_verification.service.has_liveness_movement", lambda frames, settings: True)

    result = service.identify([_image(30 + index) for index in range(3)], None, "identify-session", "127.0.0.1")

    assert result["decision"] == "identified"
    assert result["identified_subject_id"] == "person-123"
    store.close()


def test_identify_returns_no_match_below_threshold(tmp_path, monkeypatch):
    settings = Settings(
        face_data_dir=tmp_path,
        require_challenge=False,
        verify_keep=3,
        vote_min_frames=3,
        vote_min_agree=2,
        same_person_min=0.5,
        verify_threshold=0.9,
        impersonation_margin=0.01,
    )
    engine = FakeEngine(lambda image: np.eye(1, 512, 1, dtype="float32")[0])
    service, store = _service(tmp_path, settings=settings, engine=engine)
    store.add_subject_with_embeddings(
        "person-123",
        [np.eye(1, 512, 0, dtype="float32")[0]],
        [0.9],
        consent=True,
    )
    monkeypatch.setattr("face_verification.service.has_liveness_movement", lambda frames, settings: True)

    result = service.identify([_image(40 + index) for index in range(3)], None, "identify-session", "127.0.0.1")

    assert result["decision"] == "no_match"
    assert result["public_message"]
    store.close()


def test_quality_retry_has_specific_no_face_message(tmp_path):
    service, store = _service(tmp_path)

    result = service._quality_retry(["NO_FACE"], 0, 5)

    assert result["public_message"] == "No face detected - center your face in the camera guide."
    store.close()


def test_antispoof_failure_message_explains_live_check():
    result = FaceService._result("rejected", "ANTISPOOF_FAILED")

    assert "live-person check" in result["public_message"]
    assert "dark or blurry" in result["public_message"]
    assert "on-screen prompts" in result["public_message"]


def test_challenge_failure_message_explains_prompted_movement():
    result = FaceService._result("retry", "CHALLENGE_FAILED")

    assert "requested blink or head movement" in result["public_message"]
    assert "Start a new check" in result["public_message"]