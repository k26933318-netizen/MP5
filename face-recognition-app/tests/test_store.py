import sqlite3
from pathlib import Path

import numpy as np
import pytest
from cryptography.fernet import Fernet

from face_verification.config import Settings
from face_verification.store import FaceStore


def _vector(position):
    vector = np.zeros(512, dtype="float32")
    vector[position] = 1
    return vector


def _store(tmp_path, **settings):
    return FaceStore(Settings(face_data_dir=tmp_path, **settings))


def test_subject_crud_and_faiss_topk(tmp_path):
    store = _store(tmp_path)
    store.add_subject_with_embeddings(
        "subj-a",
        [_vector(0), _vector(1)],
        [0.9, 0.8],
        consent=True,
        extra={"integration": "demo", "name": "Alice Smith", "email": "alice@example.com"},
    )
    store.add_subject_with_embeddings("subj-b", [_vector(2)], [0.85], consent=True)

    assert store.count_subjects() == 2
    assert store.get_subject("subj-a")["consent_at"] is not None
    assert store.get_subject("subj-a")["extra"] == {"integration": "demo"}
    assert len(store.get_embeddings("subj-a")) == 2
    assert store.search_topk(_vector(0), exclude_subject_id="subj-a")[0][0] == "subj-b"
    queries = [_vector(0), _vector(2)]
    assert store.search_topk_batch(queries, k=2) == [
        store.search_topk(query, k=2) for query in queries
    ]
    assert store.list_active_subject_ids() == {"subj-a", "subj-b"}
    assert [item["subject_id"] for item in store.list_subjects()] == ["subj-a", "subj-b"]
    assert not any("vector" in item for item in store.list_subjects())
    assert store.delete_subject("subj-a")
    assert store.get_subject("subj-a") is None
    store.close()


def test_subject_id_and_duplicate_enrollment_validation(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="subject_id"):
        store.add_subject_with_embeddings("a person", [_vector(0)], [1.0], consent=True)
    with pytest.raises(ValueError, match="dimension 512"):
        store.add_subject_with_embeddings("wrong-dimension", [np.ones(3)], [1.0], consent=True)
    store.add_subject_with_embeddings("valid.id", [_vector(0)], [1.0], consent=True)
    with pytest.raises(ValueError, match="already_enrolled"):
        store.add_subject_with_embeddings("valid.id", [_vector(1)], [1.0], consent=True)
    store.close()


def test_encryption_round_trip_and_missing_key_refusal(tmp_path):
    key = Fernet.generate_key().decode("ascii")
    store = _store(tmp_path, face_db_key=key)
    store.add_subject_with_embeddings("encrypted", [_vector(0)], [0.9], consent=True)
    stored_blob = store._connection.execute("SELECT vector FROM embeddings").fetchone()[0]
    assert bytes(stored_blob).startswith(b"gAAAA")
    store.close()

    reopened = _store(tmp_path, face_db_key=key)
    assert np.allclose(reopened.get_embeddings("encrypted")[0].vector, _vector(0))
    reopened.close()
    with pytest.raises(RuntimeError, match="FACE_DB_KEY is required"):
        _store(tmp_path)


def test_plain_embedding_database_migrates_to_encrypted(tmp_path):
    plain = _store(tmp_path)
    plain.add_subject_with_embeddings("migrate", [_vector(4)], [0.8], consent=True)
    plain.close()
    key = Fernet.generate_key().decode("ascii")

    encrypted = _store(tmp_path, face_db_key=key)

    marker = encrypted._connection.execute(
        "SELECT value FROM meta WHERE key = 'embeddings_encrypted'"
    ).fetchone()[0]
    blob = encrypted._connection.execute("SELECT vector FROM embeddings").fetchone()[0]
    assert marker == "true"
    assert bytes(blob).startswith(b"gAAAA")
    assert np.allclose(encrypted.get_embeddings("migrate")[0].vector, _vector(4))
    encrypted.close()


def test_model_mismatch_is_refused(tmp_path):
    first = _store(tmp_path, face_model="buffalo_l")
    first.close()
    with pytest.raises(RuntimeError, match="differs from FACE_MODEL"):
        _store(tmp_path, face_model="antelopev2")


def test_challenge_ip_expiry_single_use_and_token_single_use(tmp_path):
    now = [100.0]
    store = FaceStore(Settings(face_data_dir=tmp_path), clock=lambda: now[0])
    store.create_session("session-1", "subject-1", "verify", 200, "127.0.0.1")
    store.create_challenge("challenge-1", "session-1", "127.0.0.1", ["blink", "turn_left"], 100, 120)

    assert store.consume_challenge("challenge-1", "session-1", "127.0.0.2", 101) is None
    assert store.consume_challenge("challenge-1", "session-1", "127.0.0.1", 121) is None
    assert store.consume_challenge("challenge-1", "session-1", "127.0.0.1", 101) is not None
    assert store.consume_challenge("challenge-1", "session-1", "127.0.0.1", 101) is None
    store.add_token("jti-1", "subject-1", 120)
    assert store.consume_token("jti-1", 119)
    assert not store.consume_token("jti-1", 119)
    store.close()


def test_audit_log_is_append_only(tmp_path):
    store = _store(tmp_path)
    store.audit("verified", "subject-1", {"match_score": 0.9})
    assert store.list_audit()[0]["event"] == "verified"
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._connection.execute("DELETE FROM audit_log")
    store.close()


def test_legacy_import_is_explicit_and_test_data_only(tmp_path):
    import json
    import os
    import subprocess
    import sys

    import faiss

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    vectors = np.eye(3, 512, dtype="float32")
    index = faiss.IndexFlatIP(512)
    index.add(vectors)
    faiss.write_index(index, str(legacy_dir / "faces.index"))
    (legacy_dir / "names.json").write_text(json.dumps(["legacy-id"] * 3), encoding="utf-8")
    target = tmp_path / "imported"
    script = Path(__file__).resolve().parents[1] / "tools" / "import_legacy.py"
    environment = {**os.environ, "FACE_MODEL": "buffalo_l"}

    denied = subprocess.run(
        [sys.executable, str(script), "--legacy-dir", str(legacy_dir), "--face-data-dir", str(target)],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert denied.returncode != 0
    imported = subprocess.run(
        [
            sys.executable, str(script), "--legacy-dir", str(legacy_dir),
            "--face-data-dir", str(target), "--confirm-test-data-only",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert imported.returncode == 0, imported.stderr
    store = FaceStore(Settings(face_data_dir=target))
    assert store.count_subjects() == 1
    assert len(store.get_embeddings("legacy-id")) == 3
    assert store.get_subject("legacy-id")["consent_at"] is None
    store.close()