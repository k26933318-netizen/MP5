"""SQLite-backed subject, embedding, session, challenge, token and audit storage."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from .config import DEFAULT_SETTINGS, Settings

_logger = logging.getLogger(__name__)
_SUBJECT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_EXTRA_VALUE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_FORBIDDEN_EXTRA_KEYS = {"name", "full_name", "email", "phone", "address", "date_of_birth", "voter_id"}
_SCHEMA_VERSION = "1"
_EMBEDDING_DIMENSION = 512


@dataclass(frozen=True)
class EmbeddingSample:
    """A stored normalized embedding and its non-identifying quality metadata."""

    vector: np.ndarray
    quality: float
    created_at: float


class FaceStore:
    """Transactional face data store; image pixels are never accepted or persisted."""

    def __init__(
        self,
        settings: Settings = DEFAULT_SETTINGS,
        database_path: Path | str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.database_path = Path(database_path or settings.face_data_dir / "face.db")
        self._clock = clock
        self._lock = threading.RLock()
        self._fernet = None
        self._index = None
        self._index_subject_ids: list[str] = []
        self._dimension = _EMBEDDING_DIMENSION
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA busy_timeout = 1000")
        self._initialize_schema()
        self._check_model_metadata()
        self._configure_encryption()
        self._rebuild_index()

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._connection.close()

    def add_subject_with_embeddings(
        self,
        subject_id: str,
        embeddings: Sequence[np.ndarray],
        quality_scores: Sequence[float],
        consent: bool,
        allow_replace: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Atomically insert a subject and its normalized embeddings."""
        self._validate_subject_id(subject_id)
        if not embeddings or len(embeddings) != len(quality_scores):
            raise ValueError("Embeddings and quality scores must be non-empty and aligned.")
        vectors = [self._normalize(vector) for vector in embeddings]
        now = self._clock()
        with self._write_transaction() as connection:
            existing = connection.execute(
                "SELECT subject_id FROM subjects WHERE subject_id = ?", (subject_id,)
            ).fetchone()
            if existing and not allow_replace:
                raise ValueError("already_enrolled")
            if existing:
                connection.execute("DELETE FROM embeddings WHERE subject_id = ?", (subject_id,))
                connection.execute(
                    "UPDATE subjects SET status = 'active', consent_at = ?, failed_attempts = 0, "
                    "locked_at = NULL, extra = ? WHERE subject_id = ?",
                    (now if consent else None, json.dumps(self._safe_extra(extra), separators=(",", ":")), subject_id),
                )
            else:
                connection.execute(
                    "INSERT INTO subjects(subject_id, status, created_at, consent_at, extra) "
                    "VALUES (?, 'active', ?, ?, ?)",
                    (subject_id, now, now if consent else None, json.dumps(self._safe_extra(extra), separators=(",", ":"))),
                )
            for vector, quality in zip(vectors, quality_scores):
                blob = self._encode_vector(vector)
                connection.execute(
                    "INSERT INTO embeddings(subject_id, vector, quality, created_at) VALUES (?, ?, ?, ?)",
                    (subject_id, blob, float(quality), now),
                )
        self._rebuild_index()

    def get_subject(self, subject_id: str) -> dict[str, Any] | None:
        """Return subject metadata without embeddings or caller-supplied personal data."""
        with self._lock:
            row = self._connection.execute(
                "SELECT subject_id, status, created_at, consent_at, failed_attempts, locked_at, extra "
                "FROM subjects WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
        return self._subject_dict(row) if row else None

    def get_embeddings(self, subject_id: str) -> list[EmbeddingSample]:
        """Return a subject's vectors for in-process scoring."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT vector, quality, created_at FROM embeddings WHERE subject_id = ? ORDER BY id",
                (subject_id,),
            ).fetchall()
        return [
            EmbeddingSample(self._decode_vector(row["vector"]), float(row["quality"]), float(row["created_at"]))
            for row in rows
        ]

    def delete_subject(self, subject_id: str) -> bool:
        """Delete a subject and embeddings; dependent sessions and tokens are removed too."""
        with self._write_transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM subjects WHERE subject_id = ?", (subject_id,)
            ).fetchone()
            if not exists:
                return False
            self._insert_audit(connection, "deleted", subject_id, {})
            connection.execute("DELETE FROM tokens WHERE subject_id = ?", (subject_id,))
            connection.execute("DELETE FROM sessions WHERE subject_id = ?", (subject_id,))
            connection.execute("DELETE FROM subjects WHERE subject_id = ?", (subject_id,))
        self._rebuild_index()
        return True

    def list_subjects(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List subject metadata without vectors or personal data."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT subject_id, status, created_at, consent_at, failed_attempts, locked_at, extra "
                "FROM subjects ORDER BY created_at LIMIT ? OFFSET ?",
                (max(1, min(int(limit), 500)), max(0, int(offset))),
            ).fetchall()
        return [self._subject_dict(row) for row in rows]

    def set_status(self, subject_id: str, status: str) -> bool:
        """Set a subject status and track lock time."""
        if status not in {"active", "locked", "disabled"}:
            raise ValueError("Invalid subject status.")
        with self._write_transaction() as connection:
            cursor = connection.execute(
                "UPDATE subjects SET status = ?, locked_at = ? WHERE subject_id = ?",
                (status, self._clock() if status == "locked" else None, subject_id),
            )
        return cursor.rowcount == 1

    def increment_failed(self, subject_id: str) -> int:
        """Increment a failed verification counter and return its new value."""
        with self._write_transaction() as connection:
            connection.execute(
                "UPDATE subjects SET failed_attempts = failed_attempts + 1 WHERE subject_id = ?",
                (subject_id,),
            )
            row = connection.execute(
                "SELECT failed_attempts FROM subjects WHERE subject_id = ?", (subject_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def reset_failed(self, subject_id: str) -> None:
        """Reset a subject's failed verification counter."""
        with self._write_transaction() as connection:
            connection.execute(
                "UPDATE subjects SET failed_attempts = 0 WHERE subject_id = ?", (subject_id,)
            )

    def search_topk(
        self,
        embedding: np.ndarray,
        k: int = 10,
        exclude_subject_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Search FAISS and return up to k distinct subject IDs with best cosine scores."""
        import faiss

        query = self._normalize(embedding).reshape(1, -1)
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            search_count = min(max(1, int(k)), self._index.ntotal)
            found: dict[str, float] = {}
            while True:
                scores, positions = self._index.search(query, search_count)
                for score, position in zip(scores[0], positions[0]):
                    position = int(position)
                    if position < 0:
                        continue
                    subject_id = self._index_subject_ids[position]
                    if subject_id == exclude_subject_id:
                        continue
                    found[subject_id] = max(found.get(subject_id, float("-inf")), float(score))
                if len(found) >= k or search_count == self._index.ntotal:
                    break
                search_count = min(self._index.ntotal, max(search_count + 1, search_count * 2))
            return sorted(found.items(), key=lambda item: item[1], reverse=True)[:max(0, k)]

    # --- speed-up (added) ---
    def search_topk_batch(
        self,
        embeddings: Sequence[np.ndarray],
        k: int = 10,
        exclude_subject_id: str | None = None,
    ) -> list[list[tuple[str, float]]]:
        """Search a batch of normalized queries in one FAISS matrix operation."""
        if not embeddings:
            return []
        queries = np.stack([self._normalize(embedding) for embedding in embeddings]).astype("float32")
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return [[] for _ in embeddings]
            search_count = min(max(1, int(k)), self._index.ntotal)
            found: list[dict[str, float]] = [{} for _ in embeddings]
            while True:
                scores, positions = self._index.search(queries, search_count)
                for query_index, (query_scores, query_positions) in enumerate(zip(scores, positions)):
                    for score, position in zip(query_scores, query_positions):
                        position = int(position)
                        if position < 0:
                            continue
                        subject_id = self._index_subject_ids[position]
                        if subject_id == exclude_subject_id:
                            continue
                        found[query_index][subject_id] = max(
                            found[query_index].get(subject_id, float("-inf")),
                            float(score),
                        )
                if all(len(matches) >= k for matches in found) or search_count == self._index.ntotal:
                    break
                search_count = min(self._index.ntotal, max(search_count + 1, search_count * 2))
            return [
                sorted(matches.items(), key=lambda item: item[1], reverse=True)[:max(0, k)]
                for matches in found
            ]

    def list_active_subject_ids(self) -> set[str]:
        """Read the active-subject filter once for a batched identification request."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT subject_id FROM subjects WHERE status = 'active'"
            ).fetchall()
        return {str(row[0]) for row in rows}

    def count_subjects(self) -> int:
        """Return the number of registered subjects."""
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0])

    def create_session(
        self,
        session_id: str,
        subject_id: str,
        purpose: str,
        expires_at: float,
        client_ip: str,
        allow_replace: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a backend-authorized enrollment or verification session."""
        if purpose not in {"enroll", "verify"}:
            raise ValueError("Invalid session purpose.")
        self._validate_subject_id(subject_id)
        with self._write_transaction() as connection:
            connection.execute(
                "INSERT INTO sessions(session_id, subject_id, purpose, created_at, expires_at, client_ip, "
                "allow_replace, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, subject_id, purpose, self._clock(), expires_at, client_ip,
                    int(allow_replace), json.dumps(metadata or {}, separators=(",", ":")),
                ),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Return session metadata and result, if complete."""
        with self._lock:
            row = self._connection.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return {
            "session_id": row["session_id"],
            "subject_id": row["subject_id"],
            "purpose": row["purpose"],
            "created_at": float(row["created_at"]),
            "expires_at": float(row["expires_at"]),
            "client_ip": row["client_ip"],
            "allow_replace": bool(row["allow_replace"]),
            "metadata": json.loads(row["metadata"] or "{}"),
            "submit_count": int(row["submit_count"]),
            "terminal": bool(row["terminal"]),
            "result": json.loads(row["result"]) if row["result"] else None,
        }

    def update_session(self, session_id: str, result: dict[str, Any], terminal: bool = False) -> None:
        """Persist the latest backend result for a session."""
        with self._write_transaction() as connection:
            connection.execute(
                "UPDATE sessions SET result = ?, terminal = ? WHERE session_id = ?",
                (json.dumps(result, separators=(",", ":")), int(terminal), session_id),
            )

    def increment_session_submits(self, session_id: str) -> int:
        """Atomically increment and return the session submit count."""
        with self._write_transaction() as connection:
            connection.execute(
                "UPDATE sessions SET submit_count = submit_count + 1 WHERE session_id = ?",
                (session_id,),
            )
            row = connection.execute(
                "SELECT submit_count FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def create_challenge(
        self,
        challenge_id: str,
        session_id: str,
        client_ip: str,
        actions: Sequence[str],
        issued_at: float,
        expires_at: float,
    ) -> None:
        """Persist a single-use, session- and IP-bound challenge."""
        with self._write_transaction() as connection:
            connection.execute(
                "INSERT INTO challenges(challenge_id, session_id, client_ip, actions, issued_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (challenge_id, session_id, client_ip, json.dumps(list(actions)), issued_at, expires_at),
            )

    def get_challenge(self, challenge_id: str) -> dict[str, Any] | None:
        """Read an unconsumed challenge for evaluation."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM challenges WHERE challenge_id = ?", (challenge_id,)
            ).fetchone()
        return self._challenge_dict(row) if row else None

    def consume_challenge(
        self,
        challenge_id: str,
        session_id: str,
        client_ip: str,
        now: float,
    ) -> dict[str, Any] | None:
        """Atomically consume a valid challenge exactly once."""
        with self._write_transaction() as connection:
            row = connection.execute(
                "SELECT * FROM challenges WHERE challenge_id = ? AND session_id = ? "
                "AND client_ip = ? AND consumed_at IS NULL AND expires_at >= ?",
                (challenge_id, session_id, client_ip, now),
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                "UPDATE challenges SET consumed_at = ? WHERE challenge_id = ? AND consumed_at IS NULL",
                (now, challenge_id),
            )
            if cursor.rowcount != 1:
                return None
            return self._challenge_dict(row)

    def add_token(self, jti: str, subject_id: str, expires_at: float) -> None:
        """Persist a token identifier before returning its signed token."""
        with self._write_transaction() as connection:
            connection.execute(
                "INSERT INTO tokens(jti, subject_id, expires_at) VALUES (?, ?, ?)",
                (jti, subject_id, expires_at),
            )

    def consume_token(self, jti: str, now: float) -> bool:
        """Consume an unexpired token identifier atomically."""
        with self._write_transaction() as connection:
            cursor = connection.execute(
                "UPDATE tokens SET consumed_at = ? WHERE jti = ? AND consumed_at IS NULL AND expires_at >= ?",
                (now, jti, now),
            )
        return cursor.rowcount == 1

    def token_exists(self, jti: str, now: float) -> bool:
        """Check that a signed token was issued by this store and is still valid."""
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM tokens WHERE jti = ? AND expires_at > ? AND consumed_at IS NULL",
                (jti, now),
            ).fetchone()
        return row is not None

    def audit(self, event: str, subject_id: str | None = None, data: dict[str, Any] | None = None) -> None:
        """Append a non-biometric audit event."""
        with self._write_transaction() as connection:
            self._insert_audit(connection, event, subject_id, data or {})

    def list_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent audit events without embeddings or image data."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT event, subject_id, created_at, data FROM audit_log ORDER BY id DESC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        return [
            {"event": row["event"], "subject_id": row["subject_id"], "created_at": row["created_at"], "data": json.loads(row["data"])}
            for row in rows
        ]

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            for attempt in range(self.settings.db_busy_retries + 1):
                try:
                    self._connection.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as error:
                    if "locked" not in str(error).lower() or attempt >= self.settings.db_busy_retries:
                        raise
                    time.sleep(self.settings.db_retry_delay_sec * (attempt + 1))
            try:
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS subjects(
                    subject_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('active','locked','disabled')),
                    created_at REAL NOT NULL,
                    consent_at REAL,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_at REAL,
                    extra TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS embeddings(
                    id INTEGER PRIMARY KEY,
                    subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
                    vector BLOB NOT NULL,
                    quality REAL NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS embeddings_subject_idx ON embeddings(subject_id);
                CREATE TABLE IF NOT EXISTS sessions(
                    session_id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    purpose TEXT NOT NULL CHECK(purpose IN ('enroll','verify')),
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    client_ip TEXT NOT NULL,
                    allow_replace INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    submit_count INTEGER NOT NULL DEFAULT 0,
                    terminal INTEGER NOT NULL DEFAULT 0,
                    result TEXT
                );
                CREATE TABLE IF NOT EXISTS challenges(
                    challenge_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    client_ip TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    issued_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    consumed_at REAL
                );
                CREATE TABLE IF NOT EXISTS tokens(
                    jti TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    consumed_at REAL
                );
                CREATE TABLE IF NOT EXISTS audit_log(
                    id INTEGER PRIMARY KEY,
                    event TEXT NOT NULL,
                    subject_id TEXT,
                    created_at REAL NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
                BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
                BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
                """
            )

    def _check_model_metadata(self) -> None:
        with self._write_transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (_SCHEMA_VERSION,),
            )
            row = connection.execute("SELECT value FROM meta WHERE key = 'model_name'").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES ('model_name', ?)",
                    (self.settings.face_model,),
                )
            elif row[0] != self.settings.face_model:
                if not self.settings.allow_model_mismatch:
                    raise RuntimeError("Stored face model differs from FACE_MODEL; set ALLOW_MODEL_MISMATCH=true to override.")
                _logger.warning("Configured FACE_MODEL differs from stored embeddings; results may be invalid.")

    def _configure_encryption(self) -> None:
        from cryptography.fernet import Fernet, InvalidToken

        marker = self._connection.execute(
            "SELECT value FROM meta WHERE key = 'embeddings_encrypted'"
        ).fetchone()
        encrypted = bool(marker and marker[0] == "true")
        if encrypted and not self.settings.face_db_key:
            raise RuntimeError("FACE_DB_KEY is required to open this encrypted face database.")
        if self.settings.face_db_key:
            try:
                self._fernet = Fernet(self.settings.face_db_key.encode("ascii"))
            except (ValueError, UnicodeEncodeError) as error:
                raise RuntimeError("FACE_DB_KEY must be a valid Fernet key.") from error
        if encrypted:
            return
        if self._fernet is None:
            with self._write_transaction() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('embeddings_encrypted', 'false')"
                )
            return

        with self._write_transaction() as connection:
            rows = connection.execute("SELECT id, vector FROM embeddings").fetchall()
            for row in rows:
                try:
                    plain = bytes(row["vector"])
                    self._fernet.decrypt(plain)
                except InvalidToken:
                    connection.execute(
                        "UPDATE embeddings SET vector = ? WHERE id = ?",
                        (self._fernet.encrypt(plain), row["id"]),
                    )
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('embeddings_encrypted', 'true')"
            )

    def _rebuild_index(self) -> None:
        import faiss

        with self._lock:
            rows = self._connection.execute(
                "SELECT subject_id, vector FROM embeddings ORDER BY id"
            ).fetchall()
            self._index_subject_ids = []
            if not rows:
                self._index = faiss.IndexFlatIP(self._dimension)
                return
            vectors = [self._decode_vector(row["vector"]) for row in rows]
            self._dimension = int(vectors[0].size)
            matrix = np.stack(vectors).astype("float32")
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(matrix)
            self._index_subject_ids = [row["subject_id"] for row in rows]

    def _encode_vector(self, vector: np.ndarray) -> bytes:
        blob = np.asarray(vector, dtype="<f4").tobytes()
        return self._fernet.encrypt(blob) if self._fernet is not None else blob

    def _decode_vector(self, blob: bytes) -> np.ndarray:
        if self._fernet is not None:
            blob = self._fernet.decrypt(bytes(blob))
        vector = np.frombuffer(blob, dtype="<f4").copy()
        if vector.size != self._dimension and self._index_subject_ids:
            raise RuntimeError("Stored embedding dimensions are inconsistent.")
        return self._normalize(vector)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype="float32").reshape(-1)
        if vector.size != _EMBEDDING_DIMENSION:
            raise ValueError(f"Face embeddings must have dimension {_EMBEDDING_DIMENSION}.")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("The face embedding was empty.")
        return vector / norm

    @staticmethod
    def _validate_subject_id(subject_id: str) -> None:
        if not isinstance(subject_id, str) or not _SUBJECT_ID_RE.fullmatch(subject_id):
            raise ValueError("subject_id must match ^[A-Za-z0-9_.:-]{1,128}$.")

    @staticmethod
    def _safe_extra(extra: dict[str, Any] | None) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in (extra or {}).items():
            if (
                not isinstance(key, str)
                or not _SAFE_EXTRA_VALUE.fullmatch(key)
                or key.lower() in _FORBIDDEN_EXTRA_KEYS
            ):
                continue
            if isinstance(value, str) and _SAFE_EXTRA_VALUE.fullmatch(value):
                safe[key] = value
            elif isinstance(value, (int, float, bool)):
                safe[key] = value
        return safe

    @staticmethod
    def _subject_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "subject_id": row["subject_id"],
            "status": row["status"],
            "created_at": float(row["created_at"]),
            "consent_at": float(row["consent_at"]) if row["consent_at"] is not None else None,
            "failed_attempts": int(row["failed_attempts"]),
            "locked_at": float(row["locked_at"]) if row["locked_at"] is not None else None,
            "extra": json.loads(row["extra"] or "{}"),
        }

    @staticmethod
    def _challenge_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "challenge_id": row["challenge_id"],
            "session_id": row["session_id"],
            "client_ip": row["client_ip"],
            "actions": json.loads(row["actions"]),
            "issued_at": float(row["issued_at"]),
            "expires_at": float(row["expires_at"]),
            "consumed_at": float(row["consumed_at"]) if row["consumed_at"] is not None else None,
        }

    def _insert_audit(
        self,
        connection: sqlite3.Connection,
        event: str,
        subject_id: str | None,
        data: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO audit_log(event, subject_id, created_at, data) VALUES (?, ?, ?, ?)",
            (event, subject_id, self._clock(), json.dumps(data, separators=(",", ":"))),
        )