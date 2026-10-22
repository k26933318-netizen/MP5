"""Enrollment and claimed-subject 1:1 verification workflows."""

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Sequence

import cv2
import numpy as np

from .audit import AuditLogger
from .config import DEFAULT_SETTINGS, Settings
from .liveness import (
    AntiSpoofNotConfigured,
    AntiSpoofEnsemble,
    ChallengeManager,
    evaluate_challenge_actions,
    has_liveness_movement,
)
from .quality import QualityResult, assess
from liveness import (
    ChallengeLivenessDetector,
    ChallengeResult,
    DEFAULT_CONFIG,
    ENABLE_LIVENESS,
    LivenessResult,
)
from liveness.config import (
    DETECT_EVERY_N_FRAMES,
    FAST_MODE,
    PERFORMANCE_DEBUG,
    PERFORMANCE_REPORT_INTERVAL_SECONDS,
)

_logger = logging.getLogger(__name__)


@dataclass
class _BlinkSession:
    detector: ChallengeLivenessDetector
    last_seen: float
    identity_embedding: np.ndarray | None = None
    frame_count: int = 0
    face: Any | None = None


_REASON_MESSAGES = {
    "ANTISPOOF_FAILED": (
        "The live-person check could not confirm that the camera frames show a live face. "
        "This can happen when the image is dark or blurry, or when the prompted movement is not detected. "
        "Try again in even lighting and follow the on-screen prompts."
    ),
    "CHALLENGE_FAILED": (
        "The camera did not detect the requested blink or head movement. "
        "Start a new check, keep your face visible, and perform each prompted action clearly."
    ),
    "BLINK_REQUIRED": "Blink naturally twice in front of the camera, then try again.",
    "INCONSISTENT_FRAMES": "Please keep the same person in view and try again.",
    "DUPLICATE_FACE": "Registration could not be completed.",
    "ALREADY_ENROLLED": "Registration could not be completed.",
    "SUBJECT_NOT_FOUND": "Verification failed. Please try again.",
    "IMPERSONATION_SUSPECTED": "Verification failed. Please try again.",
    "MISMATCH": "Verification failed. Please try again.",
    "LOCKED": "Please contact the polling officer.",
    "ANTISPOOF_NOT_CONFIGURED": "Face verification is temporarily unavailable.",
    "CONSENT_REQUIRED": "Consent is required to enroll.",
    "INSUFFICIENT_FRAMES": "Please capture more frames and try again.",
    "TOO_MANY_FRAMES": "Too many frames were submitted.",
    "NO_FACE": "No face detected - center your face in the camera guide.",
    "FACE_TOO_SMALL": "Face too small - move closer to the camera.",
    "LOW_CONFIDENCE": "Face not clear enough - face the camera directly.",
    "BAD_POSE": "Please look straight at the camera.",
    "BLURRY": "Image too blurry - hold still.",
    "BAD_LIGHTING": "Too dark or too bright - improve the lighting.",
    "MULTIPLE_FACES": "Only one face should be in view.",
    "EYES_CLOSED": "Please keep your eyes open.",
    "NO_MATCH": "No enrolled face matched. Center your face and try again.",
}


class FaceService:
    """Business logic for private biometric enrollment and 1:1 verification."""

    def __init__(
        self,
        engine: Any,
        store: Any,
        antispoof: Any | None = None,
        challenge_manager: ChallengeManager | None = None,
        settings: Settings = DEFAULT_SETTINGS,
        clock: Any = time.time,
    ) -> None:
        self.engine = engine
        self.store = store
        self.audit_logger = AuditLogger(store)
        self.antispoof = antispoof or AntiSpoofEnsemble(settings)
        self.challenge_manager = challenge_manager or ChallengeManager(store, settings, clock)
        self.settings = settings
        self.clock = clock
        self.verify_threshold = self._load_verify_threshold()
        self._blink_lock = threading.RLock()
        self._blink_sessions: dict[str, _BlinkSession] = {}
        self._performance_lock = threading.Lock()
        self._performance_values: dict[str, list[float]] = defaultdict(list)
        self._performance_last_report = time.perf_counter()
        self._performance_frames = 0
        self._performance_snapshot: dict[str, float] = {}
        self._blink_clahe = cv2.createCLAHE(
            clipLimit=DEFAULT_CONFIG.clahe_clip_limit,
            tileGridSize=(DEFAULT_CONFIG.clahe_grid_size, DEFAULT_CONFIG.clahe_grid_size),
        )
        self._blink_gamma_lut = np.clip(
            np.power(np.arange(256, dtype=np.float32) / 255.0, DEFAULT_CONFIG.landmark_gamma) * 255.0,
            0,
            255,
        ).astype("uint8")

    def update_blink_liveness(
        self,
        session_id: str,
        image: np.ndarray,
        face_found: bool,
    ) -> LivenessResult:
        """Update the opt-in action tracker from one verified-session preview frame."""
        now = time.monotonic()
        self._record_preview_frame()
        with self._blink_lock:
            expired = [
                key for key, state in self._blink_sessions.items()
                if now - state.last_seen > DEFAULT_CONFIG.timeout_seconds * 2
            ]
            for key in expired:
                del self._blink_sessions[key]
            tracked = self._blink_sessions.get(session_id)
            if tracked is None:
                tracked = _BlinkSession(ChallengeLivenessDetector(), now)
                self._blink_sessions[session_id] = tracked
            detector = tracked.detector
            if not face_found:
                detector.reset(session_id)
                tracked.last_seen = now
                tracked.identity_embedding = None
                tracked.face = None
                return ChallengeResult(
                    False,
                    0,
                    0.0,
                    "low_quality",
                    "low_quality",
                    "Move closer",
                    0,
                    DEFAULT_CONFIG.challenge_success_count,
                )
            tracked.frame_count += 1
            detect_now = (
                tracked.face is None
                or not FAST_MODE
                or (tracked.frame_count - 1) % DETECT_EVERY_N_FRAMES == 0
            )
            if detect_now:
                stage_started = time.perf_counter()
                analyze_liveness = getattr(self.engine, "analyze_liveness", None) if FAST_MODE else None
                faces = analyze_liveness(image) if analyze_liveness else self.engine.analyze(image)
                ready = (
                    self.engine.face_is_ready(image, faces)
                    if hasattr(self.engine, "face_is_ready")
                    else len(faces) == 1
                )
                self._record_performance("face_detection", stage_started)
                if not ready:
                    detector.reset(session_id)
                    tracked.last_seen = now
                    tracked.identity_embedding = None
                    tracked.face = None
                    return ChallengeResult(
                        False,
                        0,
                        0.0,
                        "low_quality",
                        "low_quality",
                        "Move closer",
                        0,
                        DEFAULT_CONFIG.challenge_success_count,
                    )
                tracked.face = faces[0]
            face = tracked.face
            if face is None:
                return detector.current_result(session_id)
            if not FAST_MODE:
                embedding = self.engine.embed(face, image)
                if (
                    tracked.identity_embedding is not None
                    and float(np.dot(embedding, tracked.identity_embedding)) < self.settings.same_person_min
                ):
                    detector.reset(session_id)
                    tracked.identity_embedding = embedding
                elif tracked.identity_embedding is None:
                    tracked.identity_embedding = embedding
            if detector.current_result(session_id).is_live:
                return detector.current_result(session_id)
            stage_started = time.perf_counter()
            landmarks = self._blink_landmarks(image, face)
            self._record_performance("landmarks", stage_started)
            stage_started = time.perf_counter()
            result = detector.update(image, landmarks, face_id=session_id, timestamp=now)
            self._record_performance("liveness", stage_started)
            if result.is_live and tracked.identity_embedding is None:
                stage_started = time.perf_counter()
                self.engine.analysis.models["recognition"].get(image, face)
                tracked.identity_embedding = self.engine._normalized_embedding(face)
                self._record_performance("face_embedding", stage_started)
            tracked.last_seen = now
            return result

    def _blink_landmarks(self, image: np.ndarray, face: Any) -> np.ndarray | None:
        """Refine blink landmarks on an enhanced crop without altering recognition input."""
        landmarks = getattr(face, "landmark_2d_106", None)
        analysis = getattr(self.engine, "analysis", None)
        models = getattr(analysis, "models", None)
        landmark_model = models.get("landmark_2d_106") if isinstance(models, dict) else None
        if landmarks is None or landmark_model is None or not isinstance(face, dict):
            return landmarks

        frame_height, frame_width = image.shape[:2]
        left, top, right, bottom = (float(value) for value in face.bbox)
        face_width = max(1.0, right - left)
        face_height = max(1.0, bottom - top)
        padding_x = face_width * DEFAULT_CONFIG.landmark_crop_padding_ratio
        padding_y = face_height * DEFAULT_CONFIG.landmark_crop_padding_ratio
        x1 = max(0, min(frame_width, int(np.floor(left - padding_x))))
        y1 = max(0, min(frame_height, int(np.floor(top - padding_y))))
        x2 = max(0, min(frame_width, int(np.ceil(right + padding_x))))
        y2 = max(0, min(frame_height, int(np.ceil(bottom + padding_y))))
        if x2 <= x1 or y2 <= y1:
            return landmarks

        crop = image[y1:y2, x1:x2]
        crop_height, crop_width = crop.shape[:2]
        scale = max(
            1.0,
            DEFAULT_CONFIG.landmark_min_size_px / min(crop_width, crop_height),
        )
        if scale != 1.0:
            crop = cv2.resize(
                crop,
                (round(crop_width * scale), round(crop_height * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        if float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()) < DEFAULT_CONFIG.landmark_gamma_threshold:
            crop = cv2.LUT(crop, self._blink_gamma_lut)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._blink_clahe.apply(lab[:, :, 0])
        enhanced_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        crop_face = face.__class__(face)
        crop_face.bbox = np.asarray(
            [
                (left - x1) * scale,
                (top - y1) * scale,
                (right - x1) * scale,
                (bottom - y1) * scale,
            ],
            dtype="float32",
        )
        refined = np.asarray(landmark_model.get(enhanced_crop, crop_face), dtype="float32")
        mapped = refined.copy()
        mapped[:, :2] = mapped[:, :2] / scale + np.asarray([x1, y1], dtype="float32")
        return mapped

    def get_blink_liveness(self, session_id: str) -> LivenessResult:
        """Return the session's current liveness decision without extending its lifetime."""
        with self._blink_lock:
            tracked = self._blink_sessions.get(session_id)
            if tracked is None:
                return ChallengeResult(False, 0, 0.0, "BLINK TO VERIFY", "checking", "Center your face", 0, DEFAULT_CONFIG.challenge_success_count)
            return tracked.detector.current_result(session_id)

    def reset_blink_liveness(self, session_id: str) -> None:
        """Start a fresh liveness challenge for the same verification session."""
        with self._blink_lock:
            self._blink_sessions.pop(session_id, None)

    def get_performance_debug(self) -> dict[str, float]:
        """Return the latest opt-in two-second stage timing summary."""
        if not PERFORMANCE_DEBUG:
            return {}
        with self._performance_lock:
            return dict(self._performance_snapshot)

    def get_performance_metadata(self) -> dict[str, str]:
        """Describe configured inference models without loading extra models."""
        analysis = getattr(self.engine, "analysis", None)
        models = getattr(analysis, "models", {})
        recognition = models.get("recognition") if isinstance(models, dict) else None
        return {
            "detector": type(getattr(analysis, "det_model", None)).__name__,
            "landmarks": type(models.get("landmark_2d_106")).__name__ if isinstance(models, dict) and models.get("landmark_2d_106") is not None else "unavailable",
            "recognition_model": self.settings.face_model,
            "recognition_backend": type(recognition).__name__ if recognition is not None else "unavailable",
            "num_jitters": "not applicable; InsightFace does not expose num_jitters",
        }

    def record_performance_stage(self, stage: str, started_at: float) -> None:
        """Add an API-owned stage to the shared debug timing summary."""
        self._record_performance(stage, started_at)

    def _record_preview_frame(self) -> None:
        if not PERFORMANCE_DEBUG:
            return
        with self._performance_lock:
            self._performance_frames += 1

    def _record_performance(self, stage: str, started_at: float) -> None:
        if not PERFORMANCE_DEBUG:
            return
        now = time.perf_counter()
        elapsed_ms = (now - started_at) * 1000.0
        with self._performance_lock:
            self._performance_values[stage].append(elapsed_ms)
            window_seconds = now - self._performance_last_report
            if window_seconds < PERFORMANCE_REPORT_INTERVAL_SECONDS:
                return
            snapshot = {
                name: sum(values) / len(values)
                for name, values in self._performance_values.items()
                if values
            }
            snapshot["fps"] = self._performance_frames / window_seconds
            self._performance_snapshot = snapshot
            self._performance_values.clear()
            self._performance_frames = 0
            self._performance_last_report = now

    def get_blink_liveness_debug(self, session_id: str) -> dict[str, float | None]:
        """Return opt-in scalar diagnostics for the session's blink tracker."""
        with self._blink_lock:
            tracked = self._blink_sessions.get(session_id)
            return (
                tracked.detector.debug_info(session_id)
                if tracked is not None
                else {"baseline": None, "threshold": None, "quality_score": 0.0, "fps": 0.0}
            )

    def blink_liveness_matches(
        self,
        session_id: str,
        candidates: Sequence[dict[str, Any]],
    ) -> bool:
        """Bind a live blink track to the identities in the submitted frame burst."""
        with self._blink_lock:
            tracked = self._blink_sessions.get(session_id)
            if tracked is None or tracked.identity_embedding is None or not candidates:
                return False
            similarities = [
                float(np.dot(item["embedding"], tracked.identity_embedding))
                for item in candidates
            ]
            return float(np.mean(similarities)) >= self.settings.same_person_min

    def enroll(
        self,
        subject_id: str,
        frames: Sequence[np.ndarray],
        consent: bool,
        allow_replace: bool = False,
        session_id: str | None = None,
        client_ip: str = "",
        challenge_id: str | None = None,
    ) -> dict[str, Any]:
        """Enroll a subject from quality- and liveness-approved frames."""
        existing = self.store.get_subject(subject_id)
        if existing is not None and not allow_replace:
            return self._result("already_enrolled", "ALREADY_ENROLLED", status=409)
        if not consent:
            return self._result("rejected", "CONSENT_REQUIRED", status=400)
        if not frames:
            return self._result("retry", "INSUFFICIENT_FRAMES", retry_allowed=True)
        if len(frames) > self.settings.enroll_max_frames:
            return self._result("retry", "TOO_MANY_FRAMES", retry_allowed=True)

        unavailable = self._antispoof_unavailable()
        if unavailable:
            return unavailable

        candidates, failures = self._analyze_frames(frames, "enroll")
        if len(candidates) < self.settings.enroll_min_frames:
            return self._quality_retry(failures, len(candidates), self.settings.enroll_min_frames)
        spoof_result = self._score_liveness(candidates)
        if not spoof_result["passed"]:
            self.audit_logger.record("rejected", subject_id, {"reason_code": "ANTISPOOF_FAILED"})
            return self._result("rejected", "ANTISPOOF_FAILED", **spoof_result)
        candidates = self._keep_best(candidates, self.settings.enroll_keep)

        if self.settings.require_enroll_challenge:
            challenge = self._consume_challenge(challenge_id, session_id, client_ip)
            if challenge is None or not evaluate_challenge_actions(
                challenge["actions"], [item["face"] for item in candidates], settings=self.settings
            ):
                return self._failed_attempt(subject_id, "rejected", "CHALLENGE_FAILED")

        candidates = self._drop_outlier_candidates(candidates)
        vectors = [item["embedding"] for item in candidates]
        if len(vectors) < self.settings.enroll_min_frames:
            return self._result("retry", "INCONSISTENT_FRAMES", retry_allowed=True)
        minimum_similarity = self._minimum_pairwise_similarity(vectors)
        if minimum_similarity < self.settings.same_person_min:
            return self._result(
                "retry",
                "INCONSISTENT_FRAMES",
                retry_allowed=True,
                min_pairwise=minimum_similarity,
            )

        duplicate = self._find_duplicate(vectors, subject_id)
        if duplicate is not None:
            self.audit_logger.record("duplicate_face", subject_id, {"duplicate_of": duplicate[0], "score": duplicate[1]})
            return self._result("duplicate_face", "DUPLICATE_FACE", duplicate_of=duplicate[0], status=409)

        try:
            self.store.add_subject_with_embeddings(
                subject_id,
                vectors,
                [item["quality"].score for item in candidates[:len(vectors)]],
                consent=True,
                allow_replace=allow_replace,
            )
        except ValueError as error:
            if str(error) == "already_enrolled":
                return self._result("already_enrolled", "ALREADY_ENROLLED", status=409)
            raise
        self.audit_logger.record("enroll", subject_id, {"frames_used": len(vectors)})
        return self._result("enrolled", "OK", frames_used=len(vectors), status=200)

    def verify(
        self,
        subject_id: str,
        frames: Sequence[np.ndarray],
        challenge_id: str | None,
        session_id: str,
        client_ip: str,
    ) -> dict[str, Any]:
        """Verify only against the claimed subject; never perform 1:N identification."""
        subject = self.store.get_subject(subject_id)
        if subject is None:
            return self._result("rejected", "SUBJECT_NOT_FOUND", status=401)
        if subject["status"] != "active":
            return self._result("locked", "LOCKED", attempts=subject["failed_attempts"], status=423)
        if not frames:
            return self._result("retry", "INSUFFICIENT_FRAMES", retry_allowed=True)
        if len(frames) > self.settings.verify_max_frames:
            return self._result("retry", "TOO_MANY_FRAMES", retry_allowed=True)

        fast_liveness = ENABLE_LIVENESS and FAST_MODE
        if fast_liveness:
            blink_result = self.get_blink_liveness(session_id)
            if not blink_result.is_live:
                return self._result(
                    "retry",
                    "BLINK_REQUIRED",
                    retry_allowed=True,
                    blink_count=blink_result.blink_count,
                    liveness_status=blink_result.status,
                )

        unavailable = self._antispoof_unavailable()
        if unavailable:
            return unavailable

        candidates, failures = self._analyze_frames(frames, "verify")
        minimum_usable = max(
            self.settings.vote_min_frames,
            1 + 2 * self.settings.frames_per_action
            if self.settings.require_challenge and not fast_liveness
            else 2,
        )
        if len(candidates) < minimum_usable:
            return self._quality_retry(failures, len(candidates), minimum_usable)

        if ENABLE_LIVENESS and not fast_liveness:
            blink_result = self.get_blink_liveness(session_id)
            if not blink_result.is_live or not self.blink_liveness_matches(session_id, candidates):
                return self._result(
                    "retry",
                    "BLINK_REQUIRED",
                    retry_allowed=True,
                    blink_count=blink_result.blink_count,
                    liveness_status=blink_result.status if not blink_result.is_live else "SPOOF SUSPECTED",
                )

        spoof_result = self._score_liveness(candidates)
        if not spoof_result["passed"]:
            return self._failed_attempt(subject_id, "rejected", "ANTISPOOF_FAILED", **spoof_result)

        if fast_liveness and not self.blink_liveness_matches(session_id, candidates):
            return self._result(
                "retry",
                "BLINK_REQUIRED",
                retry_allowed=True,
                blink_count=self.get_blink_liveness(session_id).blink_count,
                liveness_status="IDENTITY_CHANGED",
            )

        if self.settings.require_challenge and not fast_liveness:
            challenge = self._consume_challenge(challenge_id, session_id, client_ip)
            if challenge is None or not evaluate_challenge_actions(
                challenge["actions"], [item["face"] for item in candidates], settings=self.settings
            ):
                return self._failed_attempt(
                    subject_id,
                    "rejected",
                    "CHALLENGE_FAILED",
                    liveness_score=spoof_result["liveness_score"],
                )
        elif not self.settings.require_challenge and not fast_liveness and not has_liveness_movement(
            [(item["image"], item["face"]) for item in candidates], self.settings
        ):
            return self._failed_attempt(subject_id, "rejected", "CHALLENGE_FAILED")

        candidates = self._keep_best(candidates, self.settings.verify_keep)
        vectors = [item["embedding"] for item in candidates]
        minimum_similarity = self._minimum_pairwise_similarity(vectors)
        if minimum_similarity < self.settings.same_person_min:
            return self._failed_attempt(
                subject_id,
                "rejected",
                "INCONSISTENT_FRAMES",
                liveness_score=spoof_result["liveness_score"],
                min_pairwise=minimum_similarity,
            )

        match_started = time.perf_counter()
        stored = self.store.get_embeddings(subject_id)
        if not stored:
            return self._result("rejected", "SUBJECT_NOT_FOUND", status=401)
        stored_matrix = np.stack([sample.vector for sample in stored]).astype("float32")
        query_matrix = np.stack(vectors).astype("float32")
        centroid = self._normalize(np.mean(stored_matrix, axis=0))
        sample_scores = np.max(query_matrix @ stored_matrix.T, axis=1)
        centroid_scores = query_matrix @ centroid
        frame_scores = (
            self.settings.fusion_alpha * sample_scores
            + (1.0 - self.settings.fusion_alpha) * centroid_scores
        ).tolist()
        matching_scores = [score for score in frame_scores if score >= self.verify_threshold]
        match_score = float(np.mean(matching_scores)) if matching_scores else float(np.mean(frame_scores))

        runner_up_matches = self.store.search_topk_batch(
            vectors,
            1,
            exclude_subject_id=subject_id,
        )
        runner_up_score = max(
            (matches[0][1] for matches in runner_up_matches if matches),
            default=float("-inf"),
        )
        self._record_performance("matching", match_started)
        if runner_up_score >= match_score - self.settings.impersonation_margin:
            self.audit_logger.record(
                "impersonation_suspected",
                subject_id,
                {"runner_up": runner_up_score, "match_score": match_score},
            )
            return self._failed_attempt(
                subject_id,
                "rejected",
                "IMPERSONATION_SUSPECTED",
                match_score=match_score,
                runner_up=runner_up_score,
                min_pairwise=minimum_similarity,
                liveness_score=spoof_result["liveness_score"],
            )

        if len(matching_scores) < self.settings.vote_min_agree:
            return self._failed_attempt(
                subject_id,
                "rejected",
                "MISMATCH",
                match_score=match_score,
                runner_up=None if runner_up_score == float("-inf") else runner_up_score,
                min_pairwise=minimum_similarity,
                liveness_score=spoof_result["liveness_score"],
                frames_used=len(vectors),
            )

        self.store.reset_failed(subject_id)
        result = self._result(
            "verified",
            "OK",
            match_score=match_score,
            runner_up=None if runner_up_score == float("-inf") else runner_up_score,
            min_pairwise=minimum_similarity,
            liveness_score=spoof_result["liveness_score"],
            frames_used=len(vectors),
            attempts=0,
            status=200,
        )
        self.audit_logger.record("verified", subject_id, {"frames_used": len(vectors), "score": match_score})
        return result

    def identify(
        self,
        frames: Sequence[np.ndarray],
        challenge_id: str | None,
        session_id: str,
        client_ip: str,
    ) -> dict[str, Any]:
        """Identify a live face against active enrolled subjects using FAISS."""
        if not frames:
            return self._result("retry", "INSUFFICIENT_FRAMES", retry_allowed=True)
        if len(frames) > self.settings.verify_max_frames:
            return self._result("retry", "TOO_MANY_FRAMES", retry_allowed=True)

        fast_liveness = ENABLE_LIVENESS and FAST_MODE
        if fast_liveness:
            blink_result = self.get_blink_liveness(session_id)
            if not blink_result.is_live:
                return self._result(
                    "retry",
                    "BLINK_REQUIRED",
                    retry_allowed=True,
                    blink_count=blink_result.blink_count,
                    liveness_status=blink_result.status,
                )

        unavailable = self._antispoof_unavailable()
        if unavailable:
            return unavailable

        candidates, failures = self._analyze_frames(frames, "verify")
        minimum_usable = max(
            self.settings.vote_min_frames,
            1 + 2 * self.settings.frames_per_action
            if self.settings.require_challenge and not fast_liveness
            else 2,
        )
        if len(candidates) < minimum_usable:
            return self._quality_retry(failures, len(candidates), minimum_usable)

        if ENABLE_LIVENESS and not fast_liveness:
            blink_result = self.get_blink_liveness(session_id)
            if not blink_result.is_live or not self.blink_liveness_matches(session_id, candidates):
                return self._result(
                    "retry",
                    "BLINK_REQUIRED",
                    retry_allowed=True,
                    blink_count=blink_result.blink_count,
                    liveness_status=blink_result.status if not blink_result.is_live else "SPOOF SUSPECTED",
                )

        spoof_result = self._score_liveness(candidates)
        if not spoof_result["passed"]:
            return self._result("rejected", "ANTISPOOF_FAILED", **spoof_result)

        if fast_liveness and not self.blink_liveness_matches(session_id, candidates):
            return self._result("retry", "BLINK_REQUIRED", retry_allowed=True)

        if self.settings.require_challenge and not fast_liveness:
            challenge = self._consume_challenge(challenge_id, session_id, client_ip)
            if challenge is None or not evaluate_challenge_actions(
                challenge["actions"], [item["face"] for item in candidates], settings=self.settings
            ):
                return self._result("retry", "CHALLENGE_FAILED", retry_allowed=True)
        elif not self.settings.require_challenge and not fast_liveness and not has_liveness_movement(
            [(item["image"], item["face"]) for item in candidates], self.settings
        ):
            return self._result("retry", "CHALLENGE_FAILED", retry_allowed=True)

        candidates = self._keep_best(candidates, self.settings.verify_keep)
        vectors = [item["embedding"] for item in candidates]
        minimum_similarity = self._minimum_pairwise_similarity(vectors)
        if minimum_similarity < self.settings.same_person_min:
            return self._result("retry", "INCONSISTENT_FRAMES", retry_allowed=True)

        match_started = time.perf_counter()
        active_subject_ids = self.store.list_active_subject_ids()
        if not active_subject_ids:
            return self._result("no_match", "NO_MATCH")

        scores_by_subject: dict[str, list[float]] = {}
        votes_by_subject: Counter[str] = Counter()
        matches_by_frame = self.store.search_topk_batch(
            vectors,
            k=self.store.count_subjects(),
        )
        for matches in matches_by_frame:
            active_matches = [
                (subject_id, score)
                for subject_id, score in matches
                if subject_id in active_subject_ids
            ]
            if not active_matches:
                continue
            votes_by_subject[active_matches[0][0]] += 1
            for subject_id, score in active_matches:
                scores_by_subject.setdefault(subject_id, []).append(score)
            self._record_performance("matching", match_started)

        ranked = sorted(
            [
                (
                    subject_id,
                    float(np.mean(scores)),
                    votes_by_subject[subject_id],
                )
                for subject_id, scores in scores_by_subject.items()
            ],
            key=lambda item: (item[2], item[1]),
            reverse=True,
        )
        if not ranked:
            return self._result("no_match", "NO_MATCH")

        subject_id, match_score, votes = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else None
        if (
            match_score < self.verify_threshold
            or votes < self.settings.vote_min_agree
            or (runner_up is not None and runner_up >= match_score - self.settings.impersonation_margin)
        ):
            return self._result("no_match", "NO_MATCH")

        self.audit_logger.record(
            "identified",
            subject_id,
            {"frames_used": len(vectors), "score": match_score},
        )
        return self._result(
            "identified",
            "OK",
            identified_subject_id=subject_id,
            match_score=match_score,
            frames_used=len(vectors),
            liveness_score=spoof_result["liveness_score"],
        )

    def unlock_subject(self, subject_id: str, reason: str) -> bool:
        """Unlock a subject and append a non-biometric audit event."""
        if not re.fullmatch(r"[A-Z0-9_.:-]{1,64}", reason):
            raise ValueError("Unlock reason must be a non-identifying reason code.")
        updated = self.store.set_status(subject_id, "active")
        if updated:
            self.store.reset_failed(subject_id)
            self.audit_logger.record("unlocked", subject_id, {"reason_code": reason})
        return updated

    def _analyze_frames(self, frames: Sequence[np.ndarray], mode: str) -> tuple[list[dict[str, Any]], list[str]]:
        candidates = []
        failures = []
        for image in frames:
            stage_started = time.perf_counter()
            faces = self.engine.analyze(image)
            self._record_performance("face_detection", stage_started)
            face = faces[0] if faces else None
            stage_started = time.perf_counter()
            quality = assess(image, face, faces, mode, self.settings)
            self._record_performance("quality", stage_started)
            if not quality.ok:
                failures.append(quality.reason_code)
                continue
            stage_started = time.perf_counter()
            embedding = self.engine.embed(face, image)
            self._record_performance("face_embedding", stage_started)
            candidates.append({"image": image, "face": face, "embedding": embedding, "quality": quality})
        return candidates, failures

    def _score_liveness(self, candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        if not self.settings.require_antispoof and not self.antispoof.configured:
            return {"passed": True, "liveness_score": 1.0, "frames_used": len(candidates)}
        stage_started = time.perf_counter()
        live_candidates = []
        scores = []
        for item in candidates:
            score = self.antispoof.is_live(item["image"], item["face"])
            item["antispoof_score"] = score
            scores.append(score)
            live_candidates.append(item)
        mean_score = float(np.mean(scores)) if scores else 0.0
        self._record_performance("antispoof", stage_started)
        passes = bool(
            scores
            and mean_score >= self.settings.antispoof_threshold
            and min(scores) >= self.settings.antispoof_min_frame
        )
        result = {"passed": passes, "liveness_score": mean_score, "frames_used": len(scores)}
        if not passes:
            return result
        candidates[:] = live_candidates
        return result

    def _antispoof_unavailable(self) -> dict[str, Any] | None:
        if self.settings.require_antispoof and not self.antispoof.configured:
            return self._result("retry", "ANTISPOOF_NOT_CONFIGURED", retry_allowed=False, status=503)
        if not self.settings.require_antispoof and not self.antispoof.configured:
            _logger.warning("Passive anti-spoofing is disabled by REQUIRE_ANTISPOOF=false.")
        return None

    def _consume_challenge(
        self,
        challenge_id: str | None,
        session_id: str | None,
        client_ip: str,
    ) -> dict[str, Any] | None:
        if not challenge_id or not session_id:
            return None
        return self.challenge_manager.consume(challenge_id, session_id, client_ip)

    def _failed_attempt(
        self,
        subject_id: str,
        decision: str,
        reason_code: str,
        **details: Any,
    ) -> dict[str, Any]:
        attempts = self.store.increment_failed(subject_id)
        if attempts >= self.settings.max_attempts:
            self.store.set_status(subject_id, "locked")
            self.audit_logger.record("locked", subject_id, {"attempts": attempts})
            return self._result("locked", "LOCKED", attempts=attempts, status=423, **details)
        self.audit_logger.record("rejected", subject_id, {"reason_code": reason_code, "attempts": attempts})
        return self._result(decision, reason_code, attempts=attempts, status=401, **details)

    def _quality_retry(
        self,
        failures: Sequence[str],
        passing_count: int,
        required_count: int,
    ) -> dict[str, Any]:
        most_common = Counter(failures).most_common(1)
        reason = most_common[0][0] if most_common else "INSUFFICIENT_FRAMES"
        return self._result(
            "retry",
            reason,
            retry_allowed=True,
            frames_used=passing_count,
            frames_required=required_count,
        )

    def _find_duplicate(self, vectors: Sequence[np.ndarray], subject_id: str) -> tuple[str, float] | None:
        best = None
        for vector in vectors:
            results = self.store.search_topk(vector, 1, exclude_subject_id=subject_id)
            if results and (best is None or results[0][1] > best[1]):
                best = results[0]
        if best is not None and best[1] >= self.settings.duplicate_threshold:
            return best
        return None

    def _load_verify_threshold(self) -> float:
        if "VERIFY_THRESHOLD" in os.environ or "MATCH_THRESHOLD" in os.environ:
            return self.settings.verify_threshold
        threshold_path = Path(self.settings.face_data_dir) / "threshold.json"
        if threshold_path.is_file():
            try:
                value = json.loads(threshold_path.read_text(encoding="utf-8"))["verify_threshold"]
                return float(value)
            except (OSError, ValueError, KeyError, TypeError):
                _logger.warning("Ignoring invalid calibrated threshold file: %s", threshold_path)
        return self.settings.verify_threshold

    @staticmethod
    def _keep_best(candidates: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        return sorted(candidates, key=lambda item: item["quality"].score, reverse=True)[:count]

    def _drop_outlier_candidates(self, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(candidates) < 3:
            return list(candidates)
        matrix = np.stack([item["embedding"] for item in candidates])
        similarities = matrix @ matrix.T
        means = (similarities.sum(axis=1) - np.diag(similarities)) / (len(candidates) - 1)
        return [item for item, mean in zip(candidates, means) if mean >= self.settings.outlier_min_sim]

    @staticmethod
    def _minimum_pairwise_similarity(vectors: Sequence[np.ndarray]) -> float:
        if len(vectors) < 2:
            return 0.0
        matrix = np.stack(vectors)
        similarities = matrix @ matrix.T
        return float(np.min(similarities[np.triu_indices(len(vectors), k=1)]))

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype="float32")
        norm = float(np.linalg.norm(values))
        if norm <= 1e-12:
            raise ValueError("The face embedding was empty.")
        return values / norm

    @staticmethod
    def _result(
        decision: str,
        reason_code: str,
        status: int = 200,
        retry_allowed: bool = False,
        **details: Any,
    ) -> dict[str, Any]:
        return {
            "decision": decision,
            "reason_code": reason_code,
            "public_message": _REASON_MESSAGES.get(reason_code, ""),
            "retry_allowed": retry_allowed,
            "status": status,
            **details,
        }