"""Enrollment and claimed-subject 1:1 verification workflows."""

from collections import Counter
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence

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

_logger = logging.getLogger(__name__)
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

        unavailable = self._antispoof_unavailable()
        if unavailable:
            return unavailable

        candidates, failures = self._analyze_frames(frames, "verify")
        minimum_usable = max(self.settings.vote_min_frames, 1 + 2 * self.settings.frames_per_action if self.settings.require_challenge else 2)
        if len(candidates) < minimum_usable:
            return self._quality_retry(failures, len(candidates), minimum_usable)

        spoof_result = self._score_liveness(candidates)
        if not spoof_result["passed"]:
            return self._failed_attempt(subject_id, "rejected", "ANTISPOOF_FAILED", **spoof_result)

        if self.settings.require_challenge:
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
        elif not has_liveness_movement(
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

        stored = self.store.get_embeddings(subject_id)
        if not stored:
            return self._result("rejected", "SUBJECT_NOT_FOUND", status=401)
        stored_vectors = [sample.vector for sample in stored]
        centroid = self._normalize(np.mean(stored_vectors, axis=0))
        frame_scores = []
        for vector in vectors:
            sample_score = max(float(np.dot(vector, saved)) for saved in stored_vectors)
            centroid_score = float(np.dot(vector, centroid))
            frame_scores.append(
                self.settings.fusion_alpha * sample_score
                + (1.0 - self.settings.fusion_alpha) * centroid_score
            )
        matching_scores = [score for score in frame_scores if score >= self.verify_threshold]
        match_score = float(np.mean(matching_scores)) if matching_scores else float(np.mean(frame_scores))

        runner_up_score = float("-inf")
        for vector in vectors:
            other = self.store.search_topk(vector, 1, exclude_subject_id=subject_id)
            if other:
                runner_up_score = max(runner_up_score, other[0][1])
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

        unavailable = self._antispoof_unavailable()
        if unavailable:
            return unavailable

        candidates, failures = self._analyze_frames(frames, "verify")
        minimum_usable = max(
            self.settings.vote_min_frames,
            1 + 2 * self.settings.frames_per_action if self.settings.require_challenge else 2,
        )
        if len(candidates) < minimum_usable:
            return self._quality_retry(failures, len(candidates), minimum_usable)

        spoof_result = self._score_liveness(candidates)
        if not spoof_result["passed"]:
            return self._result("rejected", "ANTISPOOF_FAILED", **spoof_result)

        if self.settings.require_challenge:
            challenge = self._consume_challenge(challenge_id, session_id, client_ip)
            if challenge is None or not evaluate_challenge_actions(
                challenge["actions"], [item["face"] for item in candidates], settings=self.settings
            ):
                return self._result("retry", "CHALLENGE_FAILED", retry_allowed=True)
        elif not has_liveness_movement(
            [(item["image"], item["face"]) for item in candidates], self.settings
        ):
            return self._result("retry", "CHALLENGE_FAILED", retry_allowed=True)

        candidates = self._keep_best(candidates, self.settings.verify_keep)
        vectors = [item["embedding"] for item in candidates]
        minimum_similarity = self._minimum_pairwise_similarity(vectors)
        if minimum_similarity < self.settings.same_person_min:
            return self._result("retry", "INCONSISTENT_FRAMES", retry_allowed=True)

        subject_count = self.store.count_subjects()
        if subject_count == 0:
            return self._result("no_match", "NO_MATCH")

        scores_by_subject: dict[str, list[float]] = {}
        votes_by_subject: Counter[str] = Counter()
        for vector in vectors:
            matches = self.store.search_topk(vector, k=subject_count)
            active_matches = [
                (subject_id, score)
                for subject_id, score in matches
                if (subject := self.store.get_subject(subject_id)) is not None
                and subject["status"] == "active"
            ]
            if not active_matches:
                continue
            votes_by_subject[active_matches[0][0]] += 1
            for subject_id, score in active_matches:
                scores_by_subject.setdefault(subject_id, []).append(score)

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
            faces = self.engine.analyze(image)
            face = faces[0] if faces else None
            quality = assess(image, face, faces, mode, self.settings)
            if not quality.ok:
                failures.append(quality.reason_code)
                continue
            embedding = self.engine.embed(face, image)
            candidates.append({"image": image, "face": face, "embedding": embedding, "quality": quality})
        return candidates, failures

    def _score_liveness(self, candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        if not self.settings.require_antispoof and not self.antispoof.configured:
            return {"passed": True, "liveness_score": 1.0, "frames_used": len(candidates)}
        live_candidates = []
        scores = []
        for item in candidates:
            score = self.antispoof.is_live(item["image"], item["face"])
            item["antispoof_score"] = score
            scores.append(score)
            live_candidates.append(item)
        mean_score = float(np.mean(scores)) if scores else 0.0
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