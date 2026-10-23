from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from app.config import settings
from app.services.antispoof import anti_spoof
from app.services.face_quality import assess_face
from app.utils.image_utils import validate_face_quality

logger = logging.getLogger("face-recognition")


def _log_face_failure(face_detected: bool, face_count: int, bbox: list[float] | None, reason: str) -> None:
    if settings.face_diagnostics_enabled:
        logger.info("face_validation %s", json.dumps({
            "face_detected": face_detected,
            "face_count": face_count,
            "face_bbox": bbox,
            "valid_background_pixels": None,
            "near_white_ratio": None,
            "configured_threshold": settings.background_min_white_fraction,
            "uniformity_stddev": None,
            "failure_condition": reason,
        }, sort_keys=True))


@dataclass
class FaceRecognitionRuntime:
    detector: object
    recognizer: object
    model_version: str

    def detect_faces(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes, landmarks = self.detector.detect(image)
        count = len(boxes)
        if count == 0:
            raise HTTPException(status_code=422, detail={
                "code": "NO_FACE",
                "message": "No face was detected. Center your face and try again.",
            })
        if count != 1:
            raise HTTPException(status_code=422, detail={
                "code": "MULTIPLE_FACES",
                "message": "Only one face may be in the image.",
            })

        box, points = boxes[0], landmarks[0]
        if float(box[4]) < settings.min_detection_score:
            raise HTTPException(status_code=422, detail={
                "code": "LOW_IMAGE_QUALITY",
                "message": "The face could not be detected clearly. Please try again.",
            })
        validate_face_quality(image, box, points)
        return box, points

    def embedding(self, image: np.ndarray) -> list[float]:
        _, landmarks = self.detect_faces(image)
        return self.recognizer.embed(image, landmarks).astype(float).tolist()

    def assess_position(self, image: np.ndarray, require_frontal: bool = True) -> dict:
        boxes, landmarks = self.detector.detect(image)
        if len(boxes) == 0:
            _log_face_failure(False, 0, None, "face_not_detected")
            raise HTTPException(status_code=422, detail={
                "code": "NO_FACE",
                "message": "No face detected. Move your face into the guide.",
                "state": "red",
            })
        if len(boxes) != 1:
            _log_face_failure(True, len(boxes), [float(value) for value in boxes[0][:4]], "multiple_faces_detected")
            raise HTTPException(status_code=422, detail={
                "code": "MULTIPLE_FACES",
                "message": "Only one face should be in view.",
                "state": "red",
            })
        box, points = boxes[0], landmarks[0]
        if float(box[4]) < settings.min_detection_score:
            _log_face_failure(True, 1, [float(value) for value in box[:4]], "low_detector_confidence")
            raise HTTPException(status_code=422, detail={
                "code": "LOW_CONFIDENCE",
                "message": "Face could not be detected clearly. Try again.",
                "state": "red",
            })
        face = {
            "bbox": [float(value) for value in box[:4]],
            "frameWidth": int(image.shape[1]),
            "frameHeight": int(image.shape[0]),
        }
        try:
            self._validate_face(image, box, points)
            assess_face(image, box, points, require_frontal=require_frontal)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            _log_face_failure(
                True,
                1,
                face["bbox"],
                str(detail.get("code", "face_validation_failed")).lower(),
            )
            detail["face"] = face
            if detail.get("code") == "IMAGE_CONDITIONS_UNSUITABLE":
                detail["message"] = (
                    "Image conditions are not suitable. Please adjust your surroundings and try again."
                )
            raise
        return {
            "ready": True,
            "state": "yellow",
            "message": "Hold still while your face position is checked.",
            "face": face,
        }

    def analyze_frames(self, images: list[np.ndarray], mode: str, action: str | None = None) -> dict:
        stage_size = settings.challenge_frames_per_stage
        if mode == "enroll":
            if len(images) < 5 or len(images) > 8:
                raise HTTPException(status_code=400, detail={
                    "code": "INVALID_FRAME_COUNT",
                    "message": "Capture at least five steady frames to enroll.",
                })
            analysis = [self._analyze_frame(image, True) for image in images]
            self._require_stable(analysis)
            usable = analysis
        elif mode == "verify":
            if action not in {"turn_left", "turn_right"} or len(images) != stage_size * 3:
                raise HTTPException(status_code=400, detail={
                    "code": "INVALID_CHALLENGE_FRAMES",
                    "message": "The liveness challenge response is incomplete.",
                })
            analysis = []
            for index, image in enumerate(images):
                stage = index // stage_size
                analysis.append(self._analyze_frame(image, stage != 1))
            baseline = analysis[:stage_size]
            turn = analysis[stage_size:stage_size * 2]
            returned = analysis[stage_size * 2:]
            self._require_stable(baseline)
            self._require_stable(returned)
            if not self._position_in_guide(turn):
                raise HTTPException(status_code=422, detail={
                    "code": "FACE_OFF_CENTER",
                    "message": "Keep your face inside the guide while turning.",
                })
            start_yaw = float(np.median([item["metrics"]["yaw"] for item in baseline]))
            turn_yaw = float(np.median([item["metrics"]["yaw"] for item in turn]))
            end_yaw = float(np.median([item["metrics"]["yaw"] for item in returned]))
            delta = turn_yaw - start_yaw
            expected_sign = 1 if action == "turn_left" else -1
            turn_yaws = [item["metrics"]["yaw"] for item in turn]
            directional_progress = (turn_yaws[-1] - turn_yaws[0]) * expected_sign
            movement_spread = float(np.ptp(turn_yaws))
            minimum_progress = settings.challenge_yaw_delta * 0.25
            if (delta * expected_sign < settings.challenge_yaw_delta or
                    directional_progress < minimum_progress or
                    movement_spread < minimum_progress):
                raise HTTPException(status_code=403, detail={
                    "code": "LIVENESS_CHALLENGE_FAILED",
                    "message": "A sustained movement in the requested direction was not detected. Please turn slowly and retry.",
                })
            if abs(end_yaw - start_yaw) > settings.challenge_return_tolerance:
                raise HTTPException(status_code=403, detail={
                    "code": "LIVENESS_CHALLENGE_FAILED",
                    "message": "Return your face to the center before continuing.",
                })
            usable = baseline + returned
        else:
            raise ValueError("Unsupported face analysis mode.")

        live, score = anti_spoof.verify([(item["image"], item["box"]) for item in analysis])
        if not live:
            raise HTTPException(status_code=403, detail={
                "code": "PRESENTATION_ATTACK_SUSPECTED",
                "message": "The presentation-attack check did not pass. Please try again.",
            })
        vectors = [self.recognizer.embed(item["image"], item["landmarks"]) for item in usable]
        embedding = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-6 or not np.isfinite(embedding).all():
            raise HTTPException(status_code=422, detail={
                "code": "INCONSISTENT_FRAMES",
                "message": "Face images were inconsistent. Please hold still and retry.",
            })
        return {
            "embedding": (embedding / norm).astype(float).tolist(),
            "dimension": 512,
            "model_version": self.model_version,
            "checks": {"position": True, "background": True, "quality": True, "stable": True,
                       "head_movement": mode == "verify", "presentation_attack": True,
                       "antispoof_score": score},
        }

    def _analyze_frame(self, image: np.ndarray, require_frontal: bool) -> dict:
        boxes, landmarks = self.detector.detect(image)
        if len(boxes) == 0:
            _log_face_failure(False, 0, None, "face_not_detected")
            raise HTTPException(status_code=422, detail={
                "code": "NO_FACE",
                "message": "No face detected in every required frame.",
            })
        if len(boxes) != 1:
            _log_face_failure(True, len(boxes), [float(value) for value in boxes[0][:4]], "multiple_faces_detected")
            raise HTTPException(status_code=422, detail={
                "code": "MULTIPLE_FACES",
                "message": "Only one face should be visible in every required frame.",
            })
        box, points = boxes[0], landmarks[0]
        if float(box[4]) < settings.min_detection_score:
            _log_face_failure(True, 1, [float(value) for value in box[:4]], "low_detector_confidence")
            raise HTTPException(status_code=422, detail={
                "code": "LOW_CONFIDENCE",
                "message": "Face could not be detected clearly in every required frame.",
            })
        try:
            self._validate_face(image, box, points)
            metrics = assess_face(image, box, points, require_frontal=require_frontal)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            _log_face_failure(
                True,
                1,
                [float(value) for value in box[:4]],
                str(detail.get("code", "face_validation_failed")).lower(),
            )
            raise
        return {"image": image, "box": box, "landmarks": points, "metrics": metrics}

    def _validate_face(self, image: np.ndarray, box: np.ndarray, points: np.ndarray) -> None:
        from app.utils.image_utils import validate_face_quality
        validate_face_quality(image, box, points)

    def _position_in_guide(self, frames: list[dict]) -> bool:
        return all(
            abs(item["metrics"]["center_x"] - 0.5) <= settings.face_center_x_tolerance * 1.5
            and abs(item["metrics"]["center_y"] - 0.46) <= settings.face_center_y_tolerance * 1.5
            for item in frames
        )

    def _require_stable(self, frames: list[dict]) -> None:
        centers = np.asarray([
            [item["metrics"]["center_x"], item["metrics"]["center_y"], item["metrics"]["width_fraction"]]
            for item in frames
        ])
        if (not self._position_in_guide(frames) or
                float(np.max(np.ptp(centers[:, :2], axis=0))) > settings.face_stability_tolerance or
                float(np.ptp(centers[:, 2])) > settings.face_stability_tolerance):
            raise HTTPException(status_code=422, detail={
                "code": "UNSTABLE_POSITION",
                "message": "Hold your face steady inside the guide and capture again.",
            })


def create_runtime() -> FaceRecognitionRuntime:
    from insightface.app import FaceAnalysis

    from app.models.face_detector import InsightFaceDetector
    from app.models.face_recognizer import ArcFaceRecognizer

    analysis = FaceAnalysis(
        name=settings.insightface_model_name,
        root=str(settings.insightface_model_root),
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    analysis.prepare(
        ctx_id=-1,
        det_size=(settings.detection_size, settings.detection_size),
        det_thresh=settings.min_detection_score,
    )
    detector = InsightFaceDetector(analysis.models.get("detection"))
    recognizer = ArcFaceRecognizer(analysis.models.get("recognition"))
    detector_path = Path(detector.model.model_file)
    recognizer_path = Path(recognizer.model.model_file)
    detector_hash = _model_hash(detector_path)
    recognizer_hash = _model_hash(recognizer_path)
    version = (
        f"InsightFace:{settings.insightface_model_name}"
        f"+{type(detector.model).__name__}:{detector_hash}"
        f"+{type(recognizer.model).__name__}:{recognizer_hash}"
    )
    return FaceRecognitionRuntime(detector, recognizer, version)


def _model_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]
