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
    confirmation_detector: object | None = None

    def _filter_invalid_candidates(
        self, image: np.ndarray, boxes: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop detector false positives that fail basic face-image quality checks."""
        if len(boxes) <= 1:
            return boxes, landmarks

        if self.confirmation_detector is not None:
            confirmed_boxes, _ = self.confirmation_detector.detect(image, max_num=0)
            if len(confirmed_boxes) == 1:
                # Keep RetinaFace's box and landmarks for all later checks, while
                # using the independent detector to resolve a false multi-face count.
                best = int(np.argmax(boxes[:, 4]))
                return boxes[[best]], landmarks[[best]]

        # RetinaFace can return a nested box for the same face (for example, a
        # second box around only the eyes). Suppress those before counting people.
        ordered = np.argsort(boxes[:, 4])[::-1]
        kept = []
        for index in ordered:
            candidate = boxes[index, :4]
            x1, y1, x2, y2 = candidate
            candidate_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            duplicate = False
            for accepted_index in kept:
                accepted = boxes[accepted_index, :4]
                ix1, iy1 = max(x1, accepted[0]), max(y1, accepted[1])
                ix2, iy2 = min(x2, accepted[2]), min(y2, accepted[3])
                intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                accepted_area = max(0.0, accepted[2] - accepted[0]) * max(0.0, accepted[3] - accepted[1])
                smaller_area = min(candidate_area, accepted_area)
                if smaller_area <= 0 or intersection / smaller_area < 0.80:
                    continue
                candidate_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                accepted_center = ((accepted[0] + accepted[2]) / 2, (accepted[1] + accepted[3]) / 2)
                center_close = (
                    abs(candidate_center[0] - accepted_center[0]) <= 0.25 * max(x2 - x1, accepted[2] - accepted[0])
                    and abs(candidate_center[1] - accepted_center[1]) <= 0.25 * max(y2 - y1, accepted[3] - accepted[1])
                )
                if center_close:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(int(index))
        boxes, landmarks = boxes[kept], landmarks[kept]

        plausible = []
        for index, (box, points) in enumerate(zip(boxes, landmarks)):
            try:
                self._validate_face(image, box, points)
            except HTTPException:
                continue
            plausible.append(index)

        if len(plausible) == 0:
            # Keep the strongest candidate so the normal quality checks can give
            # the voter a useful instruction instead of reporting multiple faces.
            plausible = [int(np.argmax(boxes[:, 4]))]
        return boxes[plausible], landmarks[plausible]

    def detect_faces(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes, landmarks = self.detector.detect(image)
        boxes, landmarks = self._filter_invalid_candidates(image, boxes, landmarks)
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

    def assess_position(
        self,
        image: np.ndarray,
        require_frontal: bool = True,
        require_background: bool = True,
        relaxed_position: bool = False,
    ) -> dict:
        boxes, landmarks = self.detector.detect(image)
        boxes, landmarks = self._filter_invalid_candidates(image, boxes, landmarks)
        if len(boxes) == 0:
            _log_face_failure(False, 0, None, "face_not_detected")
            return {
                "ready": False,
                "state": "red",
                "face": None,
                "faceCount": 0,
                "message": "No face detected. Move your face into the guide.",
            }
        if len(boxes) != 1:
            _log_face_failure(True, len(boxes), [float(value) for value in boxes[0][:4]], "multiple_faces_detected")
            selected = int(np.argmax(boxes[:, 4]))
            box = boxes[selected]
            face = {
                "bbox": [float(value) for value in box[:4]],
                "frameWidth": int(image.shape[1]),
                "frameHeight": int(image.shape[0]),
            }
            return {
                "ready": False,
                "state": "red",
                "face": face,
                "faceCount": len(boxes),
                "message": f"Detected {len(boxes)} possible faces. Keep only yourself in the camera view.",
            }
        box, points = boxes[0], landmarks[0]
        if float(box[4]) < settings.min_detection_score:
            _log_face_failure(True, 1, [float(value) for value in box[:4]], "low_detector_confidence")
            return {
                "ready": False,
                "state": "red",
                "face": {
                    "bbox": [float(value) for value in box[:4]],
                    "frameWidth": int(image.shape[1]),
                    "frameHeight": int(image.shape[0]),
                },
                "faceCount": 1,
                "message": "Face detected. Move closer and keep it steady in the guide.",
            }
        face = {
            "bbox": [float(value) for value in box[:4]],
            "frameWidth": int(image.shape[1]),
            "frameHeight": int(image.shape[0]),
        }
        try:
            self._validate_face(image, box, points, relaxed_position=relaxed_position)
            assess_face(
                image, box, points, require_frontal=require_frontal,
                require_background=require_background, relaxed_position=relaxed_position,
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            _log_face_failure(
                True,
                1,
                face["bbox"],
                str(detail.get("code", "face_validation_failed")).lower(),
            )
            detail["face"] = face
            detail["faceCount"] = 1
            raise
        return {
            "ready": True,
            "state": "yellow",
            "message": "Hold still while your face position is checked.",
            "face": face,
            "faceCount": 1,
        }

    def analyze_frames(self, images: list[np.ndarray], mode: str, action: str | None = None) -> dict:
        stage_size = settings.challenge_frames_per_stage
        if mode == "enroll":
            if len(images) != 1 and not 3 <= len(images) <= 8:
                raise HTTPException(status_code=400, detail={
                    "code": "INVALID_FRAME_COUNT",
                    "message": "Submit one image or three to eight face frames to enroll.",
                })
            analysis = [self._analyze_frame(image, True, require_background=False, relaxed_position=True) for image in images]
            if len(analysis) > 1:
                self._require_stable(analysis, min(0.25, settings.face_stability_tolerance * 1.5))
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
                    "message": "We could not confirm a live face. Hold still to avoid blur, center your whole face in the guide, use steady lighting, and try again.",
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
            "checks": {"position": True, "quality": True,
                       "stable": mode == "verify" or len(analysis) > 1,
                       "head_movement": mode == "verify", "presentation_attack": True,
                       "antispoof_score": score,
                       **({"background": True} if mode == "verify" else {})},
        }

    def _analyze_frame(
        self,
        image: np.ndarray,
        require_frontal: bool,
        require_background: bool = True,
        relaxed_position: bool = False,
    ) -> dict:
        boxes, landmarks = self.detector.detect(image)
        boxes, landmarks = self._filter_invalid_candidates(image, boxes, landmarks)
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
            self._validate_face(image, box, points, relaxed_position=relaxed_position)
            metrics = assess_face(
                image, box, points, require_frontal=require_frontal,
                require_background=require_background, relaxed_position=relaxed_position,
            )
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

    def _validate_face(
        self, image: np.ndarray, box: np.ndarray, points: np.ndarray, relaxed_position: bool = False
    ) -> None:
        from app.utils.image_utils import validate_face_quality
        # Position can be relaxed for enrollment, but blur degrades both matching
        # and passive liveness and must use the same minimum in every workflow.
        validate_face_quality(image, box, points, min_blur=25)

    def _position_in_guide(self, frames: list[dict], tolerance_scale: float = 1.5) -> bool:
        return all(
            abs(item["metrics"]["center_x"] - 0.5) <= settings.face_center_x_tolerance * tolerance_scale
            and abs(item["metrics"]["center_y"] - 0.46) <= settings.face_center_y_tolerance * tolerance_scale
            for item in frames
        )

    def _require_stable(self, frames: list[dict], tolerance: float | None = None) -> None:
        relaxed = tolerance is not None
        tolerance = settings.face_stability_tolerance if tolerance is None else tolerance
        centers = np.asarray([
            [item["metrics"]["center_x"], item["metrics"]["center_y"], item["metrics"]["width_fraction"]]
            for item in frames
        ])
        tolerance_scale = 2.0 if relaxed else 1.5
        if (not self._position_in_guide(frames, tolerance_scale) or
                float(np.max(np.ptp(centers[:, :2], axis=0))) > tolerance or
                float(np.ptp(centers[:, 2])) > tolerance):
            raise HTTPException(status_code=422, detail={
                "code": "UNSTABLE_POSITION",
                "message": "Hold your face steady inside the guide and capture again.",
            })


def create_runtime() -> FaceRecognitionRuntime:
    from insightface import model_zoo
    import onnxruntime as ort
    import tempfile
    import zipfile

    from app.models.face_detector import RetinaFaceDetector
    from app.models.face_recognizer import ArcFaceRecognizer

    # Limit CPU threads and avoid the default memory arena on small hosts.
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.enable_cpu_mem_arena = False
    session_options.enable_mem_pattern = False

    if settings.insightface_model_name == "buffalo_sc":
        model_directory = Path(tempfile.gettempdir()) / "voting-system-face-models" / "buffalo_sc"
        arcface_path = model_directory / "w600k_mbf.onnx"
        confirmation_path = model_directory / "det_500m.onnx"
        if not arcface_path.is_file() or not confirmation_path.is_file():
            expected_archive_hash = "da14772719ba896e12fec034ee12df63e70310a6fe91143e09ad54d10cbf143e"
            archive = settings.arcface_model_archive_path
            if not archive.is_file() or _model_hash(archive) != expected_archive_hash[:16]:
                raise RuntimeError("The bundled buffalo_sc model archive is missing or failed its SHA-256 check.")
            model_directory.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as model_archive:
                archive_names = set(model_archive.namelist())
                for source_name, target_path in (
                    ("buffalo_sc/w600k_mbf.onnx", arcface_path),
                    ("buffalo_sc/det_500m.onnx", confirmation_path),
                ):
                    if source_name not in archive_names:
                        raise RuntimeError("The buffalo_sc model archive has unexpected contents.")
                    if not target_path.is_file():
                        with model_archive.open(source_name) as source, target_path.open("wb") as target:
                            target.write(source.read())
    else:
        model_directory = settings.insightface_model_root / "models" / settings.insightface_model_name
        arcface_filename = "w600k_mbf.onnx" if settings.insightface_model_name == "buffalo_s" else "w600k_r50.onnx"
        arcface_path = model_directory / arcface_filename
    if not arcface_path.is_file():
        raise RuntimeError(f"ArcFace recognition model is missing: {arcface_path}")
    recognition_model = model_zoo.get_model(
        str(arcface_path), providers=["CPUExecutionProvider"], sess_options=session_options
    )
    if recognition_model is None:
        raise RuntimeError("The configured ArcFace recognition model could not be loaded.")
    recognition_model.prepare(ctx_id=-1)
    detector = RetinaFaceDetector(settings.retinaface_model_path, settings.min_detection_score)
    recognizer = ArcFaceRecognizer(recognition_model)
    confirmation_detector = None
    if settings.insightface_model_name == "buffalo_sc":
        confirmation_detector = model_zoo.get_model(
            str(confirmation_path), providers=["CPUExecutionProvider"], sess_options=session_options
        )
        if confirmation_detector is None:
            raise RuntimeError("The compact secondary face-count model could not be loaded.")
        confirmation_detector.prepare(ctx_id=-1)
    recognizer_path = Path(recognizer.model.model_file)
    detector_hash = _model_hash(settings.retinaface_model_path)
    recognizer_hash = _model_hash(recognizer_path)
    version = (
        f"RetinaFace:MobileNet025:{detector_hash}"
        f"+{type(recognizer.model).__name__}:{recognizer_hash}"
    )
    return FaceRecognitionRuntime(detector, recognizer, version, confirmation_detector)


def _model_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]
