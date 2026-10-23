"""InsightFace model wrapper and bounded image decoding."""

import base64
import binascii
from typing import Any, Callable

import cv2
import numpy as np

from .config import DEFAULT_SETTINGS, Settings
from liveness.config import DETECTOR_MAX_WIDTH


def _runtime_options() -> tuple[list[str], int]:
    """Use the CPU provider for predictable operation on low-power machines."""
    return ["CPUExecutionProvider"], -1


def _create_face_analysis(
    settings: Settings,
    analysis_factory: Callable[..., Any] | None = None,
    allowed_modules: list[str] | None = None,
) -> Any:
    """Create InsightFace with only configured analysis modules and providers."""
    if analysis_factory is None:
        from insightface.app import FaceAnalysis

        analysis_factory = FaceAnalysis
    providers, ctx_id = _runtime_options()
    if allowed_modules is None:
        allowed_modules = ["detection", "landmark_2d_106", "recognition"]
        if settings.pose_gate:
            allowed_modules.append("landmark_3d_68")
    analysis = analysis_factory(
        name=settings.face_model,
        allowed_modules=allowed_modules,
        providers=providers,
    )
    analysis.prepare(ctx_id=ctx_id, det_size=(settings.face_det_size, settings.face_det_size))
    return analysis


class FaceEngine:
    """Injectable InsightFace analysis engine; no model loads at module import."""

    def __init__(
        self,
        settings: Settings = DEFAULT_SETTINGS,
        analysis: Any | None = None,
        analysis_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self._analysis = analysis
        self._analysis_factory = analysis_factory
        self._detection_analysis: Any | None = None

    @property
    def analysis(self) -> Any:
        if self._analysis is None:
            self._analysis = _create_face_analysis(self.settings, self._analysis_factory)
        return self._analysis

    def warm_up(self) -> None:
        """Initialize each configured inference model before a camera session."""
        from insightface.app.common import Face

        image = np.zeros((self.settings.face_det_size, self.settings.face_det_size, 3), dtype=np.uint8)
        self.analysis.get(image)
        models = getattr(self.analysis, "models", {})
        if not isinstance(models, dict):
            return
        side = self.settings.face_det_size
        face = Face(
            bbox=np.asarray([side * 0.2, side * 0.2, side * 0.8, side * 0.8], dtype="float32"),
            kps=np.asarray(
                [[side * 0.35, side * 0.4], [side * 0.65, side * 0.4],
                 [side * 0.5, side * 0.55], [side * 0.4, side * 0.7],
                 [side * 0.6, side * 0.7]],
                dtype="float32",
            ),
        )
        for task in ("landmark_2d_106", "recognition"):
            model = models.get(task)
            if model is not None:
                model.get(image, face)

    def analyze(self, image: np.ndarray) -> list[Any]:
        """Detect every face in one model pass, sorted by descending area."""
        self._validate_image(image)
        faces = self.analysis.get(image)
        return sorted(faces or [], key=lambda face: self._face_area(face.bbox), reverse=True)

    # --- speed-up (added) ---
    def analyze_liveness(self, image: np.ndarray) -> list[Any]:
        """Run downscaled face detection only; landmarks are refined on the face crop later."""
        self._validate_image(image)
        from insightface.app.common import Face

        height, width = image.shape[:2]
        scale = min(1.0, DETECTOR_MAX_WIDTH / max(width, 1))
        detector_image = image
        if scale < 1.0:
            detector_image = cv2.resize(
                image,
                (DETECTOR_MAX_WIDTH, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        boxes, keypoints = self.analysis.det_model.detect(
            detector_image,
            max_num=0,
            metric="default",
        )
        if boxes is None or len(boxes) == 0:
            return []
        inverse_scale = 1.0 / scale
        faces = []
        for index, box in enumerate(np.asarray(boxes)):
            bbox = np.asarray(box[:4], dtype=np.float32) * inverse_scale
            kps = None
            if keypoints is not None and index < len(keypoints):
                kps = np.asarray(keypoints[index], dtype=np.float32) * inverse_scale
            faces.append(
                Face(
                    bbox=bbox,
                    kps=kps,
                    det_score=float(box[4]),
                )
            )
        return sorted(faces, key=lambda face: self._face_area(face.bbox), reverse=True)

    def face_is_ready(self, image: np.ndarray, faces: list[Any]) -> bool:
        """Apply the existing one-face confidence, size, and centering requirements."""
        if len(faces) != 1:
            return False
        height, width = image.shape[:2]
        face = faces[0]
        left, top, right, bottom = (float(value) for value in face.bbox)
        face_width = max(0.0, right - left)
        face_height = max(0.0, bottom - top)
        settings = self.settings
        return bool(
            float(face.det_score) >= settings.detect_min_score
            and face_width >= width * settings.detect_min_width_ratio
            and face_height >= height * settings.detect_min_height_ratio
            and face_width * face_height / (width * height) >= settings.detect_min_area_ratio
            and settings.detect_center_x_min <= (left + right) / (2 * width) <= settings.detect_center_x_max
            and settings.detect_center_y_min <= (top + bottom) / (2 * height) <= settings.detect_center_y_max
        )

    def embed(self, face: Any, image: np.ndarray, tta: bool | None = None) -> np.ndarray:
        """Normalize a face embedding, optionally averaging original and flipped views."""
        self._validate_image(image)
        use_tta = self.settings.use_tta if tta is None else tta
        embeddings = [self._normalized_embedding(face)]
        if use_tta:
            flipped = cv2.flip(image, 1)
            flipped_faces = self.analyze(flipped)
            if not flipped_faces:
                raise ValueError("No face detected in the flipped frame.")
            embeddings.append(self._normalized_embedding(flipped_faces[0]))
        combined = np.mean(embeddings, axis=0)
        norm = float(np.linalg.norm(combined))
        if norm == 0:
            raise ValueError("The face embedding was empty.")
        return (combined / norm).astype("float32")

    def detect_face(self, image: np.ndarray) -> bool:
        """Return true only when exactly one centered face passes detector geometry."""
        return self.detect_face_details(image)[0]

    def detect_face_details(
        self, image: np.ndarray
    ) -> tuple[bool, list[tuple[np.ndarray, np.ndarray | None]]]:
        """Return the existing readiness result and raw boxes/eye landmarks in one pass."""
        self._validate_image(image)
        detector = getattr(self.analysis, "det_model", None)
        if detector is None:
            if self._detection_analysis is None:
                self._detection_analysis = _create_face_analysis(
                    self.settings,
                    self._analysis_factory,
                    allowed_modules=["detection"],
                )
            detector = getattr(self._detection_analysis, "det_model", None)
        if detector is None:
            raise RuntimeError("InsightFace detection model is unavailable.")
        height, width = image.shape[:2]
        boxes, landmarks = detector.detect(image, max_num=0, metric="default")
        if boxes is None:
            return False, []
        boxes = np.asarray(boxes)
        landmark_array = None if landmarks is None else np.asarray(landmarks)
        detected = [
            (
                np.asarray(box[:4], dtype=np.float32),
                landmark_array[index] if landmark_array is not None and index < len(landmark_array) else None,
            )
            for index, box in enumerate(boxes)
        ]
        if len(boxes) != 1:
            return False, detected
        left, top, right, bottom, score = boxes[0]
        settings = self.settings
        face_width = max(0.0, right - left)
        face_height = max(0.0, bottom - top)
        found = bool(
            float(score) >= settings.detect_min_score
            and face_width >= width * settings.detect_min_width_ratio
            and face_height >= height * settings.detect_min_height_ratio
            and face_width * face_height / (width * height) >= settings.detect_min_area_ratio
            and settings.detect_center_x_min <= (left + right) / (2 * width) <= settings.detect_center_x_max
            and settings.detect_center_y_min <= (top + bottom) / (2 * height) <= settings.detect_center_y_max
        )
        return found, detected

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("A valid OpenCV image is required.")

    @staticmethod
    def _face_area(bbox: Any) -> float:
        left, top, right, bottom = bbox
        return max(0.0, right - left) * max(0.0, bottom - top)

    @staticmethod
    def _normalized_embedding(face: Any) -> np.ndarray:
        embedding = np.asarray(face.embedding, dtype="float32").reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm == 0:
            raise ValueError("The face embedding was empty.")
        return embedding / norm


def decode_image(data_url: str, settings: Settings = DEFAULT_SETTINGS) -> np.ndarray:
    """Decode data-URL/raw base64 into BGR, enforce pixel limits, and downscale."""
    if not isinstance(data_url, str):
        raise ValueError("Image must be a base64 string.")
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("The captured image is not valid base64 data.") from error
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The captured image could not be decoded.")
    height, width = image.shape[:2]
    if height * width > settings.max_image_pixels:
        raise ValueError("The captured image exceeds the pixel limit.")
    longest_side = max(height, width)
    if longest_side > settings.max_image_side:
        scale = settings.max_image_side / longest_side
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return image