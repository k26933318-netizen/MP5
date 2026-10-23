

import logging
import os

import cv2
import numpy as np

FACE_MODEL = os.getenv("FACE_MODEL", "buffalo_l")
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "640"))
# Re-tune this threshold because blur is now measured on a resized face crop.
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "100"))
FACE_MIN_SIZE = float(os.getenv("FACE_MIN_SIZE", "100"))
FACE_MIN_DETECTION_SCORE = float(os.getenv("FACE_MIN_DETECTION_SCORE", "0.7"))
DETECT_MIN_SCORE = float(os.getenv("DETECT_MIN_SCORE", "0.5"))
DETECT_MIN_WIDTH_RATIO = float(os.getenv("DETECT_MIN_WIDTH_RATIO", "0.12"))
DETECT_MIN_HEIGHT_RATIO = float(os.getenv("DETECT_MIN_HEIGHT_RATIO", "0.12"))
DETECT_MIN_AREA_RATIO = float(os.getenv("DETECT_MIN_AREA_RATIO", "0.03"))
DETECT_CENTER_X_MIN = float(os.getenv("DETECT_CENTER_X_MIN", "0.2"))
DETECT_CENTER_X_MAX = float(os.getenv("DETECT_CENTER_X_MAX", "0.8"))
DETECT_CENTER_Y_MIN = float(os.getenv("DETECT_CENTER_Y_MIN", "0.15"))
DETECT_CENTER_Y_MAX = float(os.getenv("DETECT_CENTER_Y_MAX", "0.85"))
LIVENESS_LANDMARK_MOVEMENT = float(os.getenv("LIVENESS_LANDMARK_MOVEMENT", "2.0"))
LIVENESS_PIXEL_DIFFERENCE = float(os.getenv("LIVENESS_PIXEL_DIFFERENCE", "2.0"))

_logger = logging.getLogger(__name__)
_face_analysis = None
_detection_face_analysis = None


def _runtime_options():
    import onnxruntime

    available = onnxruntime.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in available
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    return providers, 0 if use_cuda else -1


def _create_face_analysis(allowed_modules):
    from insightface.app import FaceAnalysis

    providers, ctx_id = _runtime_options()
    analysis = FaceAnalysis(
        name=FACE_MODEL,
        allowed_modules=allowed_modules,
        providers=providers,
    )
    analysis.prepare(ctx_id=ctx_id, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
    return analysis


def _get_face_analysis():
    global _face_analysis
    if _face_analysis is None:
        _face_analysis = _create_face_analysis(
            ["detection", "landmark_2d_106", "recognition"]
        )
    return _face_analysis


def _get_detection_model():
    global _detection_face_analysis
    detector = getattr(_get_face_analysis(), "det_model", None)
    if detector is None:
        if _detection_face_analysis is None:
            _detection_face_analysis = _create_face_analysis(["detection"])
        detector = getattr(_detection_face_analysis, "det_model", None)
    if detector is None:
        raise RuntimeError("InsightFace detection model is unavailable.")
    return detector


def warm_up():
    """Load InsightFace and initialize its inference path before serving requests."""
    _get_face_analysis().get(np.zeros((FACE_DET_SIZE, FACE_DET_SIZE, 3), dtype=np.uint8))


def get_face(image):
    """Return the largest detected face in an OpenCV image."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("A valid OpenCV image is required.")

    faces = _get_face_analysis().get(image)
    if not faces:
        raise ValueError("No face detected. Please center your face and try again.")
    if len(faces) > 1:
        _logger.warning("Multiple faces detected; using the largest face.")

    return max(faces, key=lambda item: face_area(item.bbox))


def get_embedding(image):
    """Return a normalized 512-dim ArcFace embedding for the largest face."""
    return embedding_from_face(get_face(image))


def embedding_from_face(face):
    """Return a normalized embedding already computed for a detected face."""
    embedding = np.asarray(face.embedding, dtype="float32")
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("The face embedding was empty.")
    return embedding / norm


def is_good_quality(image, face):
    """Return whether an image is sharp enough and contains a confident face."""
    if image is None or face is None:
        return False

    left, top, right, bottom = face.bbox
    face_width = max(0.0, right - left)
    face_height = max(0.0, bottom - top)
    if face_width < FACE_MIN_SIZE or face_height < FACE_MIN_SIZE:
        return False
    if float(getattr(face, "det_score", 0.0)) <= FACE_MIN_DETECTION_SCORE:
        return False

    return _face_blur_variance(image, face) >= BLUR_THRESHOLD


def quality_error_message(image, face):
    """Return a user-facing reason when an image fails the quality gate."""
    if face is None:
        return "No face detected. Please center your face and try again."
    left, top, right, bottom = face.bbox
    if right - left < FACE_MIN_SIZE or bottom - top < FACE_MIN_SIZE:
        return "Face too small - move closer to the camera."
    if float(getattr(face, "det_score", 0.0)) <= FACE_MIN_DETECTION_SCORE:
        return "Face detection confidence is too low - face the camera directly."
    if _face_blur_variance(image, face) < BLUR_THRESHOLD:
        return "Image too blurry - hold still."
    return ""


def _face_blur_variance(image, face):
    height, width = image.shape[:2]
    left, top, right, bottom = face.bbox
    x1 = max(0, min(width, int(np.floor(left))))
    y1 = max(0, min(height, int(np.floor(top))))
    x2 = max(0, min(width, int(np.ceil(right))))
    y2 = max(0, min(height, int(np.ceil(bottom))))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 160))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def has_liveness_movement(first_image, second_image, first_face=None, second_face=None):
    """Detect a small change between two frames, preferring face landmarks."""
    # TODO: add a passive anti-spoofing model (for example MiniFASNet) or a blink/head-turn challenge.
    if first_face is None:
        first_face = get_face(first_image)
    if second_face is None:
        second_face = get_face(second_image)
    first_landmarks = getattr(first_face, "landmark_2d_106", None)
    second_landmarks = getattr(second_face, "landmark_2d_106", None)
    if first_landmarks is not None and second_landmarks is not None:
        first_points = np.asarray(first_landmarks, dtype="float32")
        second_points = np.asarray(second_landmarks, dtype="float32")
        if first_points.shape == second_points.shape:
            return float(np.mean(np.linalg.norm(first_points - second_points, axis=1))) >= LIVENESS_LANDMARK_MOVEMENT

    first_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.resize(first_gray, (160, 120))
    second_gray = cv2.resize(second_gray, (160, 120))
    difference = cv2.absdiff(first_gray, second_gray)
    return float(np.mean(difference)) >= LIVENESS_PIXEL_DIFFERENCE


def detect_face(image):
    """Check for one confident, reasonably sized, roughly centered face."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return False

    height, width = image.shape[:2]
    boxes, _ = _get_detection_model().detect(image, max_num=0, metric="default")
    if boxes is None or len(boxes) != 1:
        return False

    left, top, right, bottom, score = boxes[0]
    if float(score) < DETECT_MIN_SCORE:
        return False

    face_width = max(0.0, right - left)
    face_height = max(0.0, bottom - top)
    face_area_ratio = (face_width * face_height) / (width * height)
    center_x = ((left + right) / 2) / width
    center_y = ((top + bottom) / 2) / height
    return bool(
        face_width >= width * DETECT_MIN_WIDTH_RATIO
        and face_height >= height * DETECT_MIN_HEIGHT_RATIO
        and face_area_ratio >= DETECT_MIN_AREA_RATIO
        and DETECT_CENTER_X_MIN <= center_x <= DETECT_CENTER_X_MAX
        and DETECT_CENTER_Y_MIN <= center_y <= DETECT_CENTER_Y_MAX
    )


def face_area(bbox):
    left, top, right, bottom = bbox
    return max(0, right - left) * max(0, bottom - top)