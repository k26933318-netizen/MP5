from fastapi import HTTPException
import cv2
import numpy as np

from app.config import settings


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def decode_image(image_bytes: bytes, content_type: str) -> np.ndarray:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail={
            "code": "UNSUPPORTED_IMAGE_TYPE",
            "message": "Upload a JPEG, PNG, or WebP image.",
        })
    if not image_bytes or len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail={
            "code": "IMAGE_TOO_LARGE",
            "message": "The image is empty or exceeds the upload size limit.",
        })

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise HTTPException(status_code=400, detail={
            "code": "INVALID_IMAGE",
            "message": "The uploaded file is not a valid image.",
        })
    height, width = image.shape[:2]
    if min(width, height) < settings.min_image_size:
        raise HTTPException(status_code=422, detail={
            "code": "LOW_IMAGE_QUALITY",
            "message": "Please use a higher-resolution image.",
        })
    return image


def validate_face_quality(image: np.ndarray, bbox: np.ndarray, landmarks: np.ndarray) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
    height, width = image.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    face_width, face_height = x2 - x1, y2 - y1
    if min(face_width, face_height) < settings.min_face_size:
        _low_quality("Move closer so your face is clearly visible.")
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        _low_quality("Face landmarks could not be detected reliably.")
    if landmarks[0, 0] >= landmarks[1, 0] or landmarks[3, 1] <= landmarks[0, 1]:
        _low_quality("Face position is unsuitable. Look toward the camera.")

    face = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    if blur_score < 25 or brightness < 35 or brightness > 220:
        _low_quality("Please capture a sharper image in even lighting.")


def _low_quality(message: str) -> None:
    raise HTTPException(status_code=422, detail={
        "code": "LOW_IMAGE_QUALITY",
        "message": message,
    })
