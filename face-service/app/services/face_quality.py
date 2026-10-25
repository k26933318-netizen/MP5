import json
import logging
import math

import cv2
import numpy as np
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger("face-recognition")


class BackgroundEstimationError(RuntimeError):
    pass


def _reject(code: str, message: str, state: str = "red") -> None:
    raise HTTPException(status_code=422, detail={
        "code": code,
        "message": message,
        "state": state,
    })


def _yaw_proxy(landmarks: np.ndarray) -> float:
    left_eye, right_eye, nose = np.asarray(landmarks, dtype=np.float32)[:3]
    axis = right_eye - left_eye
    distance = float(np.linalg.norm(axis))
    if distance <= 1e-6:
        return float("inf")
    return float(np.dot(nose - ((left_eye + right_eye) / 2), axis / distance) / distance)


def _log_diagnostics(event: dict, level: int = logging.INFO) -> None:
    if settings.face_diagnostics_enabled:
        logger.log(level, "face_validation %s", json.dumps(event, sort_keys=True))


def _estimated_background_mask(
    image: np.ndarray,
    bbox: np.ndarray,
    landmarks: np.ndarray | None = None,
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise BackgroundEstimationError("Expected a three-channel uint8 BGR frame.")
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        raise BackgroundEstimationError("Detector returned an invalid face bounding box.")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise BackgroundEstimationError("Face bounding box is outside the decoded frame.")

    scale = min(1.0, 192.0 / max(width, height))
    small_width = max(64, int(round(width * scale)))
    small_height = max(64, int(round(height * scale)))
    small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA) if scale < 1 else image
    scale_x = small_width / width
    scale_y = small_height / height
    sx1, sy1, sx2, sy2 = (int(round(value * factor)) for value, factor in (
        (x1, scale_x), (y1, scale_y), (x2, scale_x), (y2, scale_y)
    ))
    face_width = max(2, sx2 - sx1)
    face_height = max(2, sy2 - sy1)

    labels = np.full((small_height, small_width), cv2.GC_PR_BGD, dtype=np.uint8)
    border = max(2, int(round(min(small_height, small_width) * 0.02)))
    labels[:border, :] = cv2.GC_BGD
    labels[-border:, :] = cv2.GC_BGD
    labels[:, :border] = cv2.GC_BGD
    labels[:, -border:] = cv2.GC_BGD

    face_center = (int((sx1 + sx2) / 2), int(sy1 + face_height * 0.43))
    head_axes = (max(2, int(face_width * 0.78)), max(2, int(face_height * 0.88)))
    cv2.ellipse(labels, face_center, head_axes, 0, 0, 360, cv2.GC_PR_FGD, -1)
    sure_face_axes = (max(2, int(face_width * 0.38)), max(2, int(face_height * 0.42)))
    cv2.ellipse(labels, face_center, sure_face_axes, 0, 0, 360, cv2.GC_FGD, -1)

    shoulder_top = sy2 + int(face_height * 0.04)
    person_top = max(0, sy1 - int(face_height * 0.52))
    body_points = np.asarray([[
        (max(0, int(sx1 - face_width * 0.6)), person_top),
        (min(small_width - 1, int(sx2 + face_width * 0.6)), person_top),
        (min(small_width - 1, int(sx2 + face_width * 1.65)), shoulder_top),
        (min(small_width - 1, int(sx2 + face_width * 2.1)), small_height - border - 1),
        (max(0, int(sx1 - face_width * 2.1)), small_height - border - 1),
        (max(0, int(sx1 - face_width * 1.65)), shoulder_top),
    ]], dtype=np.int32)
    subject_mask = np.zeros((small_height, small_width), dtype=np.uint8)
    cv2.fillPoly(labels, body_points, cv2.GC_PR_FGD)
    cv2.fillPoly(subject_mask, body_points, 1)
    cv2.ellipse(labels, face_center, head_axes, 0, 0, 360, cv2.GC_PR_FGD, -1)
    cv2.ellipse(subject_mask, face_center, head_axes, 0, 0, 360, 1, -1)
    cv2.ellipse(labels, face_center, sure_face_axes, 0, 0, 360, cv2.GC_FGD, -1)
    background_anchors = np.zeros((small_height, small_width), dtype=bool)
    anchor_left = max(border, int(sx1 - face_width * 1.1))
    anchor_right = min(small_width - border, int(sx2 + face_width * 1.1))
    anchor_top = max(border, int(sy1 - face_height * 0.95))
    background_anchors[:anchor_top, :] = True
    background_anchors[:, :anchor_left] = True
    background_anchors[:, anchor_right:] = True
    labels[background_anchors & (subject_mask == 0)] = cv2.GC_BGD
    if landmarks is not None:
        points = np.asarray(landmarks, dtype=np.float32)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise BackgroundEstimationError("Detector returned invalid face landmarks.")
        radius = max(2, int(round(face_width * 0.07)))
        for point_x, point_y in points:
            center = (int(round(point_x * scale_x)), int(round(point_y * scale_y)))
            cv2.circle(labels, center, radius, cv2.GC_FGD, -1)
            cv2.circle(subject_mask, center, radius, 1, -1)
    labels[:border, :] = cv2.GC_BGD
    labels[-border:, :] = cv2.GC_BGD
    labels[:, :border] = cv2.GC_BGD
    labels[:, -border:] = cv2.GC_BGD

    foreground_model = np.zeros((1, 65), dtype=np.float64)
    background_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(small, labels, None, background_model, foreground_model, 1, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        raise BackgroundEstimationError("Foreground segmentation failed.") from exc

    background_small = (
        np.isin(labels, (cv2.GC_BGD, cv2.GC_PR_BGD)) & (subject_mask == 0)
    ).astype(np.uint8)
    return cv2.resize(background_small, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def background_metrics(image: np.ndarray, bbox: np.ndarray, landmarks: np.ndarray | None = None) -> dict:
    try:
        bbox_values = [float(value) for value in np.asarray(bbox).reshape(-1)[:4]]
    except (TypeError, ValueError) as exc:
        _log_diagnostics({
            "face_detected": True,
            "face_bbox": None,
            "valid_background_pixels": 0,
            "near_white_ratio": None,
            "configured_threshold": settings.background_min_white_fraction,
            "uniformity_stddev": None,
            "failure_condition": "background_estimation_error",
            "failure_reason": str(exc),
        }, logging.ERROR)
        raise BackgroundEstimationError("Detector returned an invalid face bounding box.") from exc
    try:
        background_mask = _estimated_background_mask(image, np.asarray(bbox), landmarks)
    except BackgroundEstimationError as exc:
        _log_diagnostics({
            "face_detected": True,
            "face_bbox": bbox_values,
            "valid_background_pixels": 0,
            "near_white_ratio": None,
            "configured_threshold": settings.background_min_white_fraction,
            "uniformity_stddev": None,
            "failure_condition": "background_estimation_error",
            "failure_reason": str(exc),
        }, logging.ERROR)
        raise
    except Exception as exc:
        _log_diagnostics({
            "face_detected": True,
            "face_bbox": bbox_values,
            "valid_background_pixels": 0,
            "near_white_ratio": None,
            "configured_threshold": settings.background_min_white_fraction,
            "uniformity_stddev": None,
            "failure_condition": "background_estimation_error",
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }, logging.ERROR)
        raise BackgroundEstimationError("Background estimation failed unexpectedly.") from exc

    pixels = image[background_mask]
    valid_pixels = int(pixels.shape[0])
    minimum_valid_pixels = max(256, int(image.shape[0] * image.shape[1] * 0.02))
    if valid_pixels < minimum_valid_pixels:
        _log_diagnostics({
            "face_detected": True,
            "face_bbox": bbox_values,
            "valid_background_pixels": valid_pixels,
            "near_white_ratio": None,
            "configured_threshold": settings.background_min_white_fraction,
            "uniformity_stddev": None,
            "failure_condition": "empty_background_sample",
        }, logging.ERROR)
        raise BackgroundEstimationError("Segmentation produced too few valid background pixels.")

    brightness = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
    channel_spread = pixels.max(axis=1).astype(np.int16) - pixels.min(axis=1).astype(np.int16)
    near_white = (brightness >= settings.background_min_brightness) & (channel_spread <= 60)
    white_fraction = float(np.count_nonzero(near_white) / valid_pixels)
    white_brightness = brightness[near_white].astype(np.float32)
    uniformity = float(np.std(white_brightness)) if white_brightness.size else None
    failed_conditions = []
    if white_fraction < settings.background_min_white_fraction:
        failed_conditions.append("near_white_fraction_below_threshold")
    if uniformity is None or uniformity > settings.background_max_stddev:
        failed_conditions.append("near_white_background_not_uniform")
    metrics = {
        "white_fraction": white_fraction,
        "valid_background_pixels": valid_pixels,
        "uniformity_stddev": uniformity,
        "failure_conditions": failed_conditions,
    }
    _log_diagnostics({
        "face_detected": True,
        "face_bbox": bbox_values,
        "valid_background_pixels": valid_pixels,
        "near_white_ratio": white_fraction,
        "configured_threshold": settings.background_min_white_fraction,
        "uniformity_stddev": uniformity,
        "failure_condition": failed_conditions or None,
    })
    return metrics


def assess_face(
    image: np.ndarray,
    bbox: np.ndarray,
    landmarks: np.ndarray,
    require_frontal: bool = True,
    require_background: bool = True,
    relaxed_position: bool = False,
) -> dict:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    face_width = x2 - x1
    face_height = y2 - y1
    if face_width <= 0 or face_height <= 0 or min(face_width, face_height) < settings.min_face_size:
        _reject("FACE_TOO_SMALL", "Move closer to the camera.")
    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)
    width_fraction = face_width / width
    min_width_fraction = settings.face_min_width_fraction * (0.70 if relaxed_position else 1.0)
    max_width_fraction = min(1.0, settings.face_max_width_fraction * (1.15 if relaxed_position else 1.0))
    center_x_tolerance = min(0.5, settings.face_center_x_tolerance * (1.4 if relaxed_position else 1.0))
    center_y_tolerance = min(0.5, settings.face_center_y_tolerance * (1.4 if relaxed_position else 1.0))
    if width_fraction < min_width_fraction:
        _reject("FACE_TOO_SMALL", "Move closer to the camera.", "yellow")
    if width_fraction > max_width_fraction:
        _reject("FACE_TOO_LARGE", "Move farther from the camera.", "yellow")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        _reject("FACE_CROPPED", "Keep your complete face inside the frame.")
    if abs(center_x - 0.5) > center_x_tolerance:
        direction = "right" if center_x < 0.5 else "left"
        _reject("FACE_OFF_CENTER", f"Move slightly {direction} to center your face.", "yellow")
    if abs(center_y - 0.46) > center_y_tolerance:
        direction = "down" if center_y < 0.46 else "up"
        _reject("FACE_OFF_CENTER", f"Move slightly {direction} to center your face.", "yellow")
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        _reject("LANDMARKS_UNAVAILABLE", "Look straight at the camera.")

    left_eye, right_eye = landmarks[:2]
    nose = landmarks[2]
    mouth_midpoint = (landmarks[3] + landmarks[4]) / 2
    eye_midpoint = (left_eye + right_eye) / 2
    roll = abs(math.degrees(math.atan2(
        float(right_eye[1] - left_eye[1]),
        float(right_eye[0] - left_eye[0]),
    )))
    yaw = _yaw_proxy(landmarks)
    eye_to_mouth = float(mouth_midpoint[1] - eye_midpoint[1])
    pitch = float((nose[1] - eye_midpoint[1]) / eye_to_mouth) if eye_to_mouth > 1e-6 else float("inf")
    max_roll = min(45, settings.face_max_roll_degrees + (6 if relaxed_position else 0))
    max_yaw = min(1, settings.face_max_yaw_proxy * (1.4 if relaxed_position else 1.0))
    max_pitch_deviation = min(0.5, settings.face_max_pitch_deviation * (1.35 if relaxed_position else 1.0))
    if (roll > max_roll or
            (require_frontal and abs(yaw) > max_yaw) or
            abs(pitch - 0.5) > max_pitch_deviation):
        _reject("FACE_POSE", "Keep your face straight and look toward the camera.", "yellow")

    face = image[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    if blur < 25:
        _reject("BLURRY", "Hold still so the image is sharp.")
    if brightness < 35 or brightness > 220:
        _reject("BAD_LIGHTING", "Improve the lighting on your face.")

    background = None
    if require_background:
        try:
            background = background_metrics(image, bbox, landmarks)
        except BackgroundEstimationError:
            _reject(
                "BACKGROUND_ESTIMATION_FAILED",
                "Unable to verify your face. Please retry.",
            )
        if background["failure_conditions"]:
            _reject(
                "IMAGE_CONDITIONS_UNSUITABLE",
                "Use a plain white, evenly lit background and keep only one face in view.",
            )
    return {
        "center_x": center_x,
        "center_y": center_y,
        "width_fraction": width_fraction,
        "yaw": yaw,
        "roll": roll,
        "pitch": pitch,
        "blur": blur,
        "brightness": brightness,
        "background": background,
    }
