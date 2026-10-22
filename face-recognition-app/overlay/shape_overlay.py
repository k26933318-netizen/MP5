"""Draw smoothed OpenCV shapes around detected faces."""

from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Callable, Sequence

import cv2
import numpy as np

from .config import (
    SHAPE_ALPHA,
    SHAPE_COLOR,
    SHAPE_FILLED,
    SHAPE_PADDING,
    SHAPE_THICKNESS,
    SHAPE_TYPE,
    SMOOTHING_FACTOR,
    TRACER_CONFIG,
    TRACK_TIMEOUT_SECONDS,
)

_logger = logging.getLogger(__name__)
_VALID_SHAPES = {"ellipse", "circle", "rectangle", "rounded_rectangle", "corner_brackets"}
_track_lock = threading.RLock()


@dataclass
class _Track:
    box: np.ndarray
    last_seen: float


_tracks: dict[str, dict[int, _Track]] = {}
_next_track_id = 0


def draw_face_shape(
    frame: np.ndarray,
    face_box: Sequence[float],
    landmarks: np.ndarray | Sequence[Sequence[float]] | None = None,
    shape: str = SHAPE_TYPE,
) -> np.ndarray:
    """Smooth and draw one configured shape around a face box in-place."""
    return draw_face_shapes(frame, [(face_box, landmarks)], shape=shape)


def draw_face_shapes(
    frame: np.ndarray,
    detections: Sequence[
        tuple[Sequence[float], np.ndarray | Sequence[Sequence[float]] | None]
    ],
    *,
    track_key: str = "default",
    shape: str = SHAPE_TYPE,
) -> np.ndarray:
    """Draw shapes for all faces, smoothing box changes independently per stream."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("A non-empty OpenCV frame is required.")
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("The overlay frame must have three BGR or four BGRA channels.")
    if shape not in _VALID_SHAPES:
        raise ValueError(f"Unsupported face shape: {shape}")
    if not 0.0 <= SMOOTHING_FACTOR <= 1.0:
        raise ValueError("SMOOTHING_FACTOR must be between 0 and 1.")
    if not 0.0 <= SHAPE_ALPHA <= 1.0:
        raise ValueError("SHAPE_ALPHA must be between 0 and 1.")

    normalized = [
        (np.asarray(box, dtype=np.float32).reshape(4), points)
        for box, points in detections
        if len(box) == 4
    ]
    now = time.monotonic()
    global _next_track_id
    with _track_lock:
        for stale_key, stale_tracks in list(_tracks.items()):
            if not stale_tracks or max(track.last_seen for track in stale_tracks.values()) < now - TRACK_TIMEOUT_SECONDS:
                _tracks.pop(stale_key, None)

        previous = _tracks.get(track_key, {})
        candidates = []
        for index, (box, _) in enumerate(normalized):
            for track_id, track in previous.items():
                distance = _center_distance(box, track.box)
                max_distance = max(24.0, _box_diagonal(box), _box_diagonal(track.box)) * 1.5
                if distance <= max_distance:
                    candidates.append((distance, index, track_id))
        candidates.sort()

        matches: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, index, track_id in candidates:
            if index not in matches and track_id not in used_tracks:
                matches[index] = track_id
                used_tracks.add(track_id)

        active: dict[int, _Track] = {}
        smoothed_detections = []
        for index, (box, points) in enumerate(normalized):
            track_id = matches.get(index)
            if track_id is None:
                _next_track_id += 1
                track_id = _next_track_id
                smoothed = box
            else:
                old_box = previous[track_id].box
                factor = TRACER_CONFIG["easing_speed"] if TRACER_CONFIG["style"] == "modern" else SMOOTHING_FACTOR
                smoothed = old_box * (1.0 - factor) + box * factor
            active[track_id] = _Track(smoothed.copy(), now)
            smoothed_detections.append((smoothed, points))
        _tracks[track_key] = active

    for box, points in smoothed_detections:
        _draw_box(frame, box, points, shape)
    return frame


def _draw_box(
    frame: np.ndarray,
    box: np.ndarray,
    landmarks: np.ndarray | Sequence[Sequence[float]] | None,
    shape: str,
) -> None:
    height, width = frame.shape[:2]
    left, top, right, bottom = (float(value) for value in box)
    face_width = max(1.0, right - left)
    face_height = max(1.0, bottom - top)
    left -= face_width * SHAPE_PADDING / 100.0
    right += face_width * SHAPE_PADDING / 100.0
    top -= face_height * SHAPE_PADDING / 100.0
    bottom += face_height * SHAPE_PADDING / 100.0
    left = min(max(left, 0.0), width - 1.0)
    right = min(max(right, 0.0), width - 1.0)
    top = min(max(top, 0.0), height - 1.0)
    bottom = min(max(bottom, 0.0), height - 1.0)
    if right <= left or bottom <= top:
        return

    center = (round((left + right) / 2.0), round((top + bottom) / 2.0))
    modern_tracer = TRACER_CONFIG["style"] == "modern"
    angle = 0.0 if modern_tracer else _eye_angle(landmarks)
    thickness = max(1, int(TRACER_CONFIG["thickness"] if modern_tracer else SHAPE_THICKNESS))
    color = _drawing_color(frame, 1.0)
    axes = (max(1, round((right - left) / 2.0)), max(1, round((bottom - top) / 2.0)))

    if shape == "ellipse":
        axes = _clamp_ellipse_axes(axes, center, angle, width, height)
        if SHAPE_FILLED:
            _fill_with_alpha(
                frame,
                lambda target, fill_color: _draw_ellipse(
                    target, center, axes, angle, fill_color, -1
                ),
            )
        _draw_ellipse(frame, center, axes, angle, color, thickness)
    elif shape == "circle":
        radius = max(
            1,
            min(
                max(axes),
                center[0],
                width - 1 - center[0],
                center[1],
                height - 1 - center[1],
            ),
        )
        if SHAPE_FILLED:
            _fill_with_alpha(
                frame,
                lambda target, fill_color: _draw_circle(
                    target, center, radius, fill_color, -1
                ),
            )
            _draw_circle(frame, center, radius, color, thickness)
        else:
            _draw_circle(frame, center, radius, color, thickness)
    elif shape == "rounded_rectangle":
        radius = max(2, min(axes) // 5)
        corners = _rounded_corners(left, top, right, bottom, radius, center, angle, width, height)
        polygon = np.rint(corners).astype(np.int32)
        if SHAPE_FILLED:
            _fill_polygon(frame, polygon)
        cv2.polylines(frame, [polygon], True, color, thickness, cv2.LINE_AA)
    elif shape == "rectangle":
        corners = np.asarray(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        )
        corners = _rotate_points(corners, center, angle)
        corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
        polygon = np.rint(corners).astype(np.int32)
        if SHAPE_FILLED:
            _fill_polygon(frame, polygon)
        cv2.polylines(frame, [polygon], True, color, thickness, cv2.LINE_AA)
    elif shape == "corner_brackets":
        corners = np.asarray(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        )
        corners = _rotate_points(corners, center, angle)
        corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
        bracket_length = max(
            3.0,
            min(right - left, bottom - top)
            * (TRACER_CONFIG["bracket_size"] if modern_tracer else 0.28),
        )
        for index, origin in enumerate(corners):
            horizontal = corners[(index + 1) % 4] - origin
            vertical = corners[(index - 1) % 4] - origin
            horizontal /= max(float(np.linalg.norm(horizontal)), 1.0)
            vertical /= max(float(np.linalg.norm(vertical)), 1.0)
            for direction in (horizontal, vertical):
                end = origin + direction * bracket_length
                start_point = tuple(np.rint(origin).astype(int))
                end_point = tuple(np.rint(end).astype(int))
                cv2.line(frame, start_point, end_point, color, thickness, cv2.LINE_AA)
                if modern_tracer:
                    cv2.circle(frame, end_point, max(1, thickness // 2), color, -1, cv2.LINE_AA)

    if modern_tracer and shape == "corner_brackets":
        return
    _draw_corner_ticks(frame, (round(left), round(top), round(right), round(bottom)), color, thickness)


def _draw_corner_ticks(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    color: tuple[int, ...],
    thickness: int,
) -> None:
    left, top, right, bottom = bounds
    length = max(4, min(right - left, bottom - top) // 7)
    ticks = (
        ((left, top), (left + length, top), (left, top + length)),
        ((right, top), (right - length, top), (right, top + length)),
        ((left, bottom), (left + length, bottom), (left, bottom - length)),
        ((right, bottom), (right - length, bottom), (right, bottom - length)),
    )
    for origin, horizontal, vertical in ticks:
        cv2.line(frame, origin, horizontal, color, thickness, cv2.LINE_AA)
        cv2.line(frame, origin, vertical, color, thickness, cv2.LINE_AA)


def _drawing_color(frame: np.ndarray, opacity: float) -> tuple[int, ...]:
    base_color = (
        tuple(
            int(TRACER_CONFIG["colors"]["hold"][index:index + 2], 16)
            for index in (5, 3, 1)
        )
        if TRACER_CONFIG["style"] == "modern"
        else SHAPE_COLOR
    )
    if frame.shape[2] == 4:
        return (*base_color, round(opacity * 255))
    return base_color


def _draw_ellipse(
    frame: np.ndarray,
    center: tuple[int, int],
    axes: tuple[int, int],
    angle: float,
    color: tuple[int, ...],
    thickness: int,
) -> None:
    cv2.ellipse(frame, center, axes, angle, 0, 360, color, thickness, cv2.LINE_AA)


def _draw_circle(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, ...],
    thickness: int,
) -> None:
    cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)


def _fill_polygon(frame: np.ndarray, polygon: np.ndarray) -> None:
    _fill_with_alpha(
        frame,
        lambda target, fill_color: cv2.fillPoly(
            target, [polygon], fill_color, lineType=cv2.LINE_AA
        ),
    )


def _fill_with_alpha(
    frame: np.ndarray,
    draw: Callable[[np.ndarray, tuple[int, ...]], None],
) -> None:
    if frame.shape[2] == 4:
        draw(frame, (*SHAPE_COLOR, round(SHAPE_ALPHA * 255)))
        return
    overlay = frame.copy()
    draw(overlay, SHAPE_COLOR)
    cv2.addWeighted(overlay, SHAPE_ALPHA, frame, 1.0 - SHAPE_ALPHA, 0.0, dst=frame)


def _rounded_corners(
    left: float,
    top: float,
    right: float,
    bottom: float,
    radius: int,
    center: tuple[int, int],
    angle: float,
    width: int,
    height: int,
) -> np.ndarray:
    radius = min(radius, max(1, int((right - left) / 2)), max(1, int((bottom - top) / 2)))
    arcs = (
        (left + radius, top + radius, 180, 270),
        (right - radius, top + radius, 270, 360),
        (right - radius, bottom - radius, 0, 90),
        (left + radius, bottom - radius, 90, 180),
    )
    points = []
    for center_x, center_y, start, end in arcs:
        for angle_degrees in np.linspace(start, end, 5):
            point_angle = math.radians(float(angle_degrees))
            points.append(
                [
                    center_x + radius * math.cos(point_angle),
                    center_y + radius * math.sin(point_angle),
                ]
            )
    corners = _rotate_points(np.asarray(points, dtype=np.float32), center, angle)
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
    return corners


def _clamp_ellipse_axes(
    axes: tuple[int, int],
    center: tuple[int, int],
    angle: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    extent_x = math.hypot(axes[0] * cosine, axes[1] * sine)
    extent_y = math.hypot(axes[0] * sine, axes[1] * cosine)
    scale = min(
        1.0,
        center[0] / max(extent_x, 1.0),
        (width - 1 - center[0]) / max(extent_x, 1.0),
        center[1] / max(extent_y, 1.0),
        (height - 1 - center[1]) / max(extent_y, 1.0),
    )
    return max(1, round(axes[0] * scale)), max(1, round(axes[1] * scale))


def _rotate_points(points: np.ndarray, center: tuple[int, int], angle: float) -> np.ndarray:
    if not angle:
        return points
    radians = math.radians(angle)
    rotation = np.asarray(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=np.float32,
    )
    origin = np.asarray(center, dtype=np.float32)
    return (points - origin) @ rotation.T + origin


def _eye_angle(landmarks: np.ndarray | Sequence[Sequence[float]] | None) -> float:
    if landmarks is None:
        return 0.0
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        _logger.debug("Skipping face shape rotation because eye landmarks are unavailable.")
        return 0.0
    delta = points[1, :2] - points[0, :2]
    return math.degrees(math.atan2(float(delta[1]), float(delta[0])))


def _center_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_center = (first[:2] + first[2:]) / 2.0
    second_center = (second[:2] + second[2:]) / 2.0
    return float(np.linalg.norm(first_center - second_center))


def _box_diagonal(box: np.ndarray) -> float:
    return math.hypot(max(0.0, float(box[2] - box[0])), max(0.0, float(box[3] - box[1])))
