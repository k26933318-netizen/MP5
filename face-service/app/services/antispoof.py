import threading

import cv2
import numpy as np

from app.config import settings


class AntiSpoofDetector:
    def __init__(self):
        self._sessions = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._sessions is not None:
            return
        with self._lock:
            if self._sessions is not None:
                return
            paths = (settings.antispoof_model_a_path, settings.antispoof_model_b_path)
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise RuntimeError(f"Required anti-spoof model is missing: {missing[0]}")
            import onnxruntime

            self._sessions = tuple(
                onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                for path in paths
            )

    @staticmethod
    def _input(image: np.ndarray, bbox: np.ndarray, scale: float) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in bbox[:4])
        side = min(max(round(max(x2 - x1, y2 - y1) * scale), 1), height, width)
        center_x = float(np.clip((x1 + x2) / 2, side / 2, width - side / 2))
        center_y = float(np.clip((y1 + y2) / 2, side / 2, height - side / 2))
        left, top = int(round(center_x - side / 2)), int(round(center_y - side / 2))
        crop = image[top:top + side, left:left + side]
        if crop.size == 0:
            raise RuntimeError("Anti-spoof face crop is empty.")
        resized = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
        return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]

    def score(self, image: np.ndarray, bbox: np.ndarray) -> float:
        self.load()
        scores = []
        for session, scale in zip(self._sessions, (2.7, 4.0)):
            tensor = self._input(image, bbox, scale)
            input_name = session.get_inputs()[0].name
            logits = np.asarray(session.run(None, {input_name: tensor})[0], dtype=np.float32).reshape(-1)
            if logits.size <= 1 or not np.isfinite(logits).all():
                raise RuntimeError("Anti-spoof model returned an invalid score.")
            probabilities = np.exp(logits - np.max(logits))
            probabilities /= probabilities.sum()
            scores.append(float(probabilities[1]))
        return float(np.mean(scores))

    def verify(self, frames: list[tuple[np.ndarray, np.ndarray]]) -> tuple[bool, float]:
        if not frames:
            return False, 0.0
        scores = [self.score(image, bbox) for image, bbox in frames]
        mean_score = float(np.mean(scores))
        return (
            mean_score >= settings.antispoof_threshold and min(scores) >= settings.antispoof_min_frame,
            mean_score,
        )


anti_spoof = AntiSpoofDetector()
