import numpy as np


class InsightFaceDetector:
    def __init__(self, model):
        if model is None or not callable(getattr(model, "detect", None)):
            raise RuntimeError("InsightFace detection model is unavailable.")
        self.model = model

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes, landmarks = self.model.detect(image, max_num=0, metric="default")
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 5)), np.empty((0, 5, 2))
        return np.asarray(boxes), np.asarray(landmarks)
