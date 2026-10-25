from typing import Any

import numpy as np
from insightface.utils import face_align


class ArcFaceRecognizer:
    def __init__(self, model: Any):
        self.model = model
        if self.model is None or "arcface" not in type(self.model).__name__.lower():
            raise RuntimeError("InsightFace ArcFace recognition model is unavailable.")
        output_shape = self.model.session.get_outputs()[0].shape
        if not output_shape or output_shape[-1] != 512:
            raise RuntimeError("Configured ArcFace model must produce 512-dimensional embeddings")
        self.input_size = self.model.input_size

    def embed(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        size = self.input_size[0]
        aligned = face_align.norm_crop(image, landmark=landmarks, image_size=size)
        features = np.asarray(self.model.get_feat(aligned), dtype=np.float32).reshape(-1)
        if features.shape != (512,) or not np.isfinite(features).all():
            raise RuntimeError("ArcFace returned an invalid embedding")
        norm = float(np.linalg.norm(features))
        if norm == 0.0:
            raise RuntimeError("ArcFace returned an empty embedding")
        return features / norm
