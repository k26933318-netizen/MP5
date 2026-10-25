from pathlib import Path

import cv2
import numpy as np


class RetinaFaceDetector:
    """CPU RetinaFace MobileNet-0.25 inference with 5-point landmarks."""

    _steps = (8, 16, 32)
    _min_sizes = ((10, 20), (32, 64), (128, 256))

    def __init__(self, model_path: Path, score_threshold: float = 0.7):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise RuntimeError(f"RetinaFace ONNX model is missing: {self.model_path}")
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.height = int(shape[2]) if isinstance(shape[2], int) else 640
        self.width = int(shape[3]) if isinstance(shape[3], int) else 640
        self.score_threshold = score_threshold
        self.priors = self._make_priors()

    def _make_priors(self) -> np.ndarray:
        priors = []
        for step, sizes in zip(self._steps, self._min_sizes):
            feature_height = int(np.ceil(self.height / step))
            feature_width = int(np.ceil(self.width / step))
            for row in range(feature_height):
                for column in range(feature_width):
                    for size in sizes:
                        priors.append([
                            (column + 0.5) * step / self.width,
                            (row + 0.5) * step / self.height,
                            size / self.width,
                            size / self.height,
                        ])
        return np.asarray(priors, dtype=np.float32)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image_height, image_width = image.shape[:2]
        resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32)
        tensor -= np.asarray([104.0, 117.0, 123.0], dtype=np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        outputs = self.session.run(None, {self.input.name: tensor})
        if len(outputs) < 3:
            raise RuntimeError("RetinaFace model must return boxes, confidence, and landmarks.")
        locations, confidences, landmark_offsets = [np.asarray(value).reshape(-1, value.shape[-1]) for value in outputs[:3]]
        count = min(len(self.priors), len(locations), len(confidences), len(landmark_offsets))
        if count == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        priors = self.priors[:count]
        boxes = np.empty((count, 4), dtype=np.float32)
        centers = priors[:, :2] + locations[:count, :2] * 0.1 * priors[:, 2:]
        sizes = priors[:, 2:] * np.exp(locations[:count, 2:] * 0.2)
        boxes[:, :2] = centers - sizes / 2
        boxes[:, 2:] = centers + sizes / 2
        boxes *= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)

        points = landmark_offsets[:count].reshape(-1, 5, 2)
        points = (priors[:, None, :2] + points * 0.1 * priors[:, None, 2:])
        points *= np.asarray([image_width, image_height], dtype=np.float32)

        scores = confidences[:count, 1]
        selected = np.flatnonzero(scores >= self.score_threshold)
        if not len(selected):
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)
        order = selected[np.argsort(scores[selected])[::-1][:5000]]
        nms_indices = cv2.dnn.NMSBoxes(
            [
                [float(boxes[index, 0]), float(boxes[index, 1]),
                 float(boxes[index, 2] - boxes[index, 0]), float(boxes[index, 3] - boxes[index, 1])]
                for index in order
            ],
            [float(scores[index]) for index in order],
            self.score_threshold,
            0.4,
            top_k=750,
        )
        kept = order[np.asarray(nms_indices).reshape(-1)] if len(nms_indices) else np.empty(0, dtype=int)
        detected = np.column_stack((boxes[kept], scores[kept])).astype(np.float32)
        return detected, points[kept].astype(np.float32)
