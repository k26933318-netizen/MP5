import unittest
from unittest.mock import patch

try:
    import numpy as np
    from fastapi import HTTPException
    from app.services.detection_service import FaceRecognitionRuntime
except ImportError as error:
    raise unittest.SkipTest(f"Face-service dependencies are not installed: {error}")


class DetectorStub:
    def __init__(self, boxes, landmarks):
        self.boxes = boxes
        self.landmarks = landmarks

    def detect(self, _image):
        return self.boxes, self.landmarks


class RecognizerStub:
    def embed(self, _image, _landmarks):
        return np.ones(512, dtype=np.float32) / np.sqrt(512)


def runtime_for(boxes, landmarks):
    return FaceRecognitionRuntime(
        detector=DetectorStub(boxes, landmarks),
        recognizer=RecognizerStub(),
        model_version="test-model"
    )


class DetectionServiceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((320, 320, 3), dtype=np.uint8)
        self.box = np.array([[70, 50, 250, 270, 0.99]], dtype=np.float32)
        self.landmarks = np.array([[[120, 130], [200, 130], [160, 175], [130, 220], [190, 220]]], dtype=np.float32)

    def test_accepts_exactly_one_quality_checked_face(self):
        runtime = runtime_for(self.box, self.landmarks)
        with patch("app.services.detection_service.validate_face_quality") as quality_check:
            runtime.detect_faces(self.image)
        quality_check.assert_called_once()

    def test_rejects_no_face(self):
        runtime = runtime_for(np.empty((0, 5)), np.empty((0, 5, 2)))
        with self.assertRaises(HTTPException) as raised:
            runtime.detect_faces(self.image)
        self.assertEqual(raised.exception.detail["code"], "NO_FACE")

    def test_rejects_multiple_faces(self):
        runtime = runtime_for(np.repeat(self.box, 2, axis=0), np.repeat(self.landmarks, 2, axis=0))
        with self.assertRaises(HTTPException) as raised:
            runtime.detect_faces(self.image)
        self.assertEqual(raised.exception.detail["code"], "MULTIPLE_FACES")

    def test_rejects_small_face_as_low_quality(self):
        small_box = np.array([[10, 10, 40, 40, 0.99]], dtype=np.float32)
        runtime = runtime_for(small_box, self.landmarks)
        with self.assertRaises(HTTPException) as raised:
            runtime.detect_faces(self.image)
        self.assertEqual(raised.exception.detail["code"], "LOW_IMAGE_QUALITY")

    def test_rejects_face_with_poor_lighting(self):
        from app.utils.image_utils import validate_face_quality

        dark_image = np.zeros((320, 320, 3), dtype=np.uint8)
        with self.assertRaises(HTTPException) as raised:
            validate_face_quality(dark_image, self.box[0], self.landmarks[0])
        self.assertEqual(raised.exception.detail["code"], "LOW_IMAGE_QUALITY")

    def test_embedding_path_returns_512_values_after_alignment_step(self):
        runtime = runtime_for(self.box, self.landmarks)
        with patch("app.services.detection_service.validate_face_quality"):
            embedding = runtime.embedding(self.image)
        self.assertEqual(len(embedding), 512)


if __name__ == "__main__":
    unittest.main()
