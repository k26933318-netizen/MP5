import unittest

try:
    import numpy as np
    from fastapi import HTTPException
    from app.services.face_quality import assess_face
except ImportError as error:
    raise unittest.SkipTest(f"Face-service dependencies are not installed: {error}")


class BlurGateTests(unittest.TestCase):
    def test_enrollment_position_relaxation_does_not_allow_blurred_faces(self):
        image = np.full((480, 640, 3), 120, dtype=np.uint8)
        bbox = np.asarray([220, 100, 420, 350], dtype=np.float32)
        landmarks = np.asarray([[280, 190], [360, 190], [320, 230], [290, 275], [350, 275]])

        with self.assertRaises(HTTPException) as raised:
            assess_face(image, bbox, landmarks, require_background=False, relaxed_position=True)

        self.assertEqual(raised.exception.detail["code"], "BLURRY")
        self.assertIn("hold still", raised.exception.detail["message"].lower())


if __name__ == "__main__":
    unittest.main()
