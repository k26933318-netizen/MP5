import unittest
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

try:
    import numpy as np
    import cv2
    from fastapi import HTTPException
    from app.config import load_settings, settings as app_settings
    from app import main as face_main
    from app.services.detection_service import FaceRecognitionRuntime
    from app.services.face_quality import (
        BackgroundEstimationError,
        _estimated_background_mask,
        assess_face,
        background_metrics,
    )
    from app.utils.image_utils import decode_image
except ImportError as error:
    raise unittest.SkipTest(f"Face-service dependencies are not installed: {error}")


class DetectorStub:
    def detect(self, image):
        index = int(image[0, 0, 0])
        yaw = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.0, 7: 0.0, 8: 0.0}.get(index, 0.0)
        box = np.asarray([[220, 100, 420, 350, 0.99]], dtype=np.float32)
        landmarks = np.asarray([[
            [280, 190], [360, 190], [320 + yaw * 80, 230], [290, 275], [350, 275]
        ]], dtype=np.float32)
        return box, landmarks


class RecognizerStub:
    def embed(self, _image, _landmarks):
        return np.asarray([1.0] + [0.0] * 511, dtype=np.float32)


def metrics(image, _box, _landmarks, require_frontal=True, require_background=True, relaxed_position=False):
    index = int(image[0, 0, 0])
    yaw = {3: 0.15, 4: 0.2, 5: 0.25}.get(index, 0.0)
    return {
        "center_x": 0.5,
        "center_y": 0.46,
        "width_fraction": 0.31,
        "yaw": yaw,
        "roll": 0.0,
        "blur": 100.0,
        "brightness": 120.0,
        "background": {"white_fraction": 1.0, "brightness": 245.0, "stddev": 1.0},
    }


def runtime():
    return FaceRecognitionRuntime(DetectorStub(), RecognizerStub(), "test-model")


def frame(index):
    image = np.full((480, 640, 3), 245, dtype=np.uint8)
    image[0, 0, 0] = index
    return image


def webcam_frame(seed=0, background=(245, 245, 245)):
    rng = np.random.default_rng(seed)
    image = np.full((480, 640, 3), background, dtype=np.uint8)
    cv2.fillPoly(image, [np.asarray([[230, 300], [410, 300], [500, 479], [140, 479]], np.int32)], (60, 80, 100))
    face = image[100:350, 220:420]
    face[:] = np.clip(145 + rng.normal(0, 18, face.shape), 0, 255).astype(np.uint8)
    cv2.ellipse(image, (320, 145), (95, 55), 0, 0, 360, (30, 30, 30), -1)
    image[0, 0, 0] = 0
    return image


def run_immediate(awaitable):
    """Run an async endpoint whose test doubles complete without yielding."""
    try:
        awaitable.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("The mocked endpoint unexpectedly required an event loop.")


class FaceWorkflowTests(unittest.TestCase):
    def test_white_background_is_measured_outside_estimated_person_region(self):
        image = np.full((480, 640, 3), 245, dtype=np.uint8)
        bbox = np.asarray([220, 100, 420, 350], dtype=np.float32)
        white = background_metrics(image, bbox)
        self.assertGreaterEqual(white["white_fraction"], 0.60)

        image[:] = (80, 100, 120)
        non_white = background_metrics(image, bbox)
        self.assertLess(non_white["white_fraction"], 0.60)

    def test_assessment_accepts_a_background_with_sixty_percent_near_white_pixels(self):
        image = np.full((480, 640, 3), 245, dtype=np.uint8)
        image[:, :40] = 100
        rng = np.random.default_rng(11)
        image[100:350, 220:420] = rng.integers(70, 180, size=(250, 200, 3), dtype=np.uint8)
        bbox = np.asarray([220, 100, 420, 350], dtype=np.float32)
        landmarks = np.asarray([[280, 190], [360, 190], [320, 230], [290, 275], [350, 275]])
        region = background_metrics(image, bbox)
        self.assertGreaterEqual(region["white_fraction"], 0.60)
        self.assertLessEqual(region["white_fraction"], 0.75)
        result = assess_face(image, bbox, landmarks)
        self.assertGreaterEqual(result["background"]["white_fraction"], 0.60)

    def test_background_ratio_accepts_configured_50_60_and_75_percent_boundaries(self):
        total = 2000
        mask = np.ones((40, 50), dtype=bool)
        for target in (0.50, 0.60, 0.75):
            with self.subTest(target=target):
                image = np.full((40, 50, 3), 120, dtype=np.uint8)
                white_pixels = int(total * target)
                image.reshape(-1, 3)[:white_pixels] = 245
                with patch("app.services.face_quality._estimated_background_mask", return_value=mask), \
                        patch("app.services.face_quality.settings", replace(
                            app_settings, background_min_white_fraction=target
                        )):
                    result = background_metrics(image, np.asarray([10, 10, 30, 30]))
                self.assertEqual(result["valid_background_pixels"], total)
                self.assertAlmostEqual(result["white_fraction"], target)
                self.assertEqual(result["failure_conditions"], [])

    def test_background_ratio_below_sixty_percent_is_rejected(self):
        image = np.full((40, 50, 3), 120, dtype=np.uint8)
        image.reshape(-1, 3)[:1000] = 245
        with patch("app.services.face_quality._estimated_background_mask", return_value=np.ones((40, 50), bool)):
            result = background_metrics(image, np.asarray([10, 10, 30, 30]))
        self.assertEqual(result["white_fraction"], 0.5)
        self.assertIn("near_white_fraction_below_threshold", result["failure_conditions"])

    def test_background_threshold_is_loaded_from_environment(self):
        with patch.dict("os.environ", {"FACE_BACKGROUND_MIN_WHITE_FRACTION": "0.75"}):
            self.assertEqual(load_settings().background_min_white_fraction, 0.75)

    def test_shadowed_and_slightly_gray_wall_pixels_are_tolerated_at_sixty_percent(self):
        image = np.full((40, 50, 3), 180, dtype=np.uint8)
        image.reshape(-1, 3)[:1200] = 215
        with patch("app.services.face_quality._estimated_background_mask", return_value=np.ones((40, 50), bool)):
            result = background_metrics(image, np.asarray([10, 10, 30, 30]))
        self.assertAlmostEqual(result["white_fraction"], 0.60)
        self.assertEqual(result["uniformity_stddev"], 0.0)
        self.assertEqual(result["failure_conditions"], [])

    def test_nonuniform_near_white_background_is_rejected(self):
        image = np.full((40, 50, 3), 120, dtype=np.uint8)
        image.reshape(-1, 3)[:1200] = 190
        image.reshape(-1, 3)[:600] = 255
        with patch("app.services.face_quality._estimated_background_mask", return_value=np.ones((40, 50), bool)):
            result = background_metrics(image, np.asarray([10, 10, 30, 30]))
        self.assertIn("near_white_background_not_uniform", result["failure_conditions"])

    def test_empty_background_sample_is_an_internal_estimation_error(self):
        image = np.full((40, 50, 3), 245, dtype=np.uint8)
        with patch("app.services.face_quality._estimated_background_mask", return_value=np.zeros((40, 50), bool)):
            with self.assertRaises(BackgroundEstimationError):
                background_metrics(image, np.asarray([10, 10, 30, 30]))

    def test_segmenter_excludes_face_hair_and_clothing_from_background(self):
        image = np.full((480, 640, 3), 245, dtype=np.uint8)
        cv2.ellipse(image, (320, 180), (75, 105), 0, 0, 360, (80, 120, 175), -1)
        cv2.ellipse(image, (320, 105), (76, 28), 0, 0, 360, (25, 25, 25), -1)
        cv2.fillPoly(image, [np.asarray([[260, 270], [380, 270], [490, 479], [150, 479]], np.int32)], (55, 75, 95))
        mask = _estimated_background_mask(image, np.asarray([240, 80, 400, 280]))
        self.assertFalse(mask[180, 320], "face center must be excluded")
        self.assertFalse(mask[105, 320], "hair region must be excluded")
        self.assertFalse(mask[400, 320], "clothing region must be excluded")
        self.assertTrue(mask[20, 10], "visible wall in the corner remains background")

    def test_assessment_rejects_a_face_outside_the_positioning_guide(self):
        image = np.full((480, 640, 3), 245, dtype=np.uint8)
        bbox = np.asarray([20, 100, 220, 350], dtype=np.float32)
        landmarks = np.asarray([[80, 190], [160, 190], [120, 230], [90, 275], [150, 275]])
        with self.assertRaises(HTTPException) as raised:
            assess_face(image, bbox, landmarks)
        self.assertEqual(raised.exception.detail["code"], "FACE_OFF_CENTER")

    def test_enrollment_assessment_does_not_require_a_white_background(self):
        rng = np.random.default_rng(7)
        image = np.full((480, 640, 3), 120, dtype=np.uint8)
        image[100:350, 220:420] = rng.integers(30, 220, size=(250, 200, 3), dtype=np.uint8)
        bbox = np.asarray([220, 100, 420, 350], dtype=np.float32)
        landmarks = np.asarray([[280, 190], [360, 190], [320, 230], [290, 275], [350, 275]])
        result = assess_face(image, bbox, landmarks, require_background=False, relaxed_position=True)
        self.assertIsNone(result["background"])

    def test_mostly_white_webcam_frame_with_centered_face_passes_background_check(self):
        image = webcam_frame()
        bbox = np.asarray([220, 100, 420, 350], dtype=np.float32)
        landmarks = np.asarray([[280, 190], [360, 190], [320, 230], [290, 275], [350, 275]])
        result = assess_face(image, bbox, landmarks)
        self.assertGreaterEqual(result["background"]["white_fraction"], 0.60)
        self.assertGreater(result["background"]["valid_background_pixels"], 0)

    def test_face_detection_failure_is_not_reported_as_a_background_failure(self):
        empty_detector = type("EmptyDetector", (), {
            "detect": lambda _self, _image: (np.empty((0, 5)), np.empty((0, 5, 2)))
        })()
        subject = FaceRecognitionRuntime(empty_detector, RecognizerStub(), "test-model")
        result = subject.assess_position(frame(0))
        self.assertFalse(result["ready"])
        self.assertIsNone(result["face"])
        self.assertIn("no face", result["message"].lower())
        self.assertNotIn("surroundings", result["message"].lower())

    def test_malformed_image_is_rejected_before_face_or_background_validation(self):
        with self.assertRaises(HTTPException) as raised:
            decode_image(b"not an image", "image/jpeg")
        self.assertEqual(raised.exception.detail["code"], "INVALID_IMAGE")

    def test_public_assessment_response_does_not_expose_background_diagnostics(self):
        class RejectingRuntime:
            def assess_position(self, _image, require_frontal=True, **_kwargs):
                raise HTTPException(status_code=422, detail={
                    "code": "IMAGE_CONDITIONS_UNSUITABLE",
                    "message": "Adjust your surroundings and try again.",
                    "state": "red",
                    "near_white_ratio": 0.12,
                })

        with patch("app.main.authorize"), \
                patch("app.main.runtime", RejectingRuntime()), \
                patch("app.main.read_upload", new=AsyncMock(return_value=b"test-image")), \
                patch("app.main.decode_image", return_value=frame(0)):
            response = run_immediate(face_main.assess_position(
                SimpleNamespace(content_type="image/jpeg"),
                "test-key",
                "position",
            ))
        body = json.loads(response.body)
        self.assertEqual(body["message"], "Adjust your surroundings and try again.")
        self.assertNotIn("code", body)
        self.assertNotIn("near_white_ratio", body)
        self.assertNotIn("configured_threshold", body)

    def test_position_assessment_returns_real_face_bounds_without_background_metrics(self):
        subject = runtime()
        with patch("app.services.detection_service.FaceRecognitionRuntime._validate_face"), \
                patch("app.services.detection_service.assess_face", side_effect=metrics):
            result = subject.assess_position(frame(0))
        self.assertEqual(result["face"]["bbox"], [220.0, 100.0, 420.0, 350.0])
        self.assertEqual(result["face"]["frameWidth"], 640)
        self.assertEqual(result["face"]["frameHeight"], 480)
        self.assertNotIn("metrics", result)

    def test_enrollment_requires_a_stable_sequence_and_returns_one_embedding(self):
        subject = runtime()
        frames = [frame(index) for index in range(5)]
        with patch("app.services.detection_service.FaceRecognitionRuntime._validate_face"), \
                patch("app.services.detection_service.assess_face", side_effect=metrics), \
                patch("app.services.detection_service.anti_spoof.verify", return_value=(True, 0.95)):
            result = subject.analyze_frames(frames, "enroll")
        self.assertEqual(result["dimension"], 512)
        self.assertTrue(result["checks"]["presentation_attack"])
        self.assertEqual(len(result["embedding"]), 512)

    def test_registration_does_not_run_background_segmentation(self):
        subject = runtime()
        images = [webcam_frame(background=(130, 130, 130)) for _ in range(3)]
        with patch("app.services.face_quality.background_metrics", side_effect=AssertionError("background check ran")), \
                patch("app.services.detection_service.anti_spoof.verify", return_value=(True, 0.95)):
            result = subject.analyze_frames(images, "enroll")
        self.assertEqual(result["dimension"], 512)
        self.assertNotIn("background", result["checks"])

    def test_registration_succeeds_when_segmented_background_passes(self):
        subject = runtime()
        images = [webcam_frame() for _ in range(3)]
        with patch("app.services.detection_service.anti_spoof.verify", return_value=(True, 0.95)) as pad:
            result = subject.analyze_frames(images, "enroll")
        self.assertNotIn("background", result["checks"])
        self.assertTrue(result["checks"]["quality"])
        pad.assert_called_once()

    def test_registration_accepts_a_plain_gray_background(self):
        subject = runtime()
        images = [webcam_frame(background=(130, 130, 130)) for _ in range(3)]
        with patch("app.services.detection_service.anti_spoof.verify", return_value=(True, 0.95)):
            result = subject.analyze_frames(images, "enroll")
        self.assertEqual(result["dimension"], 512)

    def test_unstable_enrollment_frames_are_rejected(self):
        subject = runtime()
        frames = [frame(index) for index in range(5)]

        def moving_metrics(image, _box, _landmarks, require_frontal=True, require_background=True, relaxed_position=False):
            result = metrics(image, _box, _landmarks, require_frontal, require_background, relaxed_position)
            result["center_x"] = 0.5 + int(image[0, 0, 0]) * 0.03
            return result

        with patch("app.services.detection_service.FaceRecognitionRuntime._validate_face"), \
                patch("app.services.detection_service.assess_face", side_effect=moving_metrics):
            with self.assertRaises(HTTPException) as raised:
                subject.analyze_frames(frames, "enroll")
        self.assertEqual(raised.exception.detail["code"], "UNSTABLE_POSITION")

    def test_randomized_head_turn_must_match_server_issued_direction_and_return_center(self):
        subject = runtime()
        frames = [frame(index) for index in range(9)]
        with patch("app.services.detection_service.FaceRecognitionRuntime._validate_face"), \
                patch("app.services.detection_service.assess_face", side_effect=metrics), \
                patch("app.services.detection_service.anti_spoof.verify", return_value=(True, 0.95)):
            result = subject.analyze_frames(frames, "verify", "turn_left")
            self.assertTrue(result["checks"]["head_movement"])
            with self.assertRaises(HTTPException) as raised:
                subject.analyze_frames(frames, "verify", "turn_right")
        self.assertEqual(raised.exception.detail["code"], "LIVENESS_CHALLENGE_FAILED")

    def test_single_changed_head_pose_repeated_across_turn_frames_does_not_pass(self):
        subject = runtime()
        frames = [frame(index) for index in range(9)]

        def static_turn_metrics(image, _box, _landmarks, require_frontal=True, require_background=True, relaxed_position=False):
            result = metrics(image, _box, _landmarks, require_frontal, require_background, relaxed_position)
            if 3 <= int(image[0, 0, 0]) <= 5:
                result["yaw"] = 0.2
            return result

        with patch("app.services.detection_service.FaceRecognitionRuntime._validate_face"), \
                patch("app.services.detection_service.assess_face", side_effect=static_turn_metrics):
            with self.assertRaises(HTTPException) as raised:
                subject.analyze_frames(frames, "verify", "turn_left")
        self.assertEqual(raised.exception.detail["code"], "LIVENESS_CHALLENGE_FAILED")

    def test_head_turn_must_return_to_the_original_frontal_pose(self):
        subject = runtime()
        frames = [frame(index) for index in range(9)]

        def no_return_metrics(image, _box, _landmarks, require_frontal=True, require_background=True, relaxed_position=False):
            result = metrics(image, _box, _landmarks, require_frontal, require_background, relaxed_position)
            if int(image[0, 0, 0]) >= 6:
                result["yaw"] = 0.2
            return result

        with patch("app.services.detection_service.FaceRecognitionRuntime._validate_face"), \
                patch("app.services.detection_service.assess_face", side_effect=no_return_metrics):
            with self.assertRaises(HTTPException) as raised:
                subject.analyze_frames(frames, "verify", "turn_left")
        self.assertEqual(raised.exception.detail["code"], "LIVENESS_CHALLENGE_FAILED")

    def test_presentation_attack_failure_fails_closed(self):
        subject = runtime()
        with patch("app.services.detection_service.FaceRecognitionRuntime._validate_face"), \
                patch("app.services.detection_service.assess_face", side_effect=metrics), \
                patch("app.services.detection_service.anti_spoof.verify", return_value=(False, 0.2)):
            with self.assertRaises(HTTPException) as raised:
                subject.analyze_frames([frame(index) for index in range(5)], "enroll")
        self.assertEqual(raised.exception.detail["code"], "PRESENTATION_ATTACK_SUSPECTED")


if __name__ == "__main__":
    unittest.main()
