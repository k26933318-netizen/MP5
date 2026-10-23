import numpy as np
import pytest

from liveness import BlinkDetectorConfig, BlinkLivenessDetector


def _eye_landmarks(ear):
    points = np.zeros((12, 2), dtype="float32")
    for start in (0, 6):
        points[start] = [0, 0]
        points[start + 1] = [2, -ear * 5]
        points[start + 2] = [8, -ear * 5]
        points[start + 3] = [10, 0]
        points[start + 4] = [8, ear * 5]
        points[start + 5] = [2, ear * 5]
    return points


def _test_detector(**overrides):
    values = {
        "ear_threshold": 0.21,
        "consec_frames": 2,
        "required_blinks": 2,
        "time_window_seconds": 8.0,
        "timeout_seconds": 10.0,
        "calibration_seconds": 0.6,
        "smoothing_window": 3,
        "min_blink_seconds": 0.08,
        "max_blink_seconds": 0.8,
        "min_ear_variation": 0.025,
        "min_blink_interval_seconds": 0.25,
    }
    values.update(overrides)
    return BlinkLivenessDetector(BlinkDetectorConfig(**values))


def _run(detector, values, start=0.0, step=0.2, face_id="person"):
    result = None
    for index, ear in enumerate(values):
        result = detector.update(
            None,
            _eye_landmarks(ear),
            face_id=face_id,
            timestamp=start + index * step,
        )
    return result


def test_ear_is_mean_of_both_six_point_eye_contours():
    landmarks = _eye_landmarks(0.30)
    landmarks[6:] = _eye_landmarks(0.24)[6:]

    assert BlinkLivenessDetector.eye_aspect_ratio(landmarks) == pytest.approx(0.27)


def test_two_closed_then_reopened_blinks_pass():
    detector = _test_detector()
    sequence = [0.30] * 5 + [0.10, 0.10, 0.30, 0.30] + [0.10, 0.10, 0.30, 0.30]

    result = _run(detector, sequence)

    assert result.is_live
    assert result.blink_count == 2
    assert result.status == "LIVE"


def test_short_natural_blinks_are_detected_at_ten_frames_per_second():
    detector = _test_detector()
    sequence = [0.30] * 7 + [0.10, 0.10, 0.30, 0.30, 0.30] + [0.10, 0.10, 0.30, 0.30]

    result = _run(detector, sequence, step=0.1)

    assert result.is_live
    assert result.blink_count == 2


def test_calibrates_lower_ear_baseline_from_low_quality_landmarks():
    detector = _test_detector(smoothing_window=1)
    sequence = [0.18] * 7 + [0.11, 0.11, 0.18, 0.18, 0.18] + [0.11, 0.11, 0.18, 0.18]

    result = _run(detector, sequence, step=0.1)

    assert result.is_live
    assert result.blink_count == 2


def test_no_blink_static_eyes_do_not_pass_and_expire():
    detector = _test_detector(time_window_seconds=2.0, timeout_seconds=3.0)

    result = _run(detector, [0.30] * 12, step=0.25)
    timed_out = _test_detector(time_window_seconds=2.0, timeout_seconds=3.0)
    expired = _run(timed_out, [0.30] * 17, step=0.2)

    assert not result.is_live
    assert expired.status == "SPOOF SUSPECTED"
    restarted = timed_out.update(None, _eye_landmarks(0.30), face_id="person", timestamp=4.0)
    assert restarted.status == "Verifying..."


def test_long_eye_closure_is_not_counted():
    detector = _test_detector(max_blink_seconds=0.6)
    sequence = [0.30] * 5 + [0.10] * 6 + [0.30] * 5

    result = _run(detector, sequence)

    assert result.blink_count == 0
    assert not result.is_live


def test_single_frame_noise_does_not_count_as_blink():
    detector = _test_detector()
    sequence = [0.30] * 5 + [0.10] + [0.30] * 8

    result = _run(detector, sequence)

    assert result.blink_count == 0
    assert not result.is_live


def test_too_short_closure_is_not_counted():
    detector = _test_detector(smoothing_window=1, min_blink_seconds=0.08)
    result = None
    for timestamp, ear in ((0.0, 0.30), (0.2, 0.30), (0.4, 0.30), (0.6, 0.30),
                           (0.62, 0.10), (0.64, 0.10), (0.68, 0.30)):
        result = detector.update(None, _eye_landmarks(ear), timestamp=timestamp)

    assert result.blink_count == 0
    assert not result.is_live


def test_face_tracks_do_not_share_counts_and_missing_track_resets():
    detector = _test_detector(required_blinks=1)
    _run(detector, [0.30] * 5 + [0.10, 0.10, 0.30], face_id="person-a")
    person_b = _run(detector, [0.30] * 5, face_id="person-b", start=1.0)

    assert person_b.blink_count == 0
    detector.reset_missing({"person-b"})
    assert detector.current_result("person-a").blink_count == 0


def test_insightface_106_uses_eye_points_not_brow_or_mouth_points():
    landmarks = np.zeros((106, 2), dtype="float32")
    eye_points = _eye_landmarks(0.31)
    landmarks[[63, 64, 65, 66, 68, 70]] = eye_points[:6]
    landmarks[[76, 77, 78, 79, 81, 83]] = eye_points[6:]
    landmarks[[33, 35, 37, 39, 41, 43]] = 1000
    landmarks[[87, 89, 91, 93, 95, 97]] = -1000

    assert BlinkLivenessDetector.eye_aspect_ratio(landmarks) == pytest.approx(0.31)


def test_ear_around_1_1_is_rejected_without_spoofing():
    detector = _test_detector(smoothing_window=1)
    result = _run(detector, [1.1] * 60, step=0.2)

    assert not result.is_live
    assert result.status == "Verifying..."


def test_long_detection_gap_restarts_the_continuous_quality_window():
    detector = _test_detector(smoothing_window=1)
    detector.update(None, _eye_landmarks(0.30), timestamp=0.0)

    after_gap = detector.update(None, _eye_landmarks(0.30), timestamp=4.0)

    assert after_gap.status == "Verifying..."
    assert detector.debug_info()["baseline"] is None


def test_one_frame_ear_outlier_does_not_create_a_blink():
    detector = _test_detector(smoothing_window=1)
    result = _run(detector, [0.30] * 5 + [0.04] + [0.30] * 8)

    assert result.blink_count == 0
    assert not result.is_live


def test_single_blink_works_with_default_required_blink_count():
    detector = BlinkLivenessDetector()
    sequence = [0.30] * 18 + [0.12, 0.12, 0.30, 0.30]

    result = _run(detector, sequence, step=0.1)

    assert result.is_live
    assert result.blink_count == 1


def test_slow_blink_within_four_hundred_milliseconds_is_detected():
    detector = _test_detector(
        required_blinks=1,
        smoothing_window=1,
        max_blink_seconds=0.4,
    )
    sequence = [0.30] * 7 + [0.10] * 4 + [0.30] * 4

    result = _run(detector, sequence, step=0.1)

    assert result.is_live
    assert result.blink_count == 1


def test_low_quality_frame_does_not_become_a_spoof_decision():
    detector = _test_detector()
    landmarks = np.zeros((106, 2), dtype="float32")
    landmarks[:, 0] = np.linspace(80, 220, 106)
    landmarks[:, 1] = np.linspace(80, 220, 106)
    eye_points = _eye_landmarks(0.30)
    landmarks[[63, 64, 65, 66, 68, 70]] = eye_points[:6] + [100, 100]
    landmarks[[76, 77, 78, 79, 81, 83]] = eye_points[6:] + [100, 100]
    frame = np.zeros((300, 300, 3), dtype="uint8")

    result = detector.update(frame, landmarks, timestamp=0.0)

    assert result.status.startswith("Low quality")
    assert not result.is_live
