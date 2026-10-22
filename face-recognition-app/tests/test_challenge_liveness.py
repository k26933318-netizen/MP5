from dataclasses import replace

import numpy as np
import pytest
from liveness import ChallengeLivenessDetector, DEFAULT_CONFIG


def _landmarks(ear=0.30, yaw=0.0, mouth_ratio=0.10):
    points = np.zeros((106, 2), dtype="float32")
    points[:33, 0] = np.linspace(50, 250, 33)
    points[:33, 1] = np.linspace(40, 220, 33)
    points[51:63, 0] = 150 + yaw * 200
    points[51:63, 1] = np.linspace(90, 120, 12)
    points[96:104, 0] = np.linspace(100, 200, 8)
    points[96:104, 1] = np.linspace(150, 150 + mouth_ratio * 100, 8)
    for indices, start in (((63, 64, 65, 66, 68, 70), 10), ((76, 77, 78, 79, 81, 83), 40)):
        points[indices[0]] = [start, 10]
        points[indices[1]] = [start + 2, 10 - ear * 5]
        points[indices[2]] = [start + 8, 10 - ear * 5]
        points[indices[3]] = [start + 10, 10]
        points[indices[4]] = [start + 8, 10 + ear * 5]
        points[indices[5]] = [start + 2, 10 + ear * 5]
    return points


def _detector(**overrides):
    values = {"calibration_seconds": 0.3, "smoothing_window": 3, "challenge_timeout_seconds": 1.0}
    values.update(overrides)
    config = replace(DEFAULT_CONFIG, **values)
    return ChallengeLivenessDetector(
        config,
        action_picker=lambda actions: ["blink", "head_turn_left", "mouth_open"],
    )


def _feed(detector, ear=0.30, yaw=0.0, mouth_ratio=0.10, timestamp=0.0, frame=None):
    return detector.update(frame, _landmarks(ear, yaw, mouth_ratio), timestamp=timestamp)


def _calibrate(detector):
    for timestamp in (0.0, 0.1, 0.2, 0.3):
        result = _feed(detector, timestamp=timestamp)
    assert result.instruction == "Blink twice"


def test_two_normal_blinks_pass_with_timestamps():
    detector = _detector(challenge_success_count=1, blink_action_required_count=2)
    _calibrate(detector)
    result = None
    for timestamp, ear in ((0.4, 0.30), (0.5, 0.12), (0.6, 0.30), (0.7, 0.30),
                           (0.8, 0.12), (0.9, 0.30), (1.0, 0.30)):
        result = _feed(detector, ear=ear, timestamp=timestamp)

    assert result is not None and result.is_live
    assert result.blink_count == 2
    assert result.state == "live"


def test_slow_blinks_are_accepted_at_ten_fps():
    detector = _detector(challenge_success_count=1, blink_action_required_count=2)
    _calibrate(detector)
    result = None
    sequence = [(0.4, 0.30), (0.5, 0.12), (0.6, 0.12), (0.7, 0.12), (0.8, 0.30),
                (0.9, 0.30), (1.0, 0.12), (1.1, 0.12), (1.2, 0.12), (1.3, 0.30)]
    for timestamp, ear in sequence:
        result = _feed(detector, ear=ear, timestamp=timestamp)

    assert result is not None and result.is_live
    assert result.blink_count == 2


def test_noisy_open_eye_signal_does_not_count_as_blink():
    detector = _detector(challenge_success_count=1)
    _calibrate(detector)
    result = None
    for index, ear in enumerate((0.29, 0.31, 0.28, 0.32, 0.30, 0.29, 0.31, 0.30), start=4):
        result = _feed(detector, ear=ear, timestamp=index * 0.1)

    assert result is not None
    assert not result.is_live
    assert result.blink_count == 0


def test_natural_blink_passes_while_randomized_prompt_requests_a_turn():
    detector = ChallengeLivenessDetector(
        replace(DEFAULT_CONFIG, calibration_seconds=0.3, smoothing_window=1),
        action_picker=lambda actions: ["head_turn_left", "mouth_open", "blink"],
    )
    for timestamp in (0.0, 0.1, 0.2, 0.3):
        result = _feed(detector, timestamp=timestamp)
    assert result.instruction == "Turn your head slightly left"

    _feed(detector, timestamp=0.4)
    _feed(detector, ear=0.12, timestamp=0.5)
    result = _feed(detector, ear=0.30, timestamp=0.6)

    assert result.is_live
    assert result.blink_count == 1


def test_head_turn_and_mouth_open_are_two_independent_successes(monkeypatch):
    import liveness.challenge as challenge_module

    monkeypatch.setattr(challenge_module, "FAST_MODE", False)
    config = replace(
        DEFAULT_CONFIG,
        calibration_seconds=0.3,
        smoothing_window=3,
        challenge_success_count=2,
    )
    detector = ChallengeLivenessDetector(
        config,
        action_picker=lambda actions: ["head_turn_left", "mouth_open", "blink"],
    )
    for timestamp in (0.0, 0.1, 0.2, 0.3):
        result = _feed(detector, timestamp=timestamp)
    assert result.instruction == "Turn your head slightly left"

    for timestamp in (0.4, 0.5, 0.6):
        result = _feed(detector, yaw=0.08, timestamp=timestamp)
    assert result.instruction == "Open your mouth"

    for timestamp in (0.7, 0.8, 0.9):
        result = _feed(detector, yaw=0.08, mouth_ratio=0.25, timestamp=timestamp)
    assert result.is_live
    assert result.progress == result.required_progress == 2


def test_high_ear_landmark_failure_is_low_quality_not_live():
    detector = _detector(challenge_success_count=1)
    result = detector.update(None, _landmarks(ear=1.1), timestamp=0.0)

    assert result.state == "low_quality"
    assert result.instruction == "Move closer"
    assert not result.is_live


def test_dark_frame_pauses_as_quality_guidance():
    detector = _detector()
    frame = np.zeros((260, 260, 3), dtype="uint8")
    result = detector.update(frame, _landmarks(), timestamp=0.0)

    assert result.state == "low_quality"
    assert result.instruction == "More light needed"
    assert not result.is_live


def test_blurry_frame_requests_retry_without_failing_liveness():
    detector = _detector()
    frame = np.full((260, 260, 3), 80, dtype="uint8")
    result = detector.update(frame, _landmarks(), timestamp=0.0)

    assert result.state == "low_quality"
    assert result.instruction == "Try again"
    assert not result.is_live


def test_low_quality_time_does_not_exhaust_the_action_window():
    detector = _detector(challenge_timeout_seconds=0.2)
    _calibrate(detector)
    dark = np.zeros((260, 260, 3), dtype="uint8")
    for index in range(4, 40):
        result = _feed(detector, timestamp=index * 0.1, frame=dark)

    assert result.state == "low_quality"
    assert result.retries == 0


def test_metric_spike_is_rejected_before_median_smoothing():
    detector = _detector(challenge_success_count=1)
    _calibrate(detector)
    result = detector.update(None, _landmarks(yaw=0.5), timestamp=0.4)

    assert result.state == "checking"
    assert result.progress == 0
    assert detector.debug_info()["yaw"] == pytest.approx(0.0)


def test_missing_landmarks_preserve_legacy_status_and_request_recovery():
    detector = _detector()
    result = detector.update(None, np.zeros((5, 2)), timestamp=0.0)

    assert result.state == "low_quality"
    assert result.status == "LANDMARKS UNAVAILABLE"
    assert result.instruction == "Move closer"


def test_no_action_fails_only_after_two_retries():
    config = replace(
        DEFAULT_CONFIG,
        calibration_seconds=0.3,
        smoothing_window=1,
        challenge_timeout_seconds=0.2,
        challenge_max_retries=2,
    )
    detector = ChallengeLivenessDetector(
        config,
        action_picker=lambda actions: ["head_turn_left", "mouth_open", "blink"],
    )
    for timestamp in (0.0, 0.1, 0.2, 0.3):
        _feed(detector, timestamp=timestamp)
    result = None
    for index in range(4, 160):
        result = _feed(detector, timestamp=index * 0.1)
        if result.state == "failed":
            break

    assert result is not None and result.state == "failed"
    assert result.instruction == "Try again"
    assert result.retries == 2
    assert result.status != "SPOOF SUSPECTED"