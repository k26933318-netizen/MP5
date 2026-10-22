import numpy as np

from overlay import shape_overlay
from overlay.shape_overlay import draw_face_shape, draw_face_shapes


def test_draw_face_shape_returns_frame_with_transparent_shape():
    frame = np.zeros((120, 160, 4), dtype=np.uint8)

    result = draw_face_shape(frame, (40, 25, 100, 95))

    assert result is frame
    assert np.any(frame[:, :, 3] > 0)
    assert np.any(frame[15:25, 30:42, 3] > 0)


def test_draw_face_shapes_supports_multiple_boxes_and_track_isolation():
    frame = np.zeros((160, 240, 4), dtype=np.uint8)

    result = draw_face_shapes(
        frame,
        [((20, 25, 80, 110), None), ((150, 30, 215, 115), None)],
        track_key="two-faces",
    )

    assert result is frame
    assert np.any(frame[:, 20:120, 3] > 0)
    assert np.any(frame[:, 140:230, 3] > 0)


def test_all_configured_shapes_render_clipped_tilted_faces():
    shape_types = (
        "ellipse",
        "circle",
        "rectangle",
        "rounded_rectangle",
        "corner_brackets",
    )

    for index, shape in enumerate(shape_types):
        frame = np.zeros((80, 96, 4), dtype=np.uint8)
        draw_face_shapes(
            frame,
            [((-8, 6, 34, 65), np.asarray([[8, 20], [22, 24]], dtype=np.float32))],
            track_key=f"shape-{index}",
            shape=shape,
        )

        assert np.any(frame[:, :, 3] > 0), shape


def test_empty_detection_clears_track_before_face_returns():
    first_frame = np.zeros((120, 160, 4), dtype=np.uint8)
    draw_face_shapes(first_frame, [((40, 25, 100, 95), None)], track_key="disappearing")

    empty_frame = np.zeros_like(first_frame)
    draw_face_shapes(empty_frame, [], track_key="disappearing")
    returned_frame = np.zeros_like(first_frame)
    draw_face_shapes(
        returned_frame,
        [((50, 25, 110, 95), None)],
        track_key="disappearing",
    )

    assert np.any(returned_frame[:, :, 3] > 0)
    tracked_box = next(iter(shape_overlay._tracks["disappearing"].values())).box
    np.testing.assert_allclose(tracked_box, [50, 25, 110, 95])


def test_box_position_is_exponentially_smoothed_per_track():
    key = "smoothing"
    draw_face_shapes(
        np.zeros((120, 160, 4), dtype=np.uint8),
        [((20, 20, 60, 80), None)],
        track_key=key,
    )
    draw_face_shapes(
        np.zeros((120, 160, 4), dtype=np.uint8),
        [((40, 20, 80, 80), None)],
        track_key=key,
    )

    tracked_box = next(iter(shape_overlay._tracks[key].values())).box
    np.testing.assert_allclose(tracked_box, [27, 20, 67, 80])


def test_filled_shape_uses_configured_alpha_on_bgr_frames(monkeypatch):
    monkeypatch.setattr(shape_overlay, "SHAPE_FILLED", True)
    monkeypatch.setattr(shape_overlay, "SHAPE_ALPHA", 0.25)
    frame = np.zeros((80, 96, 3), dtype=np.uint8)

    draw_face_shapes(frame, [((20, 15, 70, 65), None)], track_key="filled", shape="rectangle")

    np.testing.assert_allclose(frame[40, 45], [64, 54, 41], atol=1)


def test_classic_style_keeps_the_ellipse_and_configured_color(monkeypatch):
    monkeypatch.setitem(shape_overlay.TRACER_CONFIG, "style", "classic")
    frame = np.zeros((120, 160, 4), dtype=np.uint8)

    draw_face_shapes(
        frame,
        [((40, 25, 100, 95), None)],
        track_key="classic-style",
        shape="ellipse",
    )

    assert frame[17, 70, 3] > 0
    assert frame[56, 70, 3] == 0
    np.testing.assert_array_equal(frame[17, 70, :3], shape_overlay.SHAPE_COLOR)


def test_draw_face_shapes_rejects_invalid_shape():
    frame = np.zeros((40, 40, 4), dtype=np.uint8)

    try:
        draw_face_shapes(frame, [], shape="triangle")
    except ValueError as error:
        assert "Unsupported face shape" in str(error)
    else:
        raise AssertionError("Unsupported shape type was accepted.")
