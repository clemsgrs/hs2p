from pathlib import Path

import numpy as np
import pytest


wsi_mod = pytest.importorskip("hs2p.wsi.wsi")


def test_infer_contour_thickness_is_monotonic_for_slide_size():
    small = wsi_mod.WholeSlideImage._infer_contour_thickness(
        level0_dimensions=(2048, 1536),
        scale=(1 / 16.0, 1 / 16.0),
    )
    large = wsi_mod.WholeSlideImage._infer_contour_thickness(
        level0_dimensions=(65536, 49152),
        scale=(1 / 16.0, 1 / 16.0),
    )

    assert small < large


def test_infer_contour_thickness_clamps_to_bounds():
    min_thickness = wsi_mod.WholeSlideImage._infer_contour_thickness(
        level0_dimensions=(64, 64),
        scale=(1 / 128.0, 1 / 128.0),
    )
    max_thickness = wsi_mod.WholeSlideImage._infer_contour_thickness(
        level0_dimensions=(262144, 262144),
        scale=(1.0, 1.0),
    )

    assert min_thickness == 2
    assert max_thickness == 24


def test_infer_contour_thickness_matches_fixture_calibration():
    thickness = wsi_mod.WholeSlideImage._infer_contour_thickness(
        level0_dimensions=(13824, 12800),
        scale=(1 / 16.0, 1 / 16.0),
    )

    assert thickness == 15


def test_visualize_mask_rejects_line_thickness_argument():
    wsi = object.__new__(wsi_mod.WholeSlideImage)
    wsi.level_dimensions = [(64, 64)]
    wsi.level_downsamples = [(1.0, 1.0)]
    wsi.spacings = [0.5]
    wsi.backend = "asap"
    wsi.get_best_level_for_downsample_custom = lambda downsample: 0
    class _Reader:
        def get_slide(self, spacing):
            return np.zeros((64, 64, 3), dtype=np.uint8)

    wsi.wsi = _Reader()

    contour = np.array([[[5, 5]], [[20, 5]], [[20, 20]], [[5, 20]]], dtype=np.int32)

    with pytest.raises(TypeError):
        wsi.visualize_mask([contour], [[]], line_thickness=5)


def test_visualize_mask_passes_auto_inferred_thickness_to_draw(monkeypatch):
    captured = []
    original_draw_contours = wsi_mod.cv2.drawContours

    def _capture(img, contours, contourIdx, color, thickness, *args, **kwargs):
        captured.append(thickness)
        return original_draw_contours(img, contours, contourIdx, color, thickness, *args, **kwargs)

    monkeypatch.setattr(wsi_mod.cv2, "drawContours", _capture)

    def _make_wsi(level0_dims):
        wsi = object.__new__(wsi_mod.WholeSlideImage)
        wsi.level_dimensions = [level0_dims, (int(level0_dims[0] / 16), int(level0_dims[1] / 16))]
        wsi.level_downsamples = [(1.0, 1.0), (16.0, 16.0)]
        wsi.spacings = [0.5, 8.0]
        wsi.backend = "asap"
        wsi.get_best_level_for_downsample_custom = lambda downsample: 1
        class _Reader:
            def get_slide(self, spacing):
                return np.zeros((int(level0_dims[1] / 16), int(level0_dims[0] / 16), 3), dtype=np.uint8)

        wsi.wsi = _Reader()
        return wsi

    contour = np.array([[[32, 32]], [[256, 32]], [[256, 256]], [[32, 256]]], dtype=np.int32)

    small_wsi = _make_wsi((2048, 1536))
    small_wsi.visualize_mask([contour], [[]])

    large_wsi = _make_wsi((65536, 49152))
    large_wsi.visualize_mask([contour], [[]])

    # First draw of each visualize call is tissue contour.
    small_thickness = captured[0]
    large_thickness = captured[2]

    assert 2 <= small_thickness <= 24
    assert 2 <= large_thickness <= 24
    assert large_thickness > small_thickness
