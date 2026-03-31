from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.wsi.wsi as wsimod


def _make_dummy(raw_spacings: list[float], seg_level: int = 0):
    """Build a minimal WSI stand-in for segment_tissue tests."""
    dummy = SimpleNamespace()
    dummy.raw_spacings = raw_spacings
    dummy.get_slide = lambda level: np.zeros((2, 2, 3), dtype=np.uint8)
    dummy.get_best_level_for_downsample_custom = lambda downsample: seg_level
    return dummy


def _patch_cv2_passthrough(monkeypatch):
    monkeypatch.setattr(
        wsimod.cv2,
        "cvtColor",
        lambda img, code: np.zeros((*img.shape[:2], 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        wsimod.cv2,
        "medianBlur",
        lambda channel, ksize: np.zeros(channel.shape, dtype=np.uint8),
    )
    monkeypatch.setattr(
        wsimod.cv2,
        "threshold",
        lambda src, thresh, maxval, threshold_type: (0, np.ones((2, 2), dtype=np.uint8)),
    )


_SEG_PARAMS = SimpleNamespace(
    downsample=64,
    sthresh=8,
    sthresh_up=255,
    mthresh=7,
    close=0,
    use_otsu=False,
    use_hsv=False,
)


def test_segment_tissue_strips_alpha_channel_before_hsv_conversion(monkeypatch):
    """Alpha channel is stripped before cvtColor so it receives a 3-channel image."""
    dummy = _make_dummy(raw_spacings=[0.5])
    dummy.get_slide = lambda level: np.zeros((2, 2, 4), dtype=np.uint8)

    captured_shape: dict = {}

    def _fake_cvt_color(img, code):
        captured_shape["shape"] = img.shape
        return np.zeros((2, 2, 3), dtype=np.uint8)

    _patch_cv2_passthrough(monkeypatch)
    monkeypatch.setattr(wsimod.cv2, "cvtColor", _fake_cvt_color)

    seg_level = wsimod.WSI.segment_tissue(dummy, _SEG_PARAMS)

    assert seg_level == 0
    assert captured_shape["shape"] == (2, 2, 3)
    np.testing.assert_array_equal(
        dummy.annotation_mask["tissue"], np.ones((2, 2), dtype=np.uint8)
    )


def test_segment_tissue_reads_from_correct_seg_level(monkeypatch):
    """
    segment_tissue must call get_slide(seg_level), not get_slide(0).

    When the best seg level is 4 (the low-res thumbnail), reading from level 0
    would decode the full-resolution image — orders of magnitude slower.
    This was the root cause of a ~40s segmentation bug on slides with bad
    mpp metadata where spacing_at_level_0 was overridden.

    Input:  5-level pyramid, get_best_level_for_downsample_custom returns 4
    Output: get_slide is called with level=4, not level=0
    """
    levels_read: list[int] = []

    dummy = _make_dummy(raw_spacings=[0.5, 1.0, 2.0, 4.0, 8.0], seg_level=4)
    dummy.get_slide = lambda level: (levels_read.append(level), np.zeros((2, 2, 3), dtype=np.uint8))[1]

    _patch_cv2_passthrough(monkeypatch)

    wsimod.WSI.segment_tissue(dummy, _SEG_PARAMS)

    assert levels_read == [4], (
        f"Expected get_slide(4) for a 5-level pyramid with seg_level=4, got {levels_read}. "
        "Passing level 0 would read the full-resolution image instead of the thumbnail."
    )
