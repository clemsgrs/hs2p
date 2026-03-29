from types import SimpleNamespace

import numpy as np

import hs2p.wsi.segmentation as segmod


def _make_reader(read_level):
    return SimpleNamespace(read_level=read_level)


def _patch_cv2_passthrough(monkeypatch):
    monkeypatch.setattr(
        segmod.cv2,
        "cvtColor",
        lambda img, code: np.zeros((*img.shape[:2], 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        segmod.cv2,
        "medianBlur",
        lambda channel, ksize: np.zeros(channel.shape, dtype=np.uint8),
    )
    monkeypatch.setattr(
        segmod.cv2,
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
    captured_shape: dict = {}

    def _fake_cvt_color(img, code):
        captured_shape["shape"] = img.shape
        return np.zeros((2, 2, 3), dtype=np.uint8)

    reader = _make_reader(lambda level: np.zeros((2, 2, 4), dtype=np.uint8))
    _patch_cv2_passthrough(monkeypatch)
    monkeypatch.setattr(segmod.cv2, "cvtColor", _fake_cvt_color)

    annotation_mask = segmod.segment_tissue(
        reader=reader,
        segment_params=_SEG_PARAMS,
        seg_level=0,
    )

    assert captured_shape["shape"] == (2, 2, 3)
    np.testing.assert_array_equal(annotation_mask["tissue"], np.ones((2, 2), dtype=np.uint8))


def test_segment_tissue_reads_from_correct_seg_level(monkeypatch):
    """
    segment_tissue must call read_level(seg_level), not read_level(0).

    When the best seg level is 4 (the low-res thumbnail), reading from level 0
    would decode the full-resolution image — orders of magnitude slower.
    This was the root cause of a ~40s segmentation bug on slides with bad
    mpp metadata where spacing_at_level_0 was overridden.

    Input:  5-level pyramid, get_best_level_for_downsample_custom returns 4
    Output: read_level is called with level=4, not level=0
    """
    levels_read: list[int] = []

    reader = _make_reader(
        lambda level: (levels_read.append(level), np.zeros((2, 2, 3), dtype=np.uint8))[1]
    )
    _patch_cv2_passthrough(monkeypatch)

    segmod.segment_tissue(
        reader=reader,
        segment_params=_SEG_PARAMS,
        seg_level=4,
    )

    assert levels_read == [4], (
        f"Expected get_slide(4) for a 5-level pyramid with seg_level=4, got {levels_read}. "
        "Passing level 0 would read the full-resolution image instead of the thumbnail."
    )
