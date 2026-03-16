from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.wsi.wsi as wsimod


def _make_dummy(wsi_spacings: list[float], seg_level: int = 0):
    """Build a minimal WholeSlideImage stand-in for segment_tissue tests."""
    captured: dict = {}

    dummy = SimpleNamespace()
    dummy.wsi = SimpleNamespace(
        spacings=wsi_spacings,
        get_slide=lambda spacing: np.zeros((2, 2, 4), dtype=np.uint8),
    )
    dummy.get_best_level_for_downsample_custom = lambda downsample: seg_level
    dummy.get_level_spacing = lambda level: 0.5  # hs2p-overridden spacing
    dummy._captured = captured
    return dummy, captured


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
    dummy, _ = _make_dummy(wsi_spacings=[0.5])
    captured_shape: dict = {}

    def _fake_cvt_color(img, code):
        captured_shape["shape"] = img.shape
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(wsimod.cv2, "cvtColor", _fake_cvt_color)
    _patch_cv2_passthrough(monkeypatch)
    monkeypatch.setattr(wsimod.cv2, "cvtColor", _fake_cvt_color)  # re-apply after passthrough

    seg_level = wsimod.WholeSlideImage.segment_tissue(dummy, _SEG_PARAMS)

    assert seg_level == 0
    assert captured_shape["shape"] == (2, 2, 3)
    np.testing.assert_array_equal(
        dummy.annotation_mask["tissue"], np.ones((2, 2), dtype=np.uint8)
    )


def test_segment_tissue_uses_native_backend_spacing_not_override(monkeypatch):
    """
    When spacing_at_level_0 is overridden, the backend still holds raw-metadata spacings.
    segment_tissue must call wsi.get_slide with the backend's native spacing for seg_level
    so the backend's internal level lookup resolves to the correct (low-res) pyramid level
    rather than level 0.

    Concrete case: backend sees spacings=[16000.0] (wrong mpp from file metadata),
    hs2p overrides spacing to 8.0 via spacing_at_level_0.
    get_slide must be called with 16000.0 (native), not 8.0 (override).
    """
    native_spacing = 16000.0
    dummy, _ = _make_dummy(wsi_spacings=[native_spacing], seg_level=0)

    spacings_passed: list[float] = []
    original_get_slide = dummy.wsi.get_slide

    def _recording_get_slide(spacing):
        spacings_passed.append(spacing)
        return original_get_slide(spacing)

    dummy.wsi.get_slide = _recording_get_slide
    _patch_cv2_passthrough(monkeypatch)

    wsimod.WholeSlideImage.segment_tissue(dummy, _SEG_PARAMS)

    assert len(spacings_passed) == 1, "get_slide should be called exactly once"
    assert spacings_passed[0] == native_spacing, (
        f"Expected native backend spacing {native_spacing}, got {spacings_passed[0]}. "
        "segment_tissue must not pass the hs2p-overridden spacing to the backend."
    )
