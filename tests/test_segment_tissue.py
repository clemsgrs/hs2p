from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hs2p.wsi.wsi as wsimod


def test_segment_tissue_strips_alpha_channel_before_hsv_conversion(monkeypatch):
    captured = {}

    dummy = SimpleNamespace()
    dummy.wsi = SimpleNamespace(
        get_slide=lambda spacing: np.zeros((2, 2, 4), dtype=np.uint8)
    )
    dummy.get_best_level_for_downsample_custom = lambda downsample: 0
    dummy.get_level_spacing = lambda level: 0.5

    def _fake_cvt_color(img, code):
        captured["shape"] = img.shape
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(wsimod.cv2, "cvtColor", _fake_cvt_color)
    monkeypatch.setattr(wsimod.cv2, "medianBlur", lambda channel, ksize: np.zeros((2, 2), dtype=np.uint8))
    monkeypatch.setattr(
        wsimod.cv2,
        "threshold",
        lambda src, thresh, maxval, threshold_type: (0, np.ones((2, 2), dtype=np.uint8)),
    )

    seg_level = wsimod.WholeSlideImage.segment_tissue(
        dummy,
        SimpleNamespace(
            downsample=64,
            sthresh=8,
            sthresh_up=255,
            mthresh=7,
            close=0,
            use_otsu=False,
            use_hsv=False,
        ),
    )

    assert seg_level == 0
    assert captured["shape"] == (2, 2, 3)
    np.testing.assert_array_equal(dummy.annotation_mask["tissue"], np.ones((2, 2), dtype=np.uint8))
