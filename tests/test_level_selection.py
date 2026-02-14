from __future__ import annotations

from pathlib import Path

import pytest

import hs2p.wsi as wsi_api
from hs2p.wsi.wsi import WholeSlideImage
from params import make_filter_params, make_sampling_params, make_segment_params, make_tiling_params


def test_get_best_level_for_spacing_returns_within_tolerance_level(fake_backend):
    mask = pytest.importorskip("numpy").zeros((16, 16, 1), dtype="uint8")
    fake_backend(mask)
    wsi = WholeSlideImage(path=Path("synthetic-slide.tif"), backend="asap")

    level, within_tolerance = wsi.get_best_level_for_spacing(target_spacing=2.1, tolerance=0.10)

    assert level == 1
    assert within_tolerance is True


def test_get_best_level_for_spacing_falls_back_to_finer_level_when_closest_is_too_coarse(fake_backend):
    mask = pytest.importorskip("numpy").zeros((16, 16, 1), dtype="uint8")
    fake_backend(mask)
    wsi = WholeSlideImage(path=Path("synthetic-slide.tif"), backend="asap")

    level, within_tolerance = wsi.get_best_level_for_spacing(target_spacing=3.5, tolerance=0.01)

    assert level == 1
    assert within_tolerance is False
    assert wsi.get_level_spacing(level) <= 3.5


def test_extract_coordinates_raises_when_target_spacing_is_below_level0_beyond_tolerance(monkeypatch):
    class GuardOnlyWSI:
        def __init__(self, *args, **kwargs):
            self.spacings = [1.0]

    monkeypatch.setattr(wsi_api, "WholeSlideImage", GuardOnlyWSI)

    with pytest.raises(ValueError, match="Desired spacing"):
        wsi_api.extract_coordinates(
            wsi_path=Path("synthetic-slide.tif"),
            mask_path=Path("synthetic-mask.tif"),
            backend="asap",
            segment_params=make_segment_params(),
            tiling_params=make_tiling_params(spacing=0.5, tolerance=0.05),
            filter_params=make_filter_params(),
            sampling_params=make_sampling_params(
                pixel_mapping={"background": 0, "tissue": 1},
                tissue_percentage={"background": None, "tissue": 0.0},
            ),
            disable_tqdm=True,
            num_workers=1,
        )
