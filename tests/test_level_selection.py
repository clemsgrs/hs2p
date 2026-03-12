from __future__ import annotations

from pathlib import Path

import pytest

from hs2p.api import FilterConfig, SegmentationConfig, TilingConfig
import hs2p.wsi as wsi_api
from hs2p.wsi import SamplingParameters
from hs2p.wsi.wsi import WholeSlideImage


def _segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(
        downsample=2,
        sthresh=8,
        sthresh_up=255,
        mthresh=3,
        close=0,
        use_otsu=False,
        use_hsv=False,
    )


def _filter_config() -> FilterConfig:
    return FilterConfig(
        ref_tile_size=4,
        a_t=0,
        a_h=0,
        max_n_holes=8,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def _tiling_config(*, spacing: float = 1.0, tolerance: float = 0.01) -> TilingConfig:
    return TilingConfig(
        target_spacing_um=spacing,
        target_tile_size_px=8,
        tolerance=tolerance,
        overlap=0.0,
        tissue_threshold=0.0,
        drop_holes=False,
        use_padding=False,
        backend="asap",
    )


def _sampling_parameters() -> SamplingParameters:
    return SamplingParameters(
        pixel_mapping={"background": 0, "tissue": 1},
        color_mapping={"background": None, "tissue": None},
        tissue_percentage={"background": None, "tissue": 0.0},
    )


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
            segment_params=_segmentation_config(),
            tiling_params=_tiling_config(spacing=0.5, tolerance=0.05),
            filter_params=_filter_config(),
            sampling_params=_sampling_parameters(),
            disable_tqdm=True,
            num_workers=1,
        )
