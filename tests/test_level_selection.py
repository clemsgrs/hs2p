from pathlib import Path

import pytest

from hs2p.api import TilingConfig
from hs2p.wsi.wsi import WSI


def _tiling_config(*, spacing: float = 1.0, tolerance: float = 0.01) -> TilingConfig:
    return TilingConfig(
        requested_spacing_um=spacing,
        requested_tile_size_px=8,
        tolerance=tolerance,
        overlap=0.0,
        tissue_threshold=0.0,
        backend="asap",
    )


def test_get_best_level_for_spacing_returns_within_tolerance_level(fake_backend):
    mask = pytest.importorskip("numpy").zeros((16, 16, 1), dtype="uint8")
    fake_backend(mask)
    wsi = WSI(path=Path("synthetic-slide.tif"), backend="asap")

    level, within_tolerance = wsi.get_best_level_for_spacing(
        requested_spacing_um=2.1, tolerance=0.10
    )

    assert level == 1
    assert within_tolerance is True


def test_get_best_level_for_spacing_falls_back_to_finer_level_when_closest_is_too_coarse(
    fake_backend,
):
    mask = pytest.importorskip("numpy").zeros((16, 16, 1), dtype="uint8")
    fake_backend(mask)
    wsi = WSI(path=Path("synthetic-slide.tif"), backend="asap")

    level, within_tolerance = wsi.get_best_level_for_spacing(
        requested_spacing_um=3.5, tolerance=0.01
    )

    assert level == 1
    assert within_tolerance is False
    assert wsi.get_level_spacing(level) <= 3.5
