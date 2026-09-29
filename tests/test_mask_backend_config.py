"""Configuration-validation coverage for the independent mask backend (#163)."""
import pytest

from hs2p.configs import TilingConfig, default_config
from hs2p.configs.resolvers import resolve_tiling_config


def _tiling(**overrides):
    params = dict(
        requested_spacing_um=0.5,
        requested_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        min_coverage={"tissue": 0.1},
    )
    params.update(overrides)
    return TilingConfig(**params)


@pytest.mark.parametrize("bad", [None, "", "tiff", "unknown", "CuCIM "])
def test_unknown_or_null_slide_backend_fails(bad):
    with pytest.raises((ValueError, TypeError)):
        _tiling(backend=bad)


@pytest.mark.parametrize("bad", [None, "", "tiff", "unknown"])
def test_unknown_or_null_mask_backend_fails(bad):
    with pytest.raises((ValueError, TypeError)):
        _tiling(mask_backend=bad)


def test_resolve_tiling_config_threads_mask_backend(monkeypatch):
    cfg = default_config.copy()
    cfg.tiling.backend = "openslide"
    cfg.tiling.mask_backend = "asap"
    tiling = resolve_tiling_config(cfg)
    assert tiling.backend == "openslide"
    assert tiling.mask_backend == "asap"
    assert tiling.requested_backend == "openslide"
    assert tiling.requested_mask_backend == "asap"
