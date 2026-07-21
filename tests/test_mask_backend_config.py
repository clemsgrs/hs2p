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


def test_default_yaml_defines_mask_backend_auto():
    assert default_config.tiling.mask_backend == "auto"


def test_omitting_mask_backend_defaults_to_auto():
    tiling = _tiling()
    assert tiling.mask_backend == "auto"
    assert tiling.requested_mask_backend == "auto"


@pytest.mark.parametrize("name", ["auto", "cucim", "asap", "openslide", "vips"])
def test_supported_backend_names_are_accepted(name):
    tiling = _tiling(backend=name, mask_backend=name)
    assert tiling.backend == name
    assert tiling.mask_backend == name


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


def test_requested_backends_preserved_across_replace():
    from dataclasses import replace

    tiling = _tiling(backend="auto", mask_backend="auto")
    resolved = replace(tiling, backend="cucim", mask_backend="openslide")
    # requested provenance must survive the auto-resolution replace()
    assert resolved.requested_backend == "auto"
    assert resolved.requested_mask_backend == "auto"
    assert resolved.backend == "cucim"
    assert resolved.mask_backend == "openslide"
