"""Fail-fast guard for oversized (non-pyramidal) mask reads.

A mask whose nearest pyramid level to the requested segmentation spacing is still huge
(because the mask lacks a coarse pyramid level) would force ``read_region`` to materialise a
multi-GB raster, OOM-killing the job. Both preprocessing paths read through ``hs2p.mask``,
which applies the fixed 256 Mpx cap. These tests assert the guard fires *before* any read,
and that normal under-cap masks still read/validate unchanged through both paths.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.mask as source_mask_mod
from hs2p.tiling.mask import load_annotation_label_mask, load_precomputed_tissue_mask


class _ExplodingMaskSlide:
    """A single-level mask whose only level exceeds the read cap; reading it must never
    happen, so ``read_region`` fails loudly if the guard lets execution reach it."""

    def __init__(self, level_dimensions):
        self.level_dimensions = level_dimensions
        self.level_downsamples = [(1.0, 1.0)]
        self.spacing = 0.25
        self.native_spacing = 0.25

    def read_region(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError(
            "read_region was called despite the oversized-mask guard — the guard must "
            "fail fast before any large allocation"
        )

    def close(self):
        pass


class _FakeMaskSlide:
    """A normal small mask that reads a discrete single-value raster at its only level."""

    def __init__(self, level_dimensions, *, spacing=0.25, value=0):
        self.level_dimensions = level_dimensions
        self.level_downsamples = [(1.0, 1.0)]
        self.spacing = spacing
        self.native_spacing = spacing
        self._value = int(value)

    def read_region(self, location, level, size):
        del location, level
        width, height = int(size[0]), int(size[1])
        return np.full((height, width), self._value, dtype=np.uint8)

    def close(self):
        pass


def _make_wsi_slide():
    return SimpleNamespace(
        level_downsamples=[1.0, 4.0],
        spacing=0.25,
        level_dimensions=[(100, 100), (25, 25)],
        dimensions=(100, 100),
    )


# Dimensions of the real BEETLE mask that triggered the SIGKILL (2740 Mpx, well over the cap).
_OVERSIZED_DIMS = (62407, 43898)


def _oversized_wsi_slide():
    """A slide the oversized mask registers with pixel-for-pixel, whose seg level 1 is the
    coarser read the mask has no level for."""
    return SimpleNamespace(
        level_downsamples=[1.0, 4.0],
        spacing=0.25,
        level_dimensions=[_OVERSIZED_DIMS, (15602, 10975)],
    )


def _assert_oversized_message(message: str, *, mask_path: str, backend: str) -> None:
    assert mask_path in message
    assert f"backend={backend}" in message
    assert "level 0" in message
    assert f"{_OVERSIZED_DIMS[0]}x{_OVERSIZED_DIMS[1]}" in message
    assert "256 Mpx read cap" in message
    assert "pyramid" in message


# --- guard fires before any read ------------------------------------------------------


def test_precomputed_tissue_path_routes_through_guard(monkeypatch):
    mask_path = "/data/masks/oversized-tissue.tif"
    mask_slide = _ExplodingMaskSlide(level_dimensions=[_OVERSIZED_DIMS])
    monkeypatch.setattr(source_mask_mod, "open_slide", lambda *a, **k: mask_slide)

    with pytest.raises(ValueError) as excinfo:
        load_precomputed_tissue_mask(
            mask_path=mask_path,
            slide=_oversized_wsi_slide(),
            seg_level=1,
            tissue_value=1,
            mask_backend="asap",
        )

    _assert_oversized_message(str(excinfo.value), mask_path=mask_path, backend="asap")


def test_annotation_path_routes_through_guard(monkeypatch):
    mask_path = "/data/masks/oversized-annotation.tif"
    mask_slide = _ExplodingMaskSlide(level_dimensions=[_OVERSIZED_DIMS])
    monkeypatch.setattr(source_mask_mod, "open_slide", lambda *a, **k: mask_slide)

    with pytest.raises(ValueError) as excinfo:
        load_annotation_label_mask(
            mask_path=mask_path,
            slide=_oversized_wsi_slide(),
            seg_level=1,
            valid_values={0, 1},
            mask_backend="openslide",
        )

    _assert_oversized_message(
        str(excinfo.value), mask_path=mask_path, backend="openslide"
    )


# --- under-cap masks still read/validate unchanged ------------------------------------


def test_precomputed_tissue_path_passthrough_under_cap(monkeypatch):
    # A 25 px mask over the 100 px, 0.25 um/px slide: 1.0 um/px.
    mask_slide = _FakeMaskSlide(level_dimensions=[(25, 25)], spacing=1.0, value=1)
    monkeypatch.setattr(source_mask_mod, "open_slide", lambda *a, **k: mask_slide)

    mask, mask_level, mask_spacing_um = load_precomputed_tissue_mask(
        mask_path="/data/masks/small.tif",
        slide=_make_wsi_slide(),
        seg_level=1,
        tissue_value=1,
        mask_backend="asap",
    )

    assert mask.shape == (25, 25)
    assert (mask_level, mask_spacing_um) == (0, 1.0)
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert int(mask.max()) == 255


def test_annotation_path_passthrough_under_cap(monkeypatch):
    mask_slide = _FakeMaskSlide(level_dimensions=[(25, 25)], spacing=1.0, value=1)
    monkeypatch.setattr(source_mask_mod, "open_slide", lambda *a, **k: mask_slide)

    mask, mask_level, mask_spacing_um = load_annotation_label_mask(
        mask_path="/data/masks/small.tif",
        slide=_make_wsi_slide(),
        seg_level=1,
        valid_values={0, 1},
        mask_backend="openslide",
    )

    np.testing.assert_array_equal(mask, np.ones((25, 25), dtype=np.uint8))
    assert (mask_level, mask_spacing_um) == (0, 1.0)
