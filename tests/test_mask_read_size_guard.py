"""Fail-fast guard for oversized (non-pyramidal) mask reads.

A mask whose nearest pyramid level to the requested segmentation spacing is still huge
(because the mask lacks a coarse pyramid level) would force ``read_region`` to materialise a
multi-GB raster, immediately followed by an 8x ``np.unique(int64)`` copy, OOM-killing the job.
``_read_discrete_mask_level`` is the single chokepoint both read paths funnel through, so a
size guard there protects the precomputed-tissue path (``_read_mask_level``) and the annotation
path (``_read_label_mask_at_seg``) at once. These tests assert the guard fires *before* any
read, and that normal under-cap masks still read/validate unchanged through both paths.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.tiling.mask as tiling_mask_mod
from hs2p.tiling.mask import (
    MAX_MASK_READ_PX,
    _read_discrete_mask_level,
    _read_mask_level,
    load_annotation_label_mask,
    load_precomputed_tissue_mask,
)


class _ExplodingMaskSlide:
    """A single-level mask whose only level exceeds the read cap; reading it must never
    happen, so ``read_region`` fails loudly if the guard lets execution reach it."""

    def __init__(self, level_dimensions):
        self.level_dimensions = level_dimensions
        self.level_downsamples = [1.0]
        self.spacing = 0.25

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
        self.level_downsamples = [1.0]
        self.spacing = spacing
        self._value = int(value)

    def read_region(self, location, level, size):
        del location, level
        width, height = int(size[0]), int(size[1])
        return np.full((height, width), self._value, dtype=np.uint8)

    def close(self):
        pass


def _make_wsi_slide(*, backend_name="asap"):
    return SimpleNamespace(
        level_downsamples=[1.0, 4.0],
        spacing=0.25,
        level_dimensions=[(100, 100), (25, 25)],
        dimensions=(100, 100),
        backend_name=backend_name,
    )


# Dimensions of the real BEETLE mask that triggered the SIGKILL (2740 Mpx, well over the cap).
_OVERSIZED_DIMS = (62407, 43898)


def _assert_oversized_message(message: str, *, label: str, mask_path: str) -> None:
    assert label in message
    assert mask_path in message
    assert "level 0" in message
    assert f"{_OVERSIZED_DIMS[0]}x{_OVERSIZED_DIMS[1]}" in message
    assert "Mpx" in message
    assert "MAX_MASK_READ_PX" in message
    assert "pyramid" in message
    assert "seg_downsample" in message


# --- guard fires before any read ------------------------------------------------------


@pytest.mark.parametrize("label", ["Annotation mask", "Precomputed tissue mask"])
def test_read_discrete_mask_level_guard_fires_before_read(label):
    mask_path = "/data/masks/oversized.tif"
    mask_slide = _ExplodingMaskSlide(level_dimensions=[_OVERSIZED_DIMS])

    with pytest.raises(ValueError) as excinfo:
        _read_discrete_mask_level(
            mask_path=mask_path,
            mask_slide=mask_slide,
            mask_level=0,
            is_discrete=lambda mask: True,
            label=label,
        )

    _assert_oversized_message(str(excinfo.value), label=label, mask_path=mask_path)


def test_precomputed_tissue_path_routes_through_guard(monkeypatch):
    mask_path = "/data/masks/oversized-tissue.tif"
    mask_slide = _ExplodingMaskSlide(level_dimensions=[_OVERSIZED_DIMS])
    monkeypatch.setattr(
        tiling_mask_mod, "open_slide", lambda *a, **k: mask_slide
    )
    slide = _make_wsi_slide()

    with pytest.raises(ValueError) as excinfo:
        load_precomputed_tissue_mask(
            mask_path=mask_path,
            slide=slide,
            seg_level=1,
            tissue_value=1,
        )

    _assert_oversized_message(
        str(excinfo.value), label="Precomputed tissue mask", mask_path=mask_path
    )


def test_annotation_path_routes_through_guard(monkeypatch):
    mask_path = "/data/masks/oversized-annotation.tif"
    mask_slide = _ExplodingMaskSlide(level_dimensions=[_OVERSIZED_DIMS])
    monkeypatch.setattr(
        tiling_mask_mod, "open_slide", lambda *a, **k: mask_slide
    )
    # backend_name="openslide" disables the degenerate-read openslide retry in
    # load_annotation_label_mask so the guard's ValueError surfaces directly.
    slide = _make_wsi_slide(backend_name="openslide")

    with pytest.raises(ValueError) as excinfo:
        load_annotation_label_mask(
            mask_path=mask_path,
            slide=slide,
            seg_level=1,
            valid_values={0, 1},
        )

    _assert_oversized_message(
        str(excinfo.value), label="Annotation mask", mask_path=mask_path
    )


# --- under-cap masks still read/validate unchanged ------------------------------------


def test_read_discrete_mask_level_passthrough_under_cap():
    mask_slide = _FakeMaskSlide(level_dimensions=[(25, 25)], value=0)
    result = _read_discrete_mask_level(
        mask_path="/data/masks/small.tif",
        mask_slide=mask_slide,
        mask_level=0,
        is_discrete=lambda mask: True,
        label="Precomputed tissue mask",
    )
    assert result.shape == (25, 25)
    assert result.dtype == np.uint8


def test_read_mask_level_passthrough_under_cap():
    mask_slide = _FakeMaskSlide(level_dimensions=[(25, 25)], value=1)
    result = _read_mask_level(
        mask_path="/data/masks/small.tif",
        mask_slide=mask_slide,
        mask_level=0,
        tissue_value=1,
    )
    assert result.shape == (25, 25)
    assert int(result.max()) == 1


def test_precomputed_tissue_path_passthrough_under_cap(monkeypatch):
    mask_slide = _FakeMaskSlide(level_dimensions=[(25, 25)], value=1)
    monkeypatch.setattr(
        tiling_mask_mod, "open_slide", lambda *a, **k: mask_slide
    )
    slide = _make_wsi_slide()

    mask, mask_level, mask_spacing_um = load_precomputed_tissue_mask(
        mask_path="/data/masks/small.tif",
        slide=slide,
        seg_level=1,
        tissue_value=1,
    )

    assert mask.shape == (25, 25)
    assert mask_level == 0
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert int(mask.max()) == 255


def test_annotation_path_passthrough_under_cap(monkeypatch):
    mask_slide = _FakeMaskSlide(level_dimensions=[(25, 25)], value=0)
    monkeypatch.setattr(
        tiling_mask_mod, "open_slide", lambda *a, **k: mask_slide
    )
    slide = _make_wsi_slide(backend_name="openslide")

    mask, mask_level, mask_spacing_um = load_annotation_label_mask(
        mask_path="/data/masks/small.tif",
        slide=slide,
        seg_level=1,
        valid_values={0, 1},
    )

    assert mask.shape == (25, 25)
    assert mask_level == 0
    assert mask.dtype == np.uint8


def test_max_mask_read_px_threshold_value():
    # Locks the agreed cap (256 Mpx) — sits in the empirical gap between healthy BEETLE
    # mask reads (<= 4.9 Mpx) and the fatal 2740 Mpx mask.
    assert MAX_MASK_READ_PX == 256_000_000
