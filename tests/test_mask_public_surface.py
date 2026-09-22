"""hs2p 5.0 exposes one source-mask boundary (#195): ``WSI`` is image-only and the
superseded low-level mask helpers are gone, without compatibility wrappers."""

import importlib
import inspect
from pathlib import Path

import pytest

import hs2p.wsi.wsi as wsi_mod
from hs2p.wsi.backend import BackendSelection

# (module, name) pairs that hs2p 4.x exported and 5.0 removes.
REMOVED_EXPORTS = [
    ("hs2p.wsi.reader", "open_mask_reader"),
    ("hs2p.wsi.backend", "open_mask_reader"),
    ("hs2p.wsi.masks", "read_aligned_mask"),
    ("hs2p.wsi.masks", "mask_level_downsamples"),
    ("hs2p.wsi.masks", "read_label_at_spacing"),
    ("hs2p.wsi.masks", "read_label_region_at_spacing"),
    ("hs2p.wsi", "read_aligned_mask"),
    ("hs2p.wsi", "mask_level_downsamples"),
    ("hs2p.tiling.mask", "load_precomputed_tissue_mask"),
    ("hs2p.tiling.mask", "load_annotation_label_mask"),
    ("hs2p.preprocessing", "load_precomputed_tissue_mask"),
    ("hs2p.preprocessing", "load_annotation_label_mask"),
    ("hs2p.api", "load_precomputed_tissue_mask"),
    ("hs2p.api", "load_annotation_label_mask"),
    ("hs2p", "load_precomputed_tissue_mask"),
    ("hs2p", "load_annotation_label_mask"),
]

# Private mask-specific helpers whose behavior ``hs2p.mask`` now owns.
REMOVED_PRIVATE_HELPERS = [
    ("hs2p.wsi.masks", "_collapse_label_raster"),
    ("hs2p.tiling.mask", "_resolve_mask_backend"),
    ("hs2p.tiling.mask", "_select_mask_level"),
    ("hs2p.tiling.mask", "_read_mask_level"),
    ("hs2p.tiling.mask", "_read_discrete_mask_level"),
    ("hs2p.tiling.mask", "_read_label_mask_at_seg"),
    ("hs2p.tiling.mask", "_reduce_mask_channels"),
    ("hs2p.tiling.mask", "_as_discrete_label_array"),
    ("hs2p.tiling.mask", "_mask_label_values"),
    ("hs2p.tiling.mask", "_is_discrete_binary_mask"),
    ("hs2p.tiling.mask", "_is_label_subset"),
    ("hs2p.tiling.mask", "_raise_mask_decode_error"),
    ("hs2p.tiling.mask", "MAX_MASK_READ_PX"),
]


class _FakeSlideReader:
    spacings = [0.5]
    level_dimensions = [(10, 10)]
    level_downsamples = [(1.0, 1.0)]

    def close(self):
        return None


def _open_image_only_wsi(monkeypatch) -> wsi_mod.WSI:
    monkeypatch.setattr(
        wsi_mod,
        "resolve_backend",
        lambda requested, **kwargs: BackendSelection(backend="asap", tried=("asap",)),
    )
    monkeypatch.setattr(wsi_mod, "open_slide", lambda *args, **kwargs: _FakeSlideReader())
    return wsi_mod.WSI(path=Path("/data/slide.svs"), backend="auto")


@pytest.mark.parametrize("argument", ["mask_path", "mask_backend"])
def test_wsi_rejects_attached_mask_arguments(argument):
    with pytest.raises(TypeError, match=argument):
        wsi_mod.WSI(
            path=Path("/data/slide.svs"), backend="openslide", **{argument: "mask.tif"}
        )


def test_wsi_carries_no_mask_reader_or_provenance(monkeypatch):
    wsi = _open_image_only_wsi(monkeypatch)

    assert [name for name in vars(wsi) if "mask" in name] == []


@pytest.mark.parametrize(("module", "name"), REMOVED_EXPORTS)
def test_superseded_mask_helpers_are_not_exported(module, name):
    module_obj = importlib.import_module(module)

    assert not hasattr(module_obj, name)
    assert name not in getattr(module_obj, "__all__", ())


@pytest.mark.parametrize(("module", "name"), REMOVED_PRIVATE_HELPERS)
def test_private_mask_helpers_are_gone(module, name):
    assert not hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("hs2p.preprocessing", "resolve_tissue_mask"),
        ("hs2p.preprocessing", "resolve_annotation_masks"),
        ("hs2p.api", "resolve_annotation_masks"),
    ],
)
def test_resolvers_remain_and_consume_a_mask(module, name):
    resolver = getattr(importlib.import_module(module), name)

    parameter = inspect.signature(resolver).parameters["mask"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
