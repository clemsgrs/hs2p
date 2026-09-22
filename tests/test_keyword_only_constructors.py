"""Public constructors are keyword-only (ADR 0003): positional construction raises
``TypeError`` and keyword construction binds every field by name."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from hs2p.api import (
    CompatibilitySpec,
    FilterConfig,
    PreviewConfig,
    SegmentationConfig,
    SlideSpec,
    TilingArtifacts,
    TilingConfig,
    load_tiling_result,
    save_tiling_result,
)
from hs2p.tiling.result import (
    ContourResult,
    ResolvedAnnotationMasks,
    ResolvedTissueMask,
    Sam2Thumbnail,
    TileGeometry,
    TilingResult,
)
from hs2p.wsi.geometry import LevelSelection, SpacingReadPlan
from hs2p.wsi.types import SamplingSpec
from hs2p.wsi.wsi import WSI

_TILE_GEOMETRY = dict(
    x=np.array([10, 30], dtype=np.int64),
    y=np.array([20, 40], dtype=np.int64),
    tissue_fractions=np.array([0.3, 0.7], dtype=np.float32),
    requested_tile_size_px=224,
    requested_spacing_um=0.5,
    read_level=0,
    read_tile_size_px=224,
    read_spacing_um=0.5,
    tile_size_lv0=224,
    is_within_tolerance=True,
    base_spacing_um=0.25,
    slide_dimensions=[1000, 1200],
    level_downsamples=[1.0],
    overlap=0.0,
    min_tissue_fraction=0.1,
    tile_index=np.array([0, 1], dtype=np.int32),
)

_TILING_RESULT = dict(
    tiles=TileGeometry(**_TILE_GEOMETRY),
    sample_id="slide-1",
    image_path=Path("slide-1.svs"),
    backend="asap",
    requested_backend="auto",
    tolerance=0.07,
    step_px_lv0=224,
    tissue_method="hsv",
    requested_seg_downsample=64,
    seg_downsample=64,
    seg_level=0,
    seg_spacing_um=0.5,
    seg_sthresh=8,
    seg_sthresh_up=255,
    seg_mthresh=7,
    seg_close=4,
    ref_tile_size_px=224,
    a_t=4.0,
    a_h=2.0,
    filter_white=False,
    filter_black=False,
    white_threshold=220,
    black_threshold=25,
    fraction_threshold=0.9,
)

_TILING_CONFIG = TilingConfig(
    requested_spacing_um=0.5,
    requested_tile_size_px=224,
    tolerance=0.07,
    overlap=0.0,
    min_coverage={"tissue": 0.1},
    backend="asap",
)

CONSTRUCTOR_CASES = [
    pytest.param(SegmentationConfig, dict(method="hsv"), id="SegmentationConfig"),
    pytest.param(FilterConfig, dict(ref_tile_size=224), id="FilterConfig"),
    pytest.param(PreviewConfig, dict(save_mask_preview=True), id="PreviewConfig"),
    pytest.param(
        SlideSpec, dict(sample_id="slide-1", image_path=Path("slide-1.svs")), id="SlideSpec"
    ),
    pytest.param(
        TilingArtifacts,
        dict(
            sample_id="slide-1",
            coordinates_npz_path=None,
            coordinates_meta_path=Path("slide-1.coordinates.meta.json"),
            num_tiles=0,
        ),
        id="TilingArtifacts",
    ),
    pytest.param(
        CompatibilitySpec,
        dict(
            tiling=_TILING_CONFIG,
            segmentation=SegmentationConfig(method="hsv"),
            filtering=FilterConfig(),
        ),
        id="CompatibilitySpec",
    ),
    pytest.param(
        SamplingSpec,
        dict(
            pixel_mapping={"tumor": 1},
            color_mapping=None,
            tissue_percentage={"tumor": None},
            active_annotations=("tumor",),
        ),
        id="SamplingSpec",
    ),
    pytest.param(TileGeometry, _TILE_GEOMETRY, id="TileGeometry"),
    pytest.param(TilingResult, _TILING_RESULT, id="TilingResult"),
    pytest.param(
        ContourResult,
        dict(contours=[], holes=[], mask=np.zeros((2, 2), dtype=np.uint8)),
        id="ContourResult",
    ),
    pytest.param(
        ResolvedTissueMask,
        dict(
            tissue_mask=np.zeros((2, 2), dtype=np.uint8),
            tissue_method="hsv",
            requested_seg_downsample=64,
            seg_downsample=64,
            seg_level=0,
            seg_spacing_um=0.5,
        ),
        id="ResolvedTissueMask",
    ),
    pytest.param(
        ResolvedAnnotationMasks,
        dict(
            masks={"tumor": np.zeros((2, 2), dtype=np.uint8)},
            tissue_method="annotation_mask",
            requested_seg_downsample=64,
            seg_downsample=64,
            seg_level=0,
            seg_spacing_um=0.5,
            pixel_mapping={"tumor": 1},
        ),
        id="ResolvedAnnotationMasks",
    ),
    pytest.param(
        Sam2Thumbnail,
        dict(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            seg_level=0,
            seg_spacing_um=0.5,
            source_spacing_um=0.25,
            resized=True,
        ),
        id="Sam2Thumbnail",
    ),
    pytest.param(
        LevelSelection,
        dict(level=0, read_spacing_um=0.5, is_within_tolerance=True),
        id="LevelSelection",
    ),
    pytest.param(
        SpacingReadPlan,
        dict(level=0, read_spacing_um=0.5, is_within_tolerance=True, read_size_px=(8, 8)),
        id="SpacingReadPlan",
    ),
]


@pytest.mark.parametrize("cls, kwargs", CONSTRUCTOR_CASES)
def test_positional_construction_raises_type_error(cls, kwargs):
    with pytest.raises(TypeError):
        cls(*kwargs.values())


@pytest.mark.parametrize("cls, kwargs", CONSTRUCTOR_CASES)
def test_keyword_construction_binds_every_field(cls, kwargs):
    built = cls(**kwargs)

    for name, value in kwargs.items():
        assert getattr(built, name) is value or getattr(built, name) == value


def test_wsi_positional_construction_raises_type_error(fake_backend):
    fake_backend(np.zeros((16, 16, 1), dtype=np.uint8))

    with pytest.raises(TypeError):
        WSI(Path("synthetic-slide.tif"), "asap")


def test_wsi_keyword_construction_works(fake_backend):
    fake_backend(np.zeros((16, 16, 1), dtype=np.uint8))

    wsi = WSI(path=Path("synthetic-slide.tif"), backend="asap")

    assert wsi.path == Path("synthetic-slide.tif")
    assert wsi.requested_backend == "asap"


def _comparable(result: TilingResult) -> dict:
    data = asdict(result)
    data["tiles"] = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in data["tiles"].items()
    }
    return data


def test_tiling_result_round_trips_through_artifacts_by_field_name(tmp_path: Path):
    result = TilingResult(**_TILING_RESULT)

    artifacts = save_tiling_result(result, output_dir=tmp_path)
    loaded = load_tiling_result(artifacts.coordinates_npz_path, artifacts.coordinates_meta_path)

    assert _comparable(loaded) == _comparable(result)
