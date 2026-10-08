"""End-to-end provenance, persistence, resume, and progress coverage for the independent
mask backend (#163)."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import hs2p.tiling.orchestration as orchestration_mod
from hs2p.api import (
    CompatibilitySpec,
    FilterConfig,
    SegmentationConfig,
    SlideSpec,
    TilingArtifacts,
    TilingConfig,
    load_tiling_result,
    save_tiling_result,
    validate_tiling_artifacts,
)
from hs2p.tiling.result import TileGeometry, TilingResult
from hs2p.wsi.types import CoordinateSelectionStrategy


def _tiles() -> TileGeometry:
    return TileGeometry(
        x=np.array([10, 30], dtype=np.int64),
        y=np.array([20, 40], dtype=np.int64),
        tissue_fractions=np.array([0.3, 0.7], dtype=np.float32),
        tile_index=np.array([0, 1], dtype=np.int32),
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
    )


def _result(**overrides) -> TilingResult:
    params = dict(
        tiles=_tiles(),
        sample_id="slide-1",
        image_path="slide-1.svs",
        backend="cucim",
        requested_backend="auto",
        mask_backend="openslide",
        requested_mask_backend="auto",
        tolerance=0.07,
        step_px_lv0=224,
        tissue_method="precomputed_mask",
        requested_seg_downsample=64,
        seg_downsample=64,
        seg_level=0,
        seg_spacing_um=0.5,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=224,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
        mask_path="slide-1-mask.tif",
    )
    params.update(overrides)
    return TilingResult(**params)


def test_metadata_round_trips_all_four_backends_without_conflation(tmp_path: Path):
    result = _result()
    artifacts = save_tiling_result(result, output_dir=tmp_path)
    loaded = load_tiling_result(
        artifacts.coordinates_npz_path, artifacts.coordinates_meta_path
    )
    # requested and resolved, slide and mask, all distinct and preserved separately.
    assert loaded.requested_backend == "auto"
    assert loaded.backend == "cucim"
    assert loaded.requested_mask_backend == "auto"
    assert loaded.mask_backend == "openslide"
    # artifacts carry the mask provenance too
    assert artifacts.mask_backend == "openslide"
    assert artifacts.requested_mask_backend == "auto"


def test_success_process_row_records_mask_backends_symmetrically():
    artifact = TilingArtifacts(
        sample_id="slide-1",
        coordinates_npz_path=Path("tiles/slide-1.coordinates.npz"),
        coordinates_meta_path=Path("tiles/slide-1.coordinates.meta.json"),
        num_tiles=2,
        backend="cucim",
        requested_backend="auto",
        mask_backend="openslide",
        requested_mask_backend="auto",
    )
    row = orchestration_mod._build_success_process_row(
        whole_slide=SlideSpec(
            sample_id="slide-1",
            image_path=Path("slide-1.svs"),
            mask_path=Path("slide-1-mask.tif"),
        ),
        artifact=artifact,
        selection_strategy=CoordinateSelectionStrategy.MERGED_DEFAULT_TILING,
    )
    assert row["requested_backend"] == "auto"
    assert row["backend"] == "cucim"
    assert row["requested_mask_backend"] == "auto"
    assert row["mask_backend"] == "openslide"


def test_resume_rejects_on_resolved_mask_backend_mismatch(tmp_path: Path):
    result = _result()
    artifacts = save_tiling_result(result, output_dir=tmp_path)
    whole_slide = SlideSpec(
        sample_id="slide-1",
        image_path=Path("slide-1.svs"),
        mask_path=Path("slide-1-mask.tif"),
    )
    seg = SegmentationConfig(method="precomputed_mask", downsample=64, sthresh=8, sthresh_up=255, mthresh=7, close=4)
    filt = FilterConfig(ref_tile_size=224, a_t=4, a_h=2, filter_white=False, filter_black=False, white_threshold=220, black_threshold=25, fraction_threshold=0.9)
    tiling = TilingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=224, tolerance=0.07, overlap=0.0,
        min_coverage={"tissue": 0.1}, backend="cucim", mask_backend="auto",
    )
    # Compatible: resolved mask backend matches (openslide), requested differs (auto vs asap).
    compatible = CompatibilitySpec(
        tiling=tiling, segmentation=seg, filtering=filt, mask_backend="openslide",
    )
    ok = validate_tiling_artifacts(
        whole_slide=whole_slide,
        coordinates_npz_path=artifacts.coordinates_npz_path,
        coordinates_meta_path=artifacts.coordinates_meta_path,
        compatibility=compatible,
    )
    assert ok.mask_backend == "openslide"
    # Incompatible: resolved mask backend differs.
    incompatible = replace(compatible, mask_backend="asap")
    with pytest.raises(ValueError, match="mask_backend mismatch"):
        validate_tiling_artifacts(
            whole_slide=whole_slide,
            coordinates_npz_path=artifacts.coordinates_npz_path,
            coordinates_meta_path=artifacts.coordinates_meta_path,
            compatibility=incompatible,
        )
