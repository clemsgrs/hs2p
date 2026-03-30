import json

import numpy as np
import pytest

import hs2p
from hs2p.preprocessing import (
    ContourResult,
    TileGeometry,
    TilingResult,
    detect_contours,
    generate_tiles,
    load_tiling_result,
    save_tiling_result,
)


BASE_SPACING = 0.25
DOWNSAMPLES = [1.0, 2.0, 4.0, 16.0]


def test_detect_contours_keeps_all_child_holes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    mask[20:30, 20:30] = 0
    mask[40:55, 40:55] = 0
    mask[60:80, 60:80] = 0

    contours = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        ref_tile_size_px=16,
        requested_spacing_um=0.5,
        a_t=0,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )

    assert len(contours.contours) == 1
    assert len(contours.holes) == 1
    assert len(contours.holes[0]) == 3


def test_generate_tiles_normalizes_padded_edge_tiles_over_valid_area_only():
    mask = np.full((100, 100), 255, dtype=np.uint8)
    contours = ContourResult(
        contours=detect_contours(
            mask,
            slide_dimensions=(100, 100),
            ref_tile_size_px=16,
            requested_spacing_um=0.5,
            a_t=0,
            base_spacing_um=0.5,
            level_downsamples=[1.0],
        ).contours,
        holes=[[]],
        mask=mask,
    )

    padded = generate_tiles(
        slide_dimensions=(100, 100),
        contours=contours,
        requested_tile_size_px=80,
        requested_spacing_um=0.5,
        base_spacing_um=0.5,
        level_downsamples=[1.0],
        overlap=0.0,
        use_padding=True,
        min_tissue_fraction=1.0,
    )
    unpadded = generate_tiles(
        slide_dimensions=(100, 100),
        contours=contours,
        requested_tile_size_px=80,
        requested_spacing_um=0.5,
        base_spacing_um=0.5,
        level_downsamples=[1.0],
        overlap=0.0,
        use_padding=False,
        min_tissue_fraction=1.0,
    )

    assert len(padded.coordinates) > len(unpadded.coordinates)
    assert any((coord == np.array([80, 80])).all() for coord in padded.coordinates)


def _make_tiling_result(n_tiles: int = 4) -> TilingResult:
    rng = np.random.RandomState(42)
    coords = rng.randint(0, 1000, size=(n_tiles, 2)).astype(np.int64)
    fracs = rng.uniform(0.5, 1.0, size=n_tiles).astype(np.float32)
    tiles = TileGeometry(
        coordinates=coords,
        tissue_fractions=fracs,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=1,
        effective_tile_size_px=256,
        effective_spacing_um=0.5,
        tile_size_lv0=512,
        is_within_tolerance=True,
        use_padding=True,
        base_spacing_um=0.25,
        slide_dimensions=[1000, 800],
        level_downsamples=[1.0, 2.0, 4.0],
        overlap=0.25,
        min_tissue_fraction=0.5,
    )
    return TilingResult(
        tiles=tiles,
        sample_id="slide-001",
        image_path="/tmp/slide-001.svs",
        backend="openslide",
        requested_backend="auto",
        tolerance=0.05,
        step_px_lv0=384,
        tissue_method="precomputed_mask",
        seg_downsample=64,
        seg_level=2,
        seg_spacing_um=1.0,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=256,
        a_t=4,
        a_h=0,
        max_n_holes=0,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
        tissue_mask_path="/tmp/slide-001-mask.tif",
        tissue_mask_tissue_value=1,
        mask_level=1,
        mask_spacing_um=0.5,
    )


def test_tiling_artifact_roundtrip_uses_strict_rich_metadata(tmp_path):
    result = _make_tiling_result()
    paths = save_tiling_result(result, tmp_path, "slide-001")

    meta = json.loads(paths["meta"].read_text())
    assert meta["format_version"] == 2
    assert meta["provenance"]["requested_backend"] == "auto"
    assert meta["slide"]["base_spacing_um"] == 0.25
    assert meta["segmentation"]["seg_level"] == 2
    assert meta["segmentation"]["seg_spacing_um"] == 1.0
    assert meta["segmentation"]["tissue_mask_path"] == "/tmp/slide-001-mask.tif"
    assert meta["segmentation"]["mask_level"] == 1
    assert meta["segmentation"]["mask_spacing_um"] == 0.5
    assert "requested_backend" not in meta
    assert "seg_level" not in meta
    assert "seg_spacing_um" not in meta

    loaded = load_tiling_result(paths["npz"], paths["meta"])
    np.testing.assert_array_equal(
        loaded.tile_index,
        np.arange(len(result.coordinates), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        loaded.coordinates,
        result.coordinates[np.lexsort((result.coordinates[:, 1], result.coordinates[:, 0]))],
    )
    assert loaded.requested_backend == "auto"
    assert loaded.base_spacing_um == pytest.approx(0.25)
    assert loaded.seg_level == 2
    assert loaded.mask_level == 1

    meta["unexpected_key"] = True
    paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="unexpected keys"):
        load_tiling_result(paths["npz"], paths["meta"])


def test_top_level_package_reexports_preprocessing_core_surface():
    assert hs2p.ContourResult is ContourResult
    assert hs2p.TileGeometry is TileGeometry
    assert hs2p.detect_contours is detect_contours
    assert hs2p.generate_tiles is generate_tiles
    assert hs2p.preprocess_slide is hs2p.preprocessing.preprocess_slide
