import json
from pathlib import Path

import numpy as np
import pytest

import hs2p
import hs2p.preprocessing as preprocessing_mod
from hs2p.preprocessing import (
    ContourResult,
    TileGeometry,
    TilingResult,
    _load_tiling_result_from_paths as load_tiling_result,
    _save_tiling_result as save_tiling_result,
    detect_contours,
    generate_tiles,
)
from hs2p.wsi.streaming.plans import resolve_read_step_px


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


def test_compute_tissue_fractions_normalizes_padded_tiles_over_full_tile_area():
    tissue_mask = np.ones((100, 100), dtype=np.uint8)
    candidates = np.array([[80, 80]], dtype=np.int64)

    fractions = preprocessing_mod._compute_tissue_fractions(
        candidates=candidates,
        tissue_mask=tissue_mask,
        tile_size_lv0=80,
        slide_dimensions=(100, 100),
    )

    np.testing.assert_array_equal(fractions, np.array([0.0625], dtype=np.float32))


def test_compute_tissue_fractions_truncates_projected_tile_origins():
    tissue_mask = np.zeros((3, 3), dtype=np.uint8)
    tissue_mask[1, 1] = 1
    candidates = np.array([[15, 15]], dtype=np.int64)

    fractions = preprocessing_mod._compute_tissue_fractions(
        candidates=candidates,
        tissue_mask=tissue_mask,
        tile_size_lv0=10,
        slide_dimensions=(30, 30),
    )

    np.testing.assert_array_equal(fractions, np.array([1.0], dtype=np.float32))


def _make_tiling_result(n_tiles: int = 4) -> TilingResult:
    rng = np.random.RandomState(42)
    coords = rng.randint(0, 1000, size=(n_tiles, 2)).astype(np.int64)
    x = coords[:, 0]
    y = coords[:, 1]
    fracs = rng.uniform(0.5, 1.0, size=n_tiles).astype(np.float32)
    tiles = TileGeometry(
        x=x,
        y=y,
        tissue_fractions=fracs,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=1,
        effective_tile_size_px=256,
        effective_spacing_um=0.5,
        tile_size_lv0=512,
        is_within_tolerance=True,
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
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
        mask_path="/tmp/slide-001-mask.tif",
        tissue_mask_tissue_value=1,
        mask_level=1,
        mask_spacing_um=0.5,
    )


def test_tiling_artifact_roundtrip_uses_strict_rich_metadata(tmp_path):
    result = _make_tiling_result()
    paths = save_tiling_result(result, tmp_path, "slide-001")

    meta = json.loads(paths["meta"].read_text())
    assert meta["provenance"]["requested_backend"] == "auto"
    assert meta["slide"]["base_spacing_um"] == 0.25
    assert meta["segmentation"]["seg_level"] == 2
    assert meta["segmentation"]["seg_spacing_um"] == 1.0
    assert meta["segmentation"]["mask_path"] == "/tmp/slide-001-mask.tif"
    assert meta["segmentation"]["mask_level"] == 1
    assert meta["segmentation"]["mask_spacing_um"] == 0.5
    assert set(meta["provenance"].keys()) == preprocessing_mod._PROVENANCE_KEYS
    assert set(meta["segmentation"].keys()) == preprocessing_mod._SEGMENTATION_KEYS

    loaded = load_tiling_result(paths["npz"], paths["meta"])
    np.testing.assert_array_equal(
        loaded.tile_index,
        np.arange(len(result.x), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.column_stack((loaded.x, loaded.y)),
        np.column_stack((result.x, result.y))[
            np.lexsort((result.y, result.x))
        ],
    )
    assert loaded.requested_backend == "auto"
    assert loaded.base_spacing_um == pytest.approx(0.25)
    assert loaded.seg_level == 2
    assert loaded.mask_level == 1
    assert meta["filtering"]["filter_grayspace"] is False
    assert meta["filtering"]["grayspace_saturation_threshold"] == pytest.approx(0.05)
    assert meta["filtering"]["grayspace_fraction_threshold"] == pytest.approx(0.6)
    assert meta["filtering"]["filter_blur"] is False
    assert meta["filtering"]["blur_threshold"] == pytest.approx(50.0)
    assert meta["filtering"]["qc_spacing_um"] == pytest.approx(2.0)

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


def test_preprocessing_result_uses_canonical_geometry_fields():
    result = _make_tiling_result()

    np.testing.assert_array_equal(result.x, result.tiles.x)
    np.testing.assert_array_equal(result.y, result.tiles.y)
    np.testing.assert_array_equal(result.tissue_fractions, result.tiles.tissue_fractions)
    assert len(result.x) == len(result.tiles.x)
    assert result.requested_spacing_um == pytest.approx(result.tiles.requested_spacing_um)
    assert result.requested_tile_size_px == result.tiles.requested_tile_size_px
    assert result.effective_spacing_um == pytest.approx(result.tiles.effective_spacing_um)
    assert result.effective_tile_size_px == result.tiles.effective_tile_size_px
    assert resolve_read_step_px(result) == 192
    assert result.min_tissue_fraction == pytest.approx(result.tiles.min_tissue_fraction)
    assert result.mask_path == Path("/tmp/slide-001-mask.tif")
