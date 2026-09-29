import json

import cv2
import numpy as np
import pytest

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
from hs2p.tiling.coverage import compute_tile_coverage


BASE_SPACING = 0.25
DOWNSAMPLES = [1.0, 2.0, 4.0, 16.0]


@pytest.mark.parametrize("num_workers", [1, 2])
def test_generate_tiles_bounds_coverage_work_to_selected_content(monkeypatch, num_workers):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:6, 4:6] = 255
    mask[48:50, 48:50] = 255
    contours = detect_contours(mask, slide_dimensions=(64, 64), a_t=0)
    integral_sizes = []
    original_integral = cv2.integral

    def track_integral(image, *args, **kwargs):
        integral_sizes.append(image.size)
        return original_integral(image, *args, **kwargs)

    monkeypatch.setattr(cv2, "integral", track_integral)
    result = generate_tiles(
        (64, 64), contours, requested_tile_size_px=2,
        requested_spacing_um=1.0, base_spacing_um=1.0,
        level_downsamples=[1.0], min_tissue_fraction=0.5,
        num_workers=num_workers,
    )

    np.testing.assert_array_equal(result.x, [4, 48])
    np.testing.assert_array_equal(result.y, [4, 48])
    np.testing.assert_array_equal(result.tissue_fractions, [1.0, 1.0])
    assert sum(integral_sizes) <= 8


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

    fractions = compute_tile_coverage(
        candidates=candidates,
        binary_mask=tissue_mask,
        tile_size_lv0=80,
        slide_dimensions=(100, 100),
    )

    np.testing.assert_array_equal(fractions, np.array([0.0625], dtype=np.float32))


def test_generate_tiles_uses_actual_read_geometry_when_spacing_is_within_tolerance():
    contour = np.array(
        [[[0, 0]], [[0, 1999]], [[1999, 1999]], [[1999, 0]]],
        dtype=np.int32,
    )
    contours = ContourResult(
        contours=[contour],
        holes=[[]],
        mask=np.full((2000, 2000), 255, dtype=np.uint8),
    )

    result = generate_tiles(
        slide_dimensions=(2000, 2000),
        contours=contours,
        requested_tile_size_px=448,
        requested_spacing_um=0.5,
        base_spacing_um=0.486187607049942,
        level_downsamples=[1.0],
        overlap=0.0,
        min_tissue_fraction=0.1,
        tolerance=0.07,
    )

    assert result.is_within_tolerance is True
    assert result.read_tile_size_px == 448
    assert result.tile_size_lv0 == 448
    assert np.unique(result.x)[1] == 448
    assert np.unique(result.y)[1] == 448


def test_generate_tiles_rejects_finer_image_spacing_via_shared_policy():
    contours = ContourResult(
        contours=[],
        holes=[],
        mask=np.zeros((1, 1), dtype=np.uint8),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"requested spacing 0\.125.*finest available spacing 0\.25.*"
            r"image upsampling is forbidden"
        ),
    ):
        generate_tiles(
            slide_dimensions=(100, 100),
            contours=contours,
            requested_tile_size_px=256,
            requested_spacing_um=0.125,
            base_spacing_um=0.25,
            level_downsamples=[1.0],
            tolerance=0.05,
        )


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
        read_tile_size_px=256,
        read_spacing_um=0.5,
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
        requested_seg_downsample=64,
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
