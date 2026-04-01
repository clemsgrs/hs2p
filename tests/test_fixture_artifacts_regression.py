import json
from pathlib import Path

import numpy as np
import pytest

from hs2p.api import (
    FilterConfig,
    SegmentationConfig,
    SlideSpec,
    TilingConfig,
    load_tiling_result,
    save_tiling_result,
    tile_slide,
)
from hs2p.wsi.streaming.plans import resolve_read_step_px

pytestmark = pytest.mark.integration


def _require_asap_backend(wsi_path: Path) -> str:
    wsd = pytest.importorskip("wholeslidedata")
    try:
        wsd.WholeSlideImage(wsi_path, backend="asap")
        return "asap"
    except Exception:
        pytest.skip(
            "ASAP backend is unavailable; golden coordinate regression requires ASAP"
        )


def _build_tiling_configs(
    *, backend: str, tissue_pct: float
) -> tuple[TilingConfig, SegmentationConfig, FilterConfig]:
    tiling = TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=tissue_pct,
        backend=backend,
    )
    segmentation = SegmentationConfig(
        downsample=64,
        sthresh=8,
        sthresh_up=255,
        mthresh=7,
        close=4,
        use_otsu=False,
        use_hsv=True,
    )
    filtering = FilterConfig(
        ref_tile_size=224,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )
    return tiling, segmentation, filtering


def _run_and_save_tiles(
    *,
    wsi_path: Path,
    mask_path: Path,
    backend: str,
    tissue_pct: float,
    output_dir: Path,
):
    tiling, segmentation, filtering = _build_tiling_configs(
        backend=backend,
        tissue_pct=tissue_pct,
    )
    result = tile_slide(
        SlideSpec(sample_id="test-wsi", image_path=wsi_path, mask_path=mask_path),
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        num_workers=1,
    )
    artifacts = save_tiling_result(result, output_dir=output_dir)
    generated = np.load(artifacts.coordinates_npz_path, allow_pickle=False)
    meta = json.loads(artifacts.coordinates_meta_path.read_text())
    return result, artifacts, generated, meta


def _load_normalized_tiling_result(npz_path: Path, meta_path: Path):
    return load_tiling_result(npz_path, meta_path)


def test_generated_tiles_match_checked_in_ground_truth_outputs(
    real_fixture_paths, tmp_path: Path
):
    wsi_path, mask_path = real_fixture_paths
    gt_dir = wsi_path.parent.parent / "gt"
    gt_npz_path = gt_dir / "test-wsi.coordinates.npz"
    gt_meta_path = gt_dir / "test-wsi.coordinates.meta.json"
    assert gt_npz_path.is_file(), f"Missing golden coordinates: {gt_npz_path}"
    assert gt_meta_path.is_file(), f"Missing golden metadata: {gt_meta_path}"

    backend = _require_asap_backend(wsi_path)
    tissue_pct = 0.1
    result, artifacts, generated, meta = _run_and_save_tiles(
        wsi_path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        tissue_pct=tissue_pct,
        output_dir=tmp_path,
    )

    golden = _load_normalized_tiling_result(gt_npz_path, gt_meta_path)
    generated_loaded = _load_normalized_tiling_result(
        artifacts.coordinates_npz_path,
        artifacts.coordinates_meta_path,
    )

    assert generated_loaded.sample_id == golden.sample_id == "test-wsi"
    assert generated_loaded.image_path.name == golden.image_path.name == wsi_path.name
    assert generated_loaded.mask_path is not None
    assert golden.mask_path is not None
    assert (
        generated_loaded.mask_path.name
        == golden.mask_path.name
        == mask_path.name
    )
    assert generated_loaded.backend == golden.backend == backend
    assert generated_loaded.requested_spacing_um == pytest.approx(golden.requested_spacing_um)
    assert generated_loaded.requested_tile_size_px == golden.requested_tile_size_px
    assert generated_loaded.read_level == golden.read_level
    assert generated_loaded.effective_spacing_um == pytest.approx(golden.effective_spacing_um)
    assert resolve_read_step_px(generated_loaded) == generated_loaded.effective_tile_size_px
    assert generated_loaded.effective_tile_size_px == golden.effective_tile_size_px
    assert generated_loaded.step_px_lv0 == generated_loaded.tile_size_lv0
    assert generated_loaded.tile_size_lv0 == golden.tile_size_lv0
    assert generated_loaded.overlap == pytest.approx(golden.overlap)
    assert generated_loaded.min_tissue_fraction == pytest.approx(golden.min_tissue_fraction)
    assert len(generated_loaded.coordinates) == len(golden.coordinates) == 459
    assert artifacts.num_tiles == len(golden.coordinates)
    np.testing.assert_array_equal(generated_loaded.tile_index, golden.tile_index)
    np.testing.assert_array_equal(generated_loaded.coordinates, golden.coordinates)
    assert generated_loaded.tissue_fractions.shape == golden.tissue_fractions.shape
    assert np.all(generated_loaded.tissue_fractions >= 0.0)
    assert np.all(generated_loaded.tissue_fractions <= 1.0)
    assert np.all(generated_loaded.tissue_fractions >= generated_loaded.min_tissue_fraction)


def test_repeated_tiling_run_writes_identical_artifacts(
    real_fixture_paths, tmp_path: Path
):
    wsi_path, mask_path = real_fixture_paths
    backend = _require_asap_backend(wsi_path)
    tissue_pct = 0.1

    first_result, _, first_tiles, first_meta = _run_and_save_tiles(
        wsi_path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        tissue_pct=tissue_pct,
        output_dir=tmp_path / "run1",
    )
    second_result, _, second_tiles, second_meta = _run_and_save_tiles(
        wsi_path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        tissue_pct=tissue_pct,
        output_dir=tmp_path / "run2",
    )
    assert first_tiles.files == second_tiles.files
    for key in first_tiles.files:
        np.testing.assert_array_equal(first_tiles[key], second_tiles[key])
    assert first_meta == second_meta
