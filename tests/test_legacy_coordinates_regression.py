from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hs2p.api import (
    FilterConfig,
    SegmentationConfig,
    TilingConfig,
    WholeSlide,
    save_tiling_result,
    tile_slide,
)


def _require_asap_backend(wsi_path: Path) -> str:
    wsd = pytest.importorskip("wholeslidedata")
    try:
        wsd.WholeSlideImage(wsi_path, backend="asap")
        return "asap"
    except Exception:
        pytest.skip("ASAP backend is unavailable; golden coordinate regression requires ASAP")


def _build_tiling_configs(*, backend: str, tissue_pct: float) -> tuple[TilingConfig, SegmentationConfig, FilterConfig]:
    tiling = TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=tissue_pct,
        drop_holes=False,
        use_padding=True,
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
        max_n_holes=8,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )
    return tiling, segmentation, filtering


def _run_and_save_tiles(*, wsi_path: Path, mask_path: Path, backend: str, tissue_pct: float, output_dir: Path):
    tiling, segmentation, filtering = _build_tiling_configs(
        backend=backend,
        tissue_pct=tissue_pct,
    )
    result = tile_slide(
        WholeSlide(sample_id="test-wsi", image_path=wsi_path, mask_path=mask_path),
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        num_workers=1,
    )
    artifacts = save_tiling_result(result, output_dir=output_dir)
    generated = np.load(artifacts.tiles_npz_path, allow_pickle=False)
    meta = json.loads(artifacts.tiles_meta_path.read_text())
    return result, artifacts, generated, meta


def test_generated_tiles_match_legacy_golden(real_fixture_paths, tmp_path: Path):
    wsi_path, mask_path = real_fixture_paths
    gt_path = wsi_path.parent.parent / "gt" / "test-wsi.npy"
    assert gt_path.is_file(), f"Missing golden coordinates: {gt_path}"

    backend = _require_asap_backend(wsi_path)
    tissue_pct = 0.1
    result, artifacts, generated, meta = _run_and_save_tiles(
        wsi_path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        tissue_pct=tissue_pct,
        output_dir=tmp_path,
    )

    legacy = np.load(gt_path, allow_pickle=False)

    np.testing.assert_array_equal(generated["x_lv0"], legacy["x"])
    np.testing.assert_array_equal(generated["y_lv0"], legacy["y"])
    np.testing.assert_array_equal(
        generated["tile_index"],
        np.arange(legacy.shape[0], dtype=np.int32),
    )

    assert set(meta) == {
        "sample_id",
        "image_path",
        "mask_path",
        "backend",
        "target_spacing_um",
        "target_tile_size_px",
        "read_level",
        "read_spacing_um",
        "read_tile_size_px",
        "tile_size_lv0",
        "overlap",
        "tissue_threshold",
        "num_tiles",
        "config_hash",
    }
    assert meta["sample_id"] == "test-wsi"
    assert meta["image_path"] == str(wsi_path)
    assert meta["mask_path"] == str(mask_path)
    assert meta["backend"] == backend
    assert meta["target_spacing_um"] == float(legacy["target_spacing"][0])
    assert meta["target_tile_size_px"] == int(legacy["target_tile_size"][0])
    assert meta["read_level"] == int(legacy["tile_level"][0])
    assert meta["read_spacing_um"] == pytest.approx(
        float(legacy["target_spacing"][0] * legacy["resize_factor"][0])
    )
    assert meta["read_tile_size_px"] == int(legacy["tile_size_resized"][0])
    assert meta["tile_size_lv0"] == int(legacy["tile_size_lv0"][0])
    assert meta["overlap"] == 0.0
    assert meta["tissue_threshold"] == tissue_pct
    assert meta["num_tiles"] == int(legacy.shape[0])
    assert meta["config_hash"] == result.config_hash
    assert artifacts.num_tiles == int(legacy.shape[0])


def test_repeated_tiling_run_writes_identical_artifacts(real_fixture_paths, tmp_path: Path):
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

    assert first_result.config_hash == second_result.config_hash
    assert first_tiles.files == second_tiles.files
    for key in first_tiles.files:
        np.testing.assert_array_equal(first_tiles[key], second_tiles[key])
    assert first_meta == second_meta
