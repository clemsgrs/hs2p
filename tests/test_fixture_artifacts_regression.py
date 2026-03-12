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


pytestmark = pytest.mark.integration


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


def test_generated_tiles_match_checked_in_artifacts(real_fixture_paths, tmp_path: Path):
    wsi_path, mask_path = real_fixture_paths
    gt_dir = wsi_path.parent.parent / "gt"
    gt_npz_path = gt_dir / "test-wsi.tiles.npz"
    gt_meta_path = gt_dir / "test-wsi.tiles.meta.json"
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

    golden_tiles = np.load(gt_npz_path, allow_pickle=False)
    golden_meta = json.loads(gt_meta_path.read_text())

    assert generated.files == golden_tiles.files
    for key in generated.files:
        np.testing.assert_array_equal(generated[key], golden_tiles[key])

    assert set(meta) == set(golden_meta)
    assert meta["sample_id"] == golden_meta["sample_id"] == "test-wsi"
    assert Path(meta["image_path"]).name == Path(golden_meta["image_path"]).name == wsi_path.name
    assert Path(meta["mask_path"]).name == Path(golden_meta["mask_path"]).name == mask_path.name
    assert meta["backend"] == golden_meta["backend"] == backend
    assert meta["target_spacing_um"] == pytest.approx(golden_meta["target_spacing_um"])
    assert meta["target_tile_size_px"] == golden_meta["target_tile_size_px"]
    assert meta["read_level"] == golden_meta["read_level"]
    assert meta["read_spacing_um"] == pytest.approx(golden_meta["read_spacing_um"])
    assert meta["read_tile_size_px"] == golden_meta["read_tile_size_px"]
    assert meta["tile_size_lv0"] == golden_meta["tile_size_lv0"]
    assert meta["overlap"] == pytest.approx(golden_meta["overlap"])
    assert meta["tissue_threshold"] == pytest.approx(golden_meta["tissue_threshold"])
    assert meta["num_tiles"] == golden_meta["num_tiles"]
    assert meta["config_hash"] == golden_meta["config_hash"]
    assert meta["config_hash"] == result.config_hash
    assert artifacts.num_tiles == golden_meta["num_tiles"]


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
