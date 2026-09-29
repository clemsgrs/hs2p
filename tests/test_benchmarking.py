import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.preprocessing as preprocessing_mod

pytestmark = pytest.mark.script


def test_read_benchmark_resume_preserves_completed_csv_results(monkeypatch, tmp_path):
    module = _load_benchmark_script_module()
    summary = tmp_path / "benchmark_summary.csv"
    runs = tmp_path / "benchmark_runs.csv"
    summary.write_text("mode,tiles\nregular_wsd,2\n")
    runs.write_text("mode,repeat_index,tiles\nregular_wsd,0,2\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_tile_read.py", "--config-file", "unused.json",
         "--output-dir", str(tmp_path), "--modes", "regular_wsd"],
    )

    assert module.main() == 0
    assert summary.read_text() == "mode,tiles\nregular_wsd,2\n"
    assert runs.read_text() == "mode,repeat_index,tiles\nregular_wsd,0,2\n"


def _make_grid_result(
    *,
    columns: int,
    rows: int,
    tile_size_px: int,
    step_px: int | None = None,
) -> preprocessing_mod.TilingResult:
    if step_px is None:
        step_px = tile_size_px
    x_coords: list[int] = []
    y_coords: list[int] = []
    for x_idx in range(columns):
        for y_idx in range(rows):
            x_coords.append(x_idx * step_px)
            y_coords.append(y_idx * step_px)
    overlap = 0.0 if step_px == tile_size_px else 1.0 - (step_px / tile_size_px)
    x = np.asarray(x_coords, dtype=np.int64)
    y = np.asarray(y_coords, dtype=np.int64)
    return preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            x=x,
            y=y,
            tissue_fractions=np.zeros(columns * rows, dtype=np.float32),
            tile_index=np.arange(columns * rows, dtype=np.int32),
            requested_tile_size_px=tile_size_px,
            requested_spacing_um=0.5,
            read_level=0,
            read_tile_size_px=tile_size_px,
            read_spacing_um=0.5,
            tile_size_lv0=tile_size_px,
            is_within_tolerance=True,
            base_spacing_um=0.5,
            slide_dimensions=[columns * step_px + tile_size_px, rows * step_px + tile_size_px],
            level_downsamples=[1.0],
            overlap=overlap,
            min_tissue_fraction=0.1,
        ),
        sample_id="bench-slide",
        image_path=Path("/tmp/bench-slide.svs"),
        mask_path=None,
        backend="openslide",
        requested_backend="openslide",
        step_px_lv0=step_px,
        tolerance=0.05,
        tissue_method="unknown",
        requested_seg_downsample=64,
        seg_downsample=64,
        seg_level=0,
        seg_spacing_um=0.0,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=tile_size_px,
        a_t=4,
        a_h=0,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def _load_benchmark_script_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_tile_read.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_tile_read_script",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_benchmark_utils_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_tile_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_tile_utils",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_cucim_batch_mode_reports_region_and_tile_progress(monkeypatch):
    module = _load_benchmark_script_module()
    utils = _load_benchmark_utils_module()
    result = _make_grid_result(columns=4, rows=4, tile_size_px=8)
    plans = [
        utils.TileReadPlan(x=0, y=0, read_size_px=8, block_size=1),
        utils.TileReadPlan(x=0, y=0, read_size_px=32, block_size=4),
    ]

    class _FakeCuImage:
        metadata = {"cucim": {"resolutions": {
            "level_dimensions": [(32, 32)], "level_downsamples": [1.0],
        }}}

        def __init__(self, *_args, **_kwargs):
            pass

        def read_region(self, location, size, level, num_workers):
            assert level == 0
            assert num_workers == 2
            return [
                np.zeros((int(size[0]), int(size[1]), 3), dtype=np.uint8)
                for _ in location
            ]

    updates: list[tuple[int, int]] = []

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "cucim", SimpleNamespace(CuImage=_FakeCuImage))
        elapsed, tile_count, checksum = module.benchmark_cucim_batch_mode(
            result=result,
            plans=plans,
            read_step_px=8,
            num_workers=2,
            progress_callback=lambda regions, tiles: updates.append((regions, tiles)),
        )

    assert elapsed >= 0.0
    assert tile_count == 17
    assert checksum == 0
    assert updates == [(1, 1), (1, 16)]
