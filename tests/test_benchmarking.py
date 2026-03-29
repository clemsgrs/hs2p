import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hs2p.api import TilingResult

pytestmark = pytest.mark.script


def _make_grid_result(
    *,
    columns: int,
    rows: int,
    tile_size_px: int,
    step_px: int | None = None,
) -> TilingResult:
    if step_px is None:
        step_px = tile_size_px
    x_coords: list[int] = []
    y_coords: list[int] = []
    for x_idx in range(columns):
        for y_idx in range(rows):
            x_coords.append(x_idx * step_px)
            y_coords.append(y_idx * step_px)
    return TilingResult(
        sample_id="bench-slide",
        image_path=Path("/tmp/bench-slide.svs"),
        mask_path=None,
        backend="openslide",
        x=np.asarray(x_coords, dtype=np.int64),
        y=np.asarray(y_coords, dtype=np.int64),
        tile_index=np.arange(columns * rows, dtype=np.int32),
        target_spacing_um=0.5,
        target_tile_size_px=tile_size_px,
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=tile_size_px,
        tile_size_lv0=tile_size_px,
        overlap=0.0 if step_px == tile_size_px else 1.0 - (step_px / tile_size_px),
        tissue_threshold=0.1,
        num_tiles=columns * rows,
        config_hash="bench-hash",
        read_step_px=step_px,
        step_px_lv0=step_px,
    )


def _make_custom_result(
    *,
    coords: list[tuple[int, int]],
    tile_size_px: int,
) -> TilingResult:
    return TilingResult(
        sample_id="bench-slide",
        image_path=Path("/tmp/bench-slide.svs"),
        mask_path=None,
        backend="openslide",
        x=np.asarray([x for x, _ in coords], dtype=np.int64),
        y=np.asarray([y for _, y in coords], dtype=np.int64),
        tile_index=np.arange(len(coords), dtype=np.int32),
        target_spacing_um=0.5,
        target_tile_size_px=tile_size_px,
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=tile_size_px,
        tile_size_lv0=tile_size_px,
        overlap=0.0,
        tissue_threshold=0.1,
        num_tiles=len(coords),
        config_hash="bench-hash",
        read_step_px=tile_size_px,
        step_px_lv0=tile_size_px,
    )


def _make_grouped_region(*, block_size: int, tile_size_px: int, step_px: int) -> np.ndarray:
    region_size = tile_size_px + (block_size - 1) * step_px
    region = np.zeros((region_size, region_size, 3), dtype=np.uint8)
    for x_idx in range(block_size):
        for y_idx in range(block_size):
            tile_value = x_idx * block_size + y_idx + 1
            x0 = x_idx * step_px
            y0 = y_idx * step_px
            region[y0 : y0 + tile_size_px, x0 : x0 + tile_size_px] = tile_value
    return region


def _load_benchmark_script_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_tile_read_strategies.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_tile_read_strategies_script",
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


def test_build_read_plans_without_supertiles_uses_one_plan_per_tile():
    mod = _load_benchmark_utils_module()
    result = _make_grid_result(columns=3, rows=1, tile_size_px=32)

    plans = mod.build_read_plans(result, use_supertiles=False)

    assert plans == [
        mod.TileReadPlan(x=0, y=0, read_size_px=32, block_size=1),
        mod.TileReadPlan(x=32, y=0, read_size_px=32, block_size=1),
        mod.TileReadPlan(x=64, y=0, read_size_px=32, block_size=1),
    ]


def test_build_read_plans_with_supertiles_prefers_dense_8x8_blocks():
    mod = _load_benchmark_utils_module()
    result = _make_grid_result(columns=8, rows=8, tile_size_px=32)

    plans = mod.build_read_plans(result, use_supertiles=True)

    assert plans == [mod.TileReadPlan(x=0, y=0, read_size_px=256, block_size=8)]


def test_build_read_plans_with_supertiles_uses_4x4_when_8x8_is_not_available():
    mod = _load_benchmark_utils_module()
    result = _make_grid_result(columns=4, rows=4, tile_size_px=32)

    plans = mod.build_read_plans(result, use_supertiles=True)

    assert plans == [mod.TileReadPlan(x=0, y=0, read_size_px=128, block_size=4)]


def test_build_read_plans_with_supertiles_prioritizes_larger_blocks_before_singles():
    mod = _load_benchmark_utils_module()
    result = _make_custom_result(
        coords=[
            (0, 0),
            (100, 0),
            (100, 16),
            (116, 0),
            (116, 16),
        ],
        tile_size_px=16,
    )

    plans = mod.build_read_plans(result, use_supertiles=True)

    assert plans == [
        mod.TileReadPlan(x=100, y=0, read_size_px=32, block_size=2),
        mod.TileReadPlan(x=0, y=0, read_size_px=16, block_size=1),
    ]


def test_group_read_plans_by_read_size_preserves_first_seen_order():
    mod = _load_benchmark_utils_module()
    plans = [
        mod.TileReadPlan(x=0, y=0, read_size_px=256, block_size=8),
        mod.TileReadPlan(x=512, y=0, read_size_px=32, block_size=1),
        mod.TileReadPlan(x=768, y=0, read_size_px=256, block_size=8),
    ]

    grouped = mod.group_read_plans_by_read_size(plans)

    assert list(grouped) == [256, 32]
    assert grouped[256] == [plans[0], plans[2]]
    assert grouped[32] == [plans[1]]


def test_iter_tiles_from_region_slices_in_x_major_order():
    mod = _load_benchmark_utils_module()
    region = _make_grouped_region(block_size=4, tile_size_px=8, step_px=8)
    plan = mod.TileReadPlan(x=0, y=0, read_size_px=32, block_size=4)

    tiles = list(mod.iter_tiles_from_region(region, plan, tile_size_px=8, read_step_px=8))

    assert len(tiles) == 16
    assert int(tiles[0][0, 0, 0]) == 1
    assert int(tiles[1][0, 0, 0]) == 2
    assert int(tiles[4][0, 0, 0]) == 5
    assert int(tiles[-1][0, 0, 0]) == 16


def test_iter_tiles_from_region_uses_stride_for_overlap_reads():
    mod = _load_benchmark_utils_module()
    region = _make_grouped_region(block_size=4, tile_size_px=12, step_px=8)
    plan = mod.TileReadPlan(x=0, y=0, read_size_px=36, block_size=4)

    tiles = list(mod.iter_tiles_from_region(region, plan, tile_size_px=12, read_step_px=8))

    assert len(tiles) == 16
    assert tiles[0].shape == (12, 12, 3)
    assert int(tiles[1][0, 0, 0]) == 2


def test_limit_tiling_result_trims_arrays_and_reindexes_tiles():
    mod = _load_benchmark_utils_module()
    result = _make_grid_result(columns=3, rows=2, tile_size_px=16)

    limited = mod.limit_tiling_result(result, max_tiles=4)

    assert limited.num_tiles == 4
    np.testing.assert_array_equal(limited.x, np.array([0, 0, 16, 16], dtype=np.int64))
    np.testing.assert_array_equal(limited.y, np.array([0, 16, 0, 16], dtype=np.int64))
    np.testing.assert_array_equal(limited.tile_index, np.array([0, 1, 2, 3], dtype=np.int32))


def test_load_single_slide_result_from_config_builds_fresh_tiling_result(tmp_path: Path):
    module = _load_benchmark_script_module()
    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "input"
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text(
        "sample_id,image_path,mask_path\n"
        f"test-wsi,{fixture_dir / 'test-wsi.tif'},{fixture_dir / 'test-mask.tif'}\n"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"csv: {csv_path}\n"
        f"output_dir: {tmp_path / 'output'}\n"
        "save_previews: false\n"
        "save_tiles: false\n"
        "resume: false\n"
        "tiling:\n"
        "  backend: asap\n"
        "  params:\n"
        "    target_spacing_um: 0.5\n"
        "    target_tile_size_px: 224\n"
        "    tolerance: 0.05\n"
        "    overlap: 0.0\n"
        "    tissue_threshold: 0.1\n"
        "    use_padding: true\n"
    )

    result = module.load_single_slide_result_from_config(
        config_file=config_path,
        num_workers=1,
    )

    assert result.sample_id == "test-wsi"
    assert result.read_step_px > 0
    assert result.step_px_lv0 > 0
    assert result.num_tiles > 0


def test_benchmark_wsd_mode_reports_region_and_tile_progress(monkeypatch):
    module = _load_benchmark_script_module()
    utils = _load_benchmark_utils_module()
    result = _make_grid_result(columns=2, rows=1, tile_size_px=8)
    plans = [
        utils.TileReadPlan(x=0, y=0, read_size_px=8, block_size=1),
        utils.TileReadPlan(x=8, y=0, read_size_px=8, block_size=1),
    ]

    seen: dict[str, object] = {}

    class _FakeWSI:
        def __init__(self, *_args, **_kwargs):
            seen["args"] = _args
            seen["kwargs"] = _kwargs

        def get_patch(self, *_args, **_kwargs):
            return np.ones((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(
        module,
        "coerce_wsd_path",
        lambda image_path, backend: f"coerced:{image_path}:{backend}",
    )
    updates: list[tuple[int, int]] = []

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(
            sys.modules,
            "wholeslidedata",
            SimpleNamespace(WholeSlideImage=_FakeWSI),
        )
        elapsed, tile_count, checksum = module.benchmark_wsd_mode(
            result=result,
            plans=plans,
            read_step_px=8,
            progress_callback=lambda regions, tiles: updates.append((regions, tiles)),
        )

    assert elapsed >= 0.0
    assert tile_count == 2
    assert checksum > 0
    assert updates == [(1, 1), (1, 1)]
    assert seen["args"] == ("coerced:/tmp/bench-slide.svs:openslide",)
    assert seen["kwargs"] == {"backend": "openslide"}


def test_benchmark_cucim_batch_mode_reports_region_and_tile_progress(monkeypatch):
    module = _load_benchmark_script_module()
    utils = _load_benchmark_utils_module()
    result = _make_grid_result(columns=4, rows=4, tile_size_px=8)
    plans = [
        utils.TileReadPlan(x=0, y=0, read_size_px=8, block_size=1),
        utils.TileReadPlan(x=0, y=0, read_size_px=32, block_size=4),
    ]

    class _FakeCuImage:
        def __init__(self, *_args, **_kwargs):
            pass

        def read_region(self, locations, size, level, num_workers):
            assert level == 0
            assert num_workers == 2
            return [
                np.zeros((int(size[0]), int(size[1]), 3), dtype=np.uint8)
                for _ in locations
            ]

    monkeypatch.setattr(
        module,
        "_require_cucim",
        lambda: SimpleNamespace(CuImage=_FakeCuImage),
    )
    updates: list[tuple[int, int]] = []

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
