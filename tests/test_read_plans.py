from pathlib import Path

import numpy as np

from hs2p.api import TilingResult
from hs2p.wsi.read_plans import (
    GroupedReadPlan,
    iter_grouped_read_plans,
    resolve_read_step_px,
    resolve_step_px_lv0,
)


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
        sample_id="read-plan-slide",
        image_path=Path("/tmp/read-plan-slide.svs"),
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
        config_hash="read-plan-hash",
        read_step_px=step_px,
        step_px_lv0=step_px,
    )


def test_iter_grouped_read_plans_prefers_dense_4x4_blocks():
    result = _make_grid_result(columns=4, rows=4, tile_size_px=32)

    plans = list(
        iter_grouped_read_plans(
            result=result,
            read_step_px=resolve_read_step_px(result),
            step_px_lv0=resolve_step_px_lv0(result),
        )
    )

    assert plans == [
        GroupedReadPlan(
            x=0,
            y=0,
            read_size_px=128,
            block_size=4,
            tile_indices=tuple(range(16)),
        )
    ]


def test_resolve_step_px_lv0_uses_smallest_positive_stride_when_metadata_missing():
    result = _make_grid_result(columns=3, rows=1, tile_size_px=32, step_px=24)
    result.step_px_lv0 = None

    assert resolve_step_px_lv0(result) == 24
