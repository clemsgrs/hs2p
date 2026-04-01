
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

import hs2p.preprocessing as preprocessing_mod
from hs2p.wsi.streaming.plans import (
    iter_grouped_read_plans,
    resolve_read_step_px,
    resolve_step_px_lv0,
)
from hs2p.wsi.streaming.regions import iter_region_tile_views


@dataclass(frozen=True)
class TileReadPlan:
    x: int
    y: int
    read_size_px: int
    block_size: int


def build_read_plans(
    result: preprocessing_mod.TilingResult,
    *,
    use_supertiles: bool,
) -> list[TileReadPlan]:
    if not use_supertiles:
        tile_size_px = int(result.effective_tile_size_px)
        return [
            TileReadPlan(
                x=int(x),
                y=int(y),
                read_size_px=tile_size_px,
                block_size=1,
            )
            for x, y in zip(np.asarray(result.x, dtype=np.int64), np.asarray(result.y, dtype=np.int64))
        ]

    read_step_px = resolve_read_step_px(result)
    step_px_lv0 = resolve_step_px_lv0(result)
    return [
        TileReadPlan(
            x=int(plan.x),
            y=int(plan.y),
            read_size_px=int(plan.read_size_px),
            block_size=int(plan.block_size),
        )
        for plan in iter_grouped_read_plans(
            result=result,
            read_step_px=read_step_px,
            step_px_lv0=step_px_lv0,
        )
    ]


def group_read_plans_by_read_size(
    plans: Iterable[TileReadPlan],
) -> dict[int, list[TileReadPlan]]:
    grouped: dict[int, list[TileReadPlan]] = {}
    for plan in plans:
        grouped.setdefault(int(plan.read_size_px), []).append(plan)
    return grouped


def iter_tiles_from_region(
    region: np.ndarray,
    plan: TileReadPlan,
    *,
    tile_size_px: int,
    read_step_px: int,
):
    for tile_view in iter_region_tile_views(
        region,
        origin_x=int(plan.x),
        origin_y=int(plan.y),
        block_size=int(plan.block_size),
        tile_size_px=int(tile_size_px),
        read_step_px=int(read_step_px),
    ):
        yield tile_view.tile_arr


def limit_tiling_result(
    result: preprocessing_mod.TilingResult,
    *,
    max_tiles: int,
) -> preprocessing_mod.TilingResult:
    if max_tiles <= 0 or max_tiles >= len(result.x):
        return result
    kept = slice(0, int(max_tiles))
    return replace(
        result,
        tiles=replace(
            result.tiles,
            x=result.x[kept],
            y=result.y[kept],
            tissue_fractions=result.tissue_fractions[kept],
            tile_index=np.arange(int(max_tiles), dtype=np.int32),
        ),
    )
