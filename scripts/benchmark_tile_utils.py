from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hs2p.api import TilingResult
from hs2p.wsi.read_plans import (
    iter_grouped_read_plans,
    resolve_read_step_px,
    resolve_step_px_lv0,
)
from hs2p.wsi.region_tiles import iter_region_tile_views


@dataclass(frozen=True)
class TileReadPlan:
    x: int
    y: int
    read_size_px: int
    block_size: int


def build_read_plans(
    result: TilingResult,
    *,
    use_supertiles: bool,
) -> list[TileReadPlan]:
    if not use_supertiles:
        tile_size_px = int(result.read_tile_size_px)
        return [
            TileReadPlan(
                x=int(x),
                y=int(y),
                read_size_px=tile_size_px,
                block_size=1,
            )
            for x, y in zip(
                result.x.astype(np.int64, copy=False).tolist(),
                result.y.astype(np.int64, copy=False).tolist(),
            )
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


def limit_tiling_result(result: TilingResult, *, max_tiles: int) -> TilingResult:
    if max_tiles <= 0 or max_tiles >= result.num_tiles:
        return result
    kept = slice(0, int(max_tiles))
    return TilingResult(
        sample_id=result.sample_id,
        image_path=result.image_path,
        mask_path=result.mask_path,
        backend=result.backend,
        x=result.x[kept],
        y=result.y[kept],
        tile_index=np.arange(int(max_tiles), dtype=np.int32),
        target_spacing_um=result.target_spacing_um,
        target_tile_size_px=result.target_tile_size_px,
        read_level=result.read_level,
        read_spacing_um=result.read_spacing_um,
        read_tile_size_px=result.read_tile_size_px,
        tile_size_lv0=result.tile_size_lv0,
        overlap=result.overlap,
        tissue_threshold=result.tissue_threshold,
        num_tiles=int(max_tiles),
        config_hash=result.config_hash,
        read_step_px=result.read_step_px,
        step_px_lv0=result.step_px_lv0,
        tissue_fraction=(
            result.tissue_fraction[kept]
            if result.tissue_fraction is not None
            else None
        ),
        annotation=result.annotation,
        selection_strategy=result.selection_strategy,
        output_mode=result.output_mode,
    )
