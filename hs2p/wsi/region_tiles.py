from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hs2p.wsi.read_plans import GroupedReadPlan


@dataclass(frozen=True)
class RegionTileView:
    x: int
    y: int
    tile_arr: np.ndarray


@dataclass(frozen=True)
class PlannedRegionTileView:
    tile_index: int
    x: int
    y: int
    tile_arr: np.ndarray


def iter_region_tile_views(
    region: np.ndarray,
    *,
    origin_x: int,
    origin_y: int,
    block_size: int,
    tile_size_px: int,
    read_step_px: int,
):
    region = np.asarray(region)
    if int(block_size) == 1:
        yield RegionTileView(
            x=int(origin_x),
            y=int(origin_y),
            tile_arr=region[:tile_size_px, :tile_size_px],
        )
        return
    for x_idx in range(int(block_size)):
        x0 = x_idx * int(read_step_px)
        for y_idx in range(int(block_size)):
            y0 = y_idx * int(read_step_px)
            yield RegionTileView(
                x=int(origin_x + x0),
                y=int(origin_y + y0),
                tile_arr=region[
                    y0 : y0 + int(tile_size_px),
                    x0 : x0 + int(tile_size_px),
                ],
            )


def iter_plan_region_tile_views(
    region: np.ndarray,
    *,
    read_plan: GroupedReadPlan,
    tile_size_px: int,
    read_step_px: int,
):
    tile_index_iter = iter(int(idx) for idx in read_plan.tile_indices)
    for tile_view in iter_region_tile_views(
        region,
        origin_x=int(read_plan.x),
        origin_y=int(read_plan.y),
        block_size=int(read_plan.block_size),
        tile_size_px=int(tile_size_px),
        read_step_px=int(read_step_px),
    ):
        yield PlannedRegionTileView(
            tile_index=next(tile_index_iter),
            x=int(tile_view.x),
            y=int(tile_view.y),
            tile_arr=tile_view.tile_arr,
        )
