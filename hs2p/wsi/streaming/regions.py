
from dataclasses import dataclass

import numpy as np

from hs2p.wsi.streaming.plans import GroupedReadPlan


@dataclass(frozen=True)
class TileView:
    """One tile cropped out of a region read; ``crop_x``/``crop_y`` are read-level
    offsets inside that region, not slide coordinates."""

    crop_x: int
    crop_y: int
    tile_arr: np.ndarray


@dataclass(frozen=True)
class PlannedTileView:
    """One tile with its index and level-0 origin from the tiling result."""

    tile_index: int
    x: int
    y: int
    tile_arr: np.ndarray


def iter_region_tile_views(
    region: np.ndarray,
    *,
    block_size: int,
    tile_size_px: int,
    read_step_px: int,
):
    """Crop the ``block_size`` x ``block_size`` tiles out of a grouped region, outer-X /
    inner-Y, each ``read_step_px`` read-level pixels apart."""
    region = np.asarray(region)
    if int(block_size) == 1:
        yield TileView(crop_x=0, crop_y=0, tile_arr=region[:tile_size_px, :tile_size_px])
        return
    for x_idx in range(int(block_size)):
        x0 = x_idx * int(read_step_px)
        for y_idx in range(int(block_size)):
            y0 = y_idx * int(read_step_px)
            yield TileView(
                crop_x=x0,
                crop_y=y0,
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
    """Pair each crop of ``region`` with the tile index and level-0 origin the plan
    recorded for it. The origin is the saved coordinate itself, so manifests and records
    name the same slide location as the coordinate arrays whatever the read level."""
    members = zip(read_plan.tile_indices, read_plan.tile_origins, strict=True)
    for (tile_index, (x, y)), tile_view in zip(
        members,
        iter_region_tile_views(
            region,
            block_size=int(read_plan.block_size),
            tile_size_px=int(tile_size_px),
            read_step_px=int(read_step_px),
        ),
        strict=True,
    ):
        yield PlannedTileView(
            tile_index=int(tile_index),
            x=int(x),
            y=int(y),
            tile_arr=tile_view.tile_arr,
        )
