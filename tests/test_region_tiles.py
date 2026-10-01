import numpy as np

from hs2p.wsi.streaming.regions import iter_region_tile_views


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


def test_iter_region_tile_views_uses_stride_for_overlap_reads():
    region = _make_grouped_region(block_size=4, tile_size_px=12, step_px=8)

    tiles = list(
        iter_region_tile_views(
            region,
            block_size=4,
            tile_size_px=12,
            read_step_px=8,
        )
    )

    assert len(tiles) == 16
    assert (tiles[0].crop_x, tiles[0].crop_y) == (0, 0)
    assert (tiles[1].crop_x, tiles[1].crop_y) == (0, 8)
    assert (tiles[4].crop_x, tiles[4].crop_y) == (8, 0)
    assert tiles[0].tile_arr.shape == (12, 12, 3)
    assert int(tiles[1].tile_arr[0, 0, 0]) == 2


def test_planned_views_carry_the_plan_origins_not_crop_offsets():
    from hs2p.wsi.streaming.plans import GroupedReadPlan
    from hs2p.wsi.streaming.regions import iter_plan_region_tile_views

    # a 2x2 group read at a 2x level: origins are 32 level-0 px apart, crops 16 px
    plan = GroupedReadPlan(
        x=0,
        y=0,
        read_size_px=32,
        block_size=2,
        tile_indices=(5, 6, 7, 8),
        tile_origins=((0, 0), (0, 32), (32, 0), (32, 32)),
    )
    region = _make_grouped_region(block_size=2, tile_size_px=16, step_px=16)

    records = list(
        iter_plan_region_tile_views(region, read_plan=plan, tile_size_px=16, read_step_px=16)
    )

    assert [(r.tile_index, r.x, r.y) for r in records] == [
        (5, 0, 0),
        (6, 0, 32),
        (7, 32, 0),
        (8, 32, 32),
    ]
    assert [int(r.tile_arr[0, 0, 0]) for r in records] == [1, 2, 3, 4]
