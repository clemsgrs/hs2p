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
            origin_x=100,
            origin_y=200,
            block_size=4,
            tile_size_px=12,
            read_step_px=8,
        )
    )

    assert len(tiles) == 16
    assert tiles[0].x == 100 and tiles[0].y == 200
    assert tiles[1].x == 100 and tiles[1].y == 208
    assert tiles[4].x == 108 and tiles[4].y == 200
    assert tiles[0].tile_arr.shape == (12, 12, 3)
    assert int(tiles[1].tile_arr[0, 0, 0]) == 2
