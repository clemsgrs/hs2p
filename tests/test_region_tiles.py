from pathlib import Path

import numpy as np

import hs2p.preprocessing as preprocessing_mod
from hs2p.wsi.streaming.plans import GroupedReadPlan, resolve_read_step_px
from hs2p.wsi.streaming.regions import iter_plan_region_tile_views, iter_region_tile_views


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


def _make_result(
    *,
    tile_size_px: int,
    step_px: int,
) -> preprocessing_mod.TilingResult:
    overlap = 0.0 if step_px == tile_size_px else 1.0 - (step_px / tile_size_px)
    coords = np.array(
        [[0, 0], [0, step_px], [step_px, 0], [step_px, step_px]],
        dtype=np.int64,
    )
    return preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            x=coords[:, 0],
            y=coords[:, 1],
            tissue_fractions=np.zeros(4, dtype=np.float32),
            tile_index=np.arange(4, dtype=np.int32),
            requested_tile_size_px=tile_size_px,
            requested_spacing_um=0.5,
            read_level=0,
            read_tile_size_px=tile_size_px,
            read_spacing_um=0.5,
            tile_size_lv0=tile_size_px,
            is_within_tolerance=True,
            base_spacing_um=0.5,
            slide_dimensions=[(2 * step_px) + tile_size_px, (2 * step_px) + tile_size_px],
            level_downsamples=[1.0],
            overlap=overlap,
            min_tissue_fraction=0.1,
        ),
        sample_id="region-tiles-slide",
        image_path=Path("/tmp/region-tiles-slide.svs"),
        mask_path=None,
        backend="openslide",
        requested_backend="openslide",
        step_px_lv0=step_px,
        tolerance=0.05,
        tissue_method="unknown",
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


def test_iter_region_tile_views_slices_in_x_major_order():
    region = _make_grouped_region(block_size=4, tile_size_px=8, step_px=8)

    tiles = list(
        iter_region_tile_views(
            region,
            origin_x=0,
            origin_y=0,
            block_size=4,
            tile_size_px=8,
            read_step_px=8,
        )
    )

    assert len(tiles) == 16
    assert tiles[0].x == 0 and tiles[0].y == 0
    assert int(tiles[0].tile_arr[0, 0, 0]) == 1
    assert int(tiles[1].tile_arr[0, 0, 0]) == 2
    assert int(tiles[4].tile_arr[0, 0, 0]) == 5
    assert int(tiles[-1].tile_arr[0, 0, 0]) == 16


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


def test_iter_plan_region_tile_views_attaches_tile_indices_in_slice_order():
    result = _make_result(tile_size_px=8, step_px=8)
    plan = GroupedReadPlan(
        x=0,
        y=0,
        read_size_px=16,
        block_size=2,
        tile_indices=(0, 1, 2, 3),
    )
    region = _make_grouped_region(block_size=2, tile_size_px=8, step_px=8)

    tiles = list(
        iter_plan_region_tile_views(
            region,
            read_plan=plan,
            tile_size_px=int(result.read_tile_size_px),
            read_step_px=resolve_read_step_px(result),
        )
    )

    assert [tile.tile_index for tile in tiles] == [0, 1, 2, 3]
    assert [(tile.x, tile.y) for tile in tiles] == [(0, 0), (0, 8), (8, 0), (8, 8)]
    assert int(tiles[0].tile_arr[0, 0, 0]) == 1
    assert int(tiles[-1].tile_arr[0, 0, 0]) == 4
