from pathlib import Path

import numpy as np

import hs2p.preprocessing as preprocessing_mod
from hs2p.wsi.streaming.plans import (
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
        sample_id="read-plan-slide",
        image_path=Path("/tmp/read-plan-slide.svs"),
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


def _two_x_overlap_result(*, step_px_lv0: int, columns: int = 8, rows: int = 8):
    """A 16 px read at a 2x level with 10% overlap, origins ``step_px_lv0`` apart."""
    from dataclasses import replace

    result = _make_grid_result(
        columns=columns, rows=rows, tile_size_px=16, step_px=step_px_lv0
    )
    return replace(
        result,
        tiles=replace(
            result.tiles,
            read_level=1,
            read_tile_size_px=16,
            tile_size_lv0=32,
            level_downsamples=[1.0, 2.0],
            overlap=0.1,
        ),
        step_px_lv0=step_px_lv0,
    )


def _crop_offsets_match_individual_reads(result, plans, *, read_step_px: int) -> bool:
    downsample = float(result.level_downsamples[result.read_level])
    for plan in plans:
        origin_x = int(np.floor(plan.x / downsample))
        origin_y = int(np.floor(plan.y / downsample))
        for pos, tile_idx in enumerate(plan.tile_indices):
            x_idx, y_idx = pos // plan.block_size, pos % plan.block_size
            own_x = int(np.floor(int(result.x[tile_idx]) / downsample))
            own_y = int(np.floor(int(result.y[tile_idx]) / downsample))
            if (own_x, own_y) != (origin_x + x_idx * read_step_px, origin_y + y_idx * read_step_px):
                return False
    return True


def test_grouped_plans_keep_full_blocks_when_the_stride_projects_exactly():
    result = _two_x_overlap_result(step_px_lv0=28)
    read_step_px = resolve_read_step_px(result)
    assert read_step_px == 14

    plans = list(
        iter_grouped_read_plans(
            result=result, read_step_px=read_step_px, step_px_lv0=resolve_step_px_lv0(result)
        )
    )

    assert [plan.block_size for plan in plans] == [8]
    assert plans[0].read_size_px == 16 + 7 * 14
    assert _crop_offsets_match_individual_reads(result, plans, read_step_px=read_step_px)


def test_grouped_plans_only_join_members_whose_origins_land_on_the_crop_grid():
    # a legacy artifact: level-0 stride 29 (28.8 rounded on its own) is 14.5 px on the
    # 2x level while the crop stride is 14, so origins drift by one pixel every two tiles
    result = _two_x_overlap_result(step_px_lv0=29)
    read_step_px = resolve_read_step_px(result)

    plans = list(
        iter_grouped_read_plans(
            result=result, read_step_px=read_step_px, step_px_lv0=resolve_step_px_lv0(result)
        )
    )

    assert sorted(idx for plan in plans for idx in plan.tile_indices) == list(range(64))
    assert max(plan.block_size for plan in plans) == 2
    assert _crop_offsets_match_individual_reads(result, plans, read_step_px=read_step_px)
