from dataclasses import replace
from pathlib import Path

import numpy as np

import hs2p.preprocessing as preprocessing_mod
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
    return preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            coordinates=np.column_stack(
                [
                    np.asarray(x_coords, dtype=np.int64),
                    np.asarray(y_coords, dtype=np.int64),
                ]
            ),
            tissue_fractions=np.zeros(columns * rows, dtype=np.float32),
            tile_index=np.arange(columns * rows, dtype=np.int32),
            requested_tile_size_px=tile_size_px,
            requested_spacing_um=0.5,
            read_level=0,
            effective_tile_size_px=tile_size_px,
            effective_spacing_um=0.5,
            tile_size_lv0=tile_size_px,
            is_within_tolerance=True,
            base_spacing_um=0.5,
            slide_dimensions=[columns * step_px + tile_size_px, rows * step_px + tile_size_px],
            level_downsamples=[1.0],
            overlap=overlap,
            min_tissue_fraction=0.1,
            use_padding=True,
        ),
        sample_id="read-plan-slide",
        image_path=Path("/tmp/read-plan-slide.svs"),
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
    result = replace(result, step_px_lv0=None)

    assert resolve_step_px_lv0(result) == 24
