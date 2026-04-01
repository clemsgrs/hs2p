from __future__ import annotations
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from hs2p.wsi.backends.common import resolve_padded_read_bounds
from hs2p.wsi.streaming.batched import BatchedReadRequest, iter_cucim_batched_read_regions
from hs2p.wsi.streaming.plans import (
    GroupedReadPlan,
    iter_grouped_read_plans,
    resolve_read_step_px,
    resolve_step_px_lv0,
)
from hs2p.wsi.reader import SlideReader, open_slide
from hs2p.wsi.streaming.regions import PlannedTileView, iter_plan_region_tile_views


def open_reader_for_result(
    result: Any,
    *,
    gpu_decode: bool = False,
) -> SlideReader:
    return open_slide(
        result.image_path,
        backend=str(result.backend),
        spacing_override=result.base_spacing_um,
        gpu_decode=gpu_decode,
    )


def iter_tile_records_from_reader(
    reader: SlideReader,
    *,
    result: Any,
    supertile_sizes: Sequence[int] | None = None,
) -> Iterator[PlannedTileView]:
    read_step_px = resolve_read_step_px(result)
    step_px_lv0 = resolve_step_px_lv0(result)
    read_plans = iter_grouped_read_plans(
        result=result,
        read_step_px=read_step_px,
        step_px_lv0=step_px_lv0,
        supertile_sizes=supertile_sizes,
    )
    yield from _iter_tile_records_from_reader_plans(
        reader,
        read_plans=read_plans,
        read_level=int(result.read_level),
        tile_size_px=int(result.effective_tile_size_px),
        read_step_px=int(read_step_px),
    )


def iter_tile_records_from_result(
    *,
    result: Any,
    num_workers: int = 1,
    gpu_decode: bool = False,
    supertile_sizes: Sequence[int] | None = None,
) -> Iterator[PlannedTileView]:
    if str(result.backend) == "cucim":
        yield from _iter_cucim_tile_records_from_result(
            result=result,
            num_workers=num_workers,
            gpu_decode=gpu_decode,
            supertile_sizes=supertile_sizes,
        )
        return
    with open_reader_for_result(result, gpu_decode=gpu_decode) as reader:
        yield from iter_tile_records_from_reader(
            reader,
            result=result,
            supertile_sizes=supertile_sizes,
        )


def iter_tile_arrays_from_result(
    *,
    result: Any,
    num_workers: int = 1,
    gpu_decode: bool = False,
    supertile_sizes: Sequence[int] | None = None,
) -> Iterator[np.ndarray]:
    for record in iter_tile_records_from_result(
        result=result,
        num_workers=num_workers,
        gpu_decode=gpu_decode,
        supertile_sizes=supertile_sizes,
    ):
        yield record.tile_arr


def _iter_tile_records_from_region(
    region: np.ndarray,
    *,
    read_plan: GroupedReadPlan,
    tile_size_px: int,
    read_step_px: int,
) -> Iterator[PlannedTileView]:
    for tile_view in iter_plan_region_tile_views(
        np.asarray(region),
        read_plan=read_plan,
        tile_size_px=int(tile_size_px),
        read_step_px=int(read_step_px),
    ):
        yield tile_view


def _iter_tile_records_from_reader_plans(
    reader: SlideReader,
    *,
    read_plans,
    read_level: int,
    tile_size_px: int,
    read_step_px: int,
) -> Iterator[PlannedTileView]:
    for read_plan in read_plans:
        region = reader.read_region(
            (int(read_plan.x), int(read_plan.y)),
            int(read_level),
            (int(read_plan.read_size_px), int(read_plan.read_size_px)),
        )
        yield from _iter_tile_records_from_region(
            np.asarray(region),
            read_plan=read_plan,
            tile_size_px=int(tile_size_px),
            read_step_px=int(read_step_px),
        )


def _iter_cucim_tile_records_from_result(
    *,
    result: Any,
    num_workers: int,
    gpu_decode: bool,
    supertile_sizes: Sequence[int] | None,
) -> Iterator[PlannedTileView]:
    read_step_px = resolve_read_step_px(result)
    step_px_lv0 = resolve_step_px_lv0(result)
    read_plans = list(
        iter_grouped_read_plans(
            result=result,
            read_step_px=read_step_px,
            step_px_lv0=step_px_lv0,
            supertile_sizes=supertile_sizes,
        )
    )
    requested_level_dimensions = tuple(result.level_dimensions[int(result.read_level)])
    requested_downsample = float(result.level_downsamples[int(result.read_level)][0])
    in_bounds_requests = []
    padded_plans = []
    for read_plan in read_plans:
        bounds = resolve_padded_read_bounds(
            location=(int(read_plan.x), int(read_plan.y)),
            size=(int(read_plan.read_size_px), int(read_plan.read_size_px)),
            level_dimensions=requested_level_dimensions,
            downsample=requested_downsample,
        )
        if (
            bounds.read_size == (int(read_plan.read_size_px), int(read_plan.read_size_px))
            and bounds.paste_offset == (0, 0)
        ):
            in_bounds_requests.append(
                BatchedReadRequest(
                    location=(int(read_plan.x), int(read_plan.y)),
                    size=(int(read_plan.read_size_px), int(read_plan.read_size_px)),
                    payload=read_plan,
                )
            )
        else:
            padded_plans.append(read_plan)
    batched_regions = iter_cucim_batched_read_regions(
        image_path=result.image_path,
        requests=in_bounds_requests,
        level=int(result.read_level),
        num_workers=int(num_workers),
        gpu_decode=gpu_decode,
    )

    for request, region in batched_regions:
        yield from _iter_tile_records_from_region(
            np.asarray(region),
            read_plan=request.payload,
            tile_size_px=int(result.effective_tile_size_px),
            read_step_px=int(read_step_px),
        )

    if not padded_plans:
        return

    with open_reader_for_result(result, gpu_decode=gpu_decode) as reader:
        for read_plan in padded_plans:
            region = reader.read_region(
                (int(read_plan.x), int(read_plan.y)),
                int(result.read_level),
                (int(read_plan.read_size_px), int(read_plan.read_size_px)),
            )
            yield from _iter_tile_records_from_region(
                np.asarray(region),
                read_plan=read_plan,
                tile_size_px=int(result.effective_tile_size_px),
                read_step_px=int(read_step_px),
            )


__all__ = [
    "PlannedTileView",
    "iter_tile_arrays_from_result",
    "iter_tile_records_from_reader",
    "iter_tile_records_from_result",
    "open_reader_for_result",
]
