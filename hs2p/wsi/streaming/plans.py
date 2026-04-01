
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GroupedReadPlan:
    x: int
    y: int
    read_size_px: int
    block_size: int
    tile_indices: tuple[int, ...]


@dataclass
class SupertileIndex:
    """Random-access index over a set of grouped read plans.

    Attributes:
        plans: Ordered list of ``GroupedReadPlan`` objects (one per super-tile).
        tile_to_st: ``(num_tiles,)`` int32 array mapping tile index → super-tile id.
        tile_crop_x: ``(num_tiles,)`` int32 array of pixel X offset within the super-tile region.
        tile_crop_y: ``(num_tiles,)`` int32 array of pixel Y offset within the super-tile region.
        ordered_indices: ``(num_tiles,)`` int64 array of tile indices reordered so tiles
            belonging to the same super-tile are contiguous.
    """

    plans: list[GroupedReadPlan]
    tile_to_st: np.ndarray
    tile_crop_x: np.ndarray
    tile_crop_y: np.ndarray
    ordered_indices: np.ndarray


def resolve_read_step_px(result: Any) -> int:
    effective_tile_size_px = int(result.effective_tile_size_px)
    if effective_tile_size_px <= 0:
        raise ValueError("effective_tile_size_px must be > 0")
    return max(
        1,
        int(round(effective_tile_size_px * (1.0 - float(result.overlap)), 0)),
    )


def resolve_step_px_lv0(result: Any) -> int:
    if result.step_px_lv0 is not None:
        return int(result.step_px_lv0)
    x = np.asarray(result.x, dtype=np.int64)
    y = np.asarray(result.y, dtype=np.int64)
    if x.shape[0] > 1:
        unique_x = np.unique(np.sort(x))
        diffs = np.diff(unique_x)
        diffs = diffs[diffs > 0]
        if diffs.size > 0:
            return int(diffs.min())
        unique_y = np.unique(np.sort(y))
        diffs = np.diff(unique_y)
        diffs = diffs[diffs > 0]
        if diffs.size > 0:
            return int(diffs.min())
    return max(
        1,
        int(round(int(result.tile_size_lv0) * (1.0 - float(result.overlap)), 0)),
    )


def iter_grouped_read_plans(
    *,
    result: Any,
    read_step_px: int,
    step_px_lv0: int,
    supertile_sizes: Sequence[int] | None = None,
):
    if supertile_sizes is None:
        supertile_sizes = (8, 4, 2)
    grouped_sizes = tuple(
        sorted({int(size) for size in supertile_sizes if int(size) > 1}, reverse=True)
    )
    if step_px_lv0 <= 0:
        step_px_lv0 = int(result.tile_size_lv0)
    x = np.asarray(result.x, dtype=np.int64)
    y = np.asarray(result.y, dtype=np.int64)
    coordinates = np.column_stack((x, y))
    coord_to_index = {
        (int(x), int(y)): idx
        for idx, (x, y) in enumerate(coordinates.tolist())
    }
    consumed = np.zeros(len(coordinates), dtype=bool)
    tile_size_px = int(result.effective_tile_size_px)
    grouped_plans: dict[int, list[GroupedReadPlan]] = {size: [] for size in grouped_sizes}
    grouped_plans[1] = []

    def _build_grouped_plan(idx: int, block_size: int) -> GroupedReadPlan | None:
        if consumed[idx]:
            return None
        x0 = int(coordinates[idx, 0])
        y0 = int(coordinates[idx, 1])
        indices: list[int] = []
        for x_idx in range(block_size):
            for y_idx in range(block_size):
                coord = (
                    x0 + x_idx * step_px_lv0,
                    y0 + y_idx * step_px_lv0,
                )
                match_idx = coord_to_index.get(coord)
                if match_idx is None or consumed[match_idx]:
                    return None
                indices.append(match_idx)
            if len(indices) < (x_idx + 1) * block_size:
                return None
        return GroupedReadPlan(
            x=x0,
            y=y0,
            read_size_px=tile_size_px + (block_size - 1) * read_step_px,
            block_size=block_size,
            tile_indices=tuple(indices),
        )

    for block_size in grouped_sizes:
        if len(coordinates) < block_size * block_size:
            continue
        for idx in range(len(coordinates)):
            plan = _build_grouped_plan(idx, block_size)
            if plan is None:
                continue
            for match_idx in plan.tile_indices:
                consumed[match_idx] = True
            grouped_plans[block_size].append(plan)

    for idx in range(len(coordinates)):
        if consumed[idx]:
            continue
        consumed[idx] = True
        grouped_plans[1].append(
            GroupedReadPlan(
                x=int(coordinates[idx, 0]),
                y=int(coordinates[idx, 1]),
                read_size_px=tile_size_px,
                block_size=1,
                tile_indices=(idx,),
            )
        )

    for block_size in (*grouped_sizes, 1):
        yield from grouped_plans[block_size]


def group_read_plans_by_size(
    read_plans: Iterable[GroupedReadPlan],
) -> dict[int, list[GroupedReadPlan]]:
    grouped: dict[int, list[GroupedReadPlan]] = {}
    for plan in read_plans:
        grouped.setdefault(int(plan.read_size_px), []).append(plan)
    return grouped


def build_supertile_index(result: Any) -> SupertileIndex:
    """Build a random-access index over the grouped read plans for *result*.

    Returns a :class:`SupertileIndex` that maps every tile index to its
    super-tile id and pixel crop offsets within that super-tile region.
    This is the building block for random-access DataLoader tile reading.

    The iteration order within each :class:`GroupedReadPlan` follows the
    outer-X / inner-Y convention: ``tile_indices[pos]`` corresponds to
    ``x_idx = pos // block_size``, ``y_idx = pos % block_size``.
    """
    read_step_px = resolve_read_step_px(result)
    step_px_lv0 = resolve_step_px_lv0(result)

    num_tiles = int(result.num_tiles)
    tile_to_st = np.empty(num_tiles, dtype=np.int32)
    tile_crop_x = np.empty(num_tiles, dtype=np.int32)
    tile_crop_y = np.empty(num_tiles, dtype=np.int32)
    plans: list[GroupedReadPlan] = []
    ordered_indices: list[int] = []

    for plan in iter_grouped_read_plans(
        result=result,
        read_step_px=read_step_px,
        step_px_lv0=step_px_lv0,
    ):
        st_id = len(plans)
        for pos, tile_idx in enumerate(plan.tile_indices):
            x_idx = pos // plan.block_size
            y_idx = pos % plan.block_size
            tile_to_st[tile_idx] = st_id
            tile_crop_x[tile_idx] = x_idx * read_step_px
            tile_crop_y[tile_idx] = y_idx * read_step_px
            ordered_indices.append(tile_idx)
        plans.append(plan)

    return SupertileIndex(
        plans=plans,
        tile_to_st=tile_to_st,
        tile_crop_x=tile_crop_x,
        tile_crop_y=tile_crop_y,
        ordered_indices=np.array(ordered_indices, dtype=np.int64),
    )
