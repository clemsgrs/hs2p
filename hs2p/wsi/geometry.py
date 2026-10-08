
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Pixel semantics controlling whether finer-than-source reads may upsample.
ContentKind = Literal["image", "label"]


@dataclass(frozen=True, kw_only=True)
class LevelSelection:
    level: int
    read_spacing_um: float
    is_within_tolerance: bool


def project_discrete_grid_origins(
    coordinates: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Project level-0 origin coordinates into a discrete target grid.

    Origin coordinates are rounded down so a top-left anchor stays
    within the same source pixel footprint after projection.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"coordinates must have shape (N, 2), got {coordinates.shape}"
        )
    projected = np.empty_like(coordinates, dtype=np.int64)
    projected[:, 0] = np.floor(
        coordinates[:, 0].astype(np.float64, copy=False) * float(scale_x)
    ).astype(np.int64)
    projected[:, 1] = np.floor(
        coordinates[:, 1].astype(np.float64, copy=False) * float(scale_y)
    ).astype(np.int64)
    return projected


def compute_level_spacings(
    *,
    level0_spacing_um: float,
    level_downsamples: list[tuple[float, float]],
) -> list[float]:
    return [float(level0_spacing_um) * float(ds_x) for ds_x, _ in level_downsamples]


def select_level_for_downsample(
    requested_downsample: float,
    level_downsamples: list[tuple[float, float]],
) -> int:
    if len(level_downsamples) == 0:
        raise ValueError("level_downsamples must not be empty")
    return int(
        np.argmin(
            [
                abs(float(downsample[0]) - requested_downsample)
                for downsample in level_downsamples
            ]
        )
    )


def select_level(
    *,
    requested_spacing_um: float,
    level0_spacing_um: float,
    level_downsamples: list[tuple[float, float]],
    tolerance: float = 0.05,
) -> LevelSelection:
    level_spacings = compute_level_spacings(
        level0_spacing_um=level0_spacing_um,
        level_downsamples=level_downsamples,
    )
    level = int(
        np.argmin(
            [abs(read_spacing - requested_spacing_um) for read_spacing in level_spacings]
        )
    )
    best_spacing = level_spacings[level]
    relative_error = abs(best_spacing - requested_spacing_um) / requested_spacing_um
    is_within_tolerance = relative_error <= tolerance

    if not is_within_tolerance:
        while level > 0 and best_spacing > requested_spacing_um:
            level -= 1
            best_spacing = level_spacings[level]
            relative_error = abs(best_spacing - requested_spacing_um) / requested_spacing_um
            is_within_tolerance = relative_error <= tolerance

    return LevelSelection(
        level=level,
        read_spacing_um=best_spacing,
        is_within_tolerance=is_within_tolerance,
    )


def select_level_for_spacing_read(
    *,
    requested_spacing_um: float,
    level0_spacing_um: float,
    level_downsamples: list[tuple[float, float]],
    tolerance: float,
    content_kind: ContentKind,
) -> LevelSelection:
    """Select a level and enforce the content-aware upsampling policy.

    Image pixels may only be read at native or coarser spacing; label pixels may
    be replicated because nearest-neighbour interpolation preserves their vocabulary.
    """
    if content_kind not in ("image", "label"):
        raise ValueError(
            f"unknown content_kind {content_kind!r}; expected 'image' or 'label'"
        )
    selection = select_level(
        requested_spacing_um=requested_spacing_um,
        level0_spacing_um=level0_spacing_um,
        level_downsamples=level_downsamples,
        tolerance=tolerance,
    )
    if (
        content_kind == "image"
        and requested_spacing_um < level0_spacing_um
        and not selection.is_within_tolerance
    ):
        raise ValueError(
            f"requested spacing {requested_spacing_um} µm/px is finer than finest "
            f"available spacing {level0_spacing_um} µm/px; "
            "image upsampling is forbidden"
        )
    return selection


@dataclass(frozen=True, kw_only=True)
class SpacingReadPlan:
    """How to read a region of ``target_size_px`` (at ``requested_spacing_um``).

    ``read_size_px`` is the size to read at ``level`` (its native ``read_spacing_um``)
    so that, after downscaling to ``target_size_px``, the result is at the requested
    spacing. When the chosen level is within tolerance the read size equals the target
    (no scaling — the tiny spacing difference is accepted, never upsampled).
    """

    level: int
    read_spacing_um: float
    is_within_tolerance: bool
    read_size_px: tuple[int, int]


def plan_spacing_read(
    *,
    requested_spacing_um: float,
    level0_spacing_um: float,
    level_downsamples: list[tuple[float, float]],
    target_size_px: tuple[int, int],
    tolerance: float,
    content_kind: ContentKind,
) -> SpacingReadPlan:
    """Resolve (level, read_size) for reading ``target_size_px`` at a spacing.

    The shared kernel behind both :meth:`hs2p.wsi.wsi.WSI.read_region_at_spacing`
    and the tiling pipeline's read-size derivation: pick the finest level ``<=`` the
    requested spacing (via :func:`select_level`), then size the read at that level to
    cover ``target_size_px`` after downscaling. Within tolerance ⇒ read the target
    size directly (treated as exact); otherwise scale up by
    ``requested_spacing_um / read_spacing_um``. ``content_kind`` is explicit:
    ``"image"`` forbids finer-than-source requests outside tolerance, while
    ``"label"`` allows nearest-neighbour replication by the caller.
    """
    sel = select_level_for_spacing_read(
        requested_spacing_um=requested_spacing_um,
        level0_spacing_um=level0_spacing_um,
        level_downsamples=level_downsamples,
        tolerance=tolerance,
        content_kind=content_kind,
    )
    target_w, target_h = int(target_size_px[0]), int(target_size_px[1])
    if sel.is_within_tolerance:
        read_w, read_h = target_w, target_h
    else:
        ratio = float(requested_spacing_um) / float(sel.read_spacing_um)
        read_w = round(target_w * ratio)
        read_h = round(target_h * ratio)
    return SpacingReadPlan(
        level=sel.level,
        read_spacing_um=sel.read_spacing_um,
        is_within_tolerance=sel.is_within_tolerance,
        read_size_px=(read_w, read_h),
    )


@dataclass(frozen=True, kw_only=True)
class TileStride:
    """Stride between neighbouring tile origins, at the read level and at level 0."""

    read_step_px: int
    step_px_lv0: int


def resolve_tile_stride(
    *, read_tile_size_px: int, tile_size_lv0: int, overlap: float
) -> TileStride:
    """Define the tile stride once, in read-level pixels, and project it to level 0.

    ``overlap`` is applied to the read-level tile size and rounded there; the level-0
    stride is that read-level stride scaled by the tile's level-0 footprint. Rounding
    the two independently (32 x 0.9 = 28.8 -> 29 at level 0, 16 x 0.9 = 14.4 -> 14 at
    the read level) puts tile origins at 14.5 read-level pixels while grouped reads crop
    every 14, so batched tiles drift from the pixels their coordinates name.
    """
    read_tile_size_px = int(read_tile_size_px)
    tile_size_lv0 = int(tile_size_lv0)
    if read_tile_size_px <= 0 or tile_size_lv0 <= 0:
        raise ValueError(
            "read_tile_size_px and tile_size_lv0 must be > 0, "
            f"got {read_tile_size_px} and {tile_size_lv0}"
        )
    read_step_px = max(1, round(read_tile_size_px * (1.0 - float(overlap))))
    step_px_lv0 = max(1, round(read_step_px * tile_size_lv0 / read_tile_size_px))
    return TileStride(read_step_px=read_step_px, step_px_lv0=step_px_lv0)


def tile_size_lv0_from_plan(plan: SpacingReadPlan, *, level0_spacing_um: float) -> int:
    """Level-0 footprint, in pixels, of a tile read with ``plan``.

    The read covers ``read_size_px`` pixels at ``read_spacing_um``. Within tolerance that
    is the target size at the level's native spacing, not at the requested one. Tiling
    and every tile-footprint estimate take the footprint from here so they agree.
    """
    return round(plan.read_size_px[0] * plan.read_spacing_um / float(level0_spacing_um))
