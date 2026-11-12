from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import cv2
import numpy as np

from hs2p.configs.models import TILE_GRIDS, TileGrid
from hs2p.tiling.contours import _normalize_level_downsamples
from hs2p.tiling.coverage import _compute_tile_coverage
from hs2p.tiling.result import ContourResult, TileGeometry, TilingResult
from hs2p.wsi.geometry import (
    plan_spacing_read,
    project_discrete_grid_origins,
    resolve_tile_stride,
    tile_size_lv0_from_plan,
)


def canonicalize_tiling_result(tiles: TileGeometry) -> TileGeometry:
    """Deduplicate and sort tile coordinates in column-major order."""
    coords = np.column_stack((tiles.x, tiles.y))
    fracs = tiles.tissue_fractions
    if len(coords) > 1:
        _, unique_idx = np.unique(coords, axis=0, return_index=True)
        unique_idx.sort()
        coords = coords[unique_idx]
        fracs = fracs[unique_idx]
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        coords = coords[order]
        fracs = fracs[order]

    return replace(
        tiles,
        x=coords[:, 0],
        y=coords[:, 1],
        tissue_fractions=fracs,
        tile_index=np.arange(len(coords), dtype=np.int32),
    )


def resolve_base_spacing_um(result: TilingResult | TileGeometry) -> float:
    if result.base_spacing_um is None:
        raise ValueError("result is missing base_spacing_um metadata")
    return float(result.base_spacing_um)


def generate_tiles(
    slide_dimensions: tuple[int, int],
    contours: ContourResult,
    *,
    requested_tile_size_px: int = 256,
    requested_spacing_um: float = 0.5,
    base_spacing_um: float,
    level_downsamples: list[float] | list[tuple[float, float]],
    overlap: float = 0.0,
    min_tissue_fraction: float = 0.5,
    tolerance: float = 0.05,
    num_workers: int = 1,
    grid: TileGrid = "contour",
) -> TileGeometry:
    """Lay tiles over ``contours`` and keep those the tissue mask covers enough.

    ``grid`` picks where origins lie. ``"contour"`` anchors one grid at each contour's
    bounding box. ``"slide"`` lays one lattice over the whole slide: origins at every
    multiple of the stride from ``(0, 0)`` that falls inside the slide, kept when the
    contours' tissue covers at least ``min_tissue_fraction`` of the whole tile. A
    background-only tile is never kept, so a ``0.0`` threshold keeps every tile the tissue
    touches rather than the whole slide.
    """
    if grid not in TILE_GRIDS:
        raise ValueError(f"grid must be one of {list(TILE_GRIDS)}, got {grid!r}")
    normalized_downsamples = _normalize_level_downsamples(level_downsamples)
    level_sel = plan_spacing_read(
        requested_spacing_um=requested_spacing_um,
        level0_spacing_um=base_spacing_um,
        level_downsamples=[
            (float(downsample), float(downsample))
            for downsample in normalized_downsamples
        ],
        target_size_px=(requested_tile_size_px, requested_tile_size_px),
        tolerance=tolerance,
        content_kind="image",
    )
    read_tile_size_px = level_sel.read_size_px[0]
    tile_size_lv0 = tile_size_lv0_from_plan(level_sel, level0_spacing_um=base_spacing_um)
    step_lv0 = resolve_tile_stride(
        read_tile_size_px=read_tile_size_px, tile_size_lv0=tile_size_lv0, overlap=overlap
    ).step_px_lv0

    def _empty_result() -> TileGeometry:
        return TileGeometry(
            x=np.empty(0, dtype=np.int64),
            y=np.empty(0, dtype=np.int64),
            tissue_fractions=np.empty(0, dtype=np.float32),
            requested_tile_size_px=requested_tile_size_px,
            requested_spacing_um=requested_spacing_um,
            read_level=level_sel.level,
            read_tile_size_px=read_tile_size_px,
            read_spacing_um=level_sel.read_spacing_um,
            tile_size_lv0=tile_size_lv0,
            is_within_tolerance=level_sel.is_within_tolerance,
            tissue_mask=contours.mask,
            base_spacing_um=base_spacing_um,
            slide_dimensions=list(slide_dimensions),
            level_downsamples=list(normalized_downsamples),
            overlap=overlap,
            min_tissue_fraction=min_tissue_fraction,
            grid=grid,
        )

    if len(contours.contours) == 0:
        return _empty_result()

    def _process_contour(idx: int) -> tuple[np.ndarray, np.ndarray]:
        contour = contours.contours[idx]
        return _tiles_for_contour(
            contour=contour,
            contour_holes=contours.holes[idx],
            tissue_mask=contours.mask,
            slide_dimensions=slide_dimensions,
            tile_size_lv0=tile_size_lv0,
            step_lv0=step_lv0,
            min_tissue_fraction=min_tissue_fraction,
        )

    if grid == "slide":
        results = [
            _tiles_on_slide_grid(
                contours=contours,
                slide_dimensions=slide_dimensions,
                tile_size_lv0=tile_size_lv0,
                step_lv0=step_lv0,
                min_tissue_fraction=min_tissue_fraction,
                num_workers=num_workers,
            )
        ]
    elif num_workers > 1 and len(contours.contours) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_contour, range(len(contours.contours))))
    else:
        results = [_process_contour(i) for i in range(len(contours.contours))]

    all_coords = []
    all_fracs = []
    for coords, fracs in results:
        if len(coords) > 0:
            all_coords.append(coords)
            all_fracs.append(fracs)

    if not all_coords:
        return _empty_result()

    merged_coords = np.concatenate(all_coords, axis=0)
    merged_fracs = np.concatenate(all_fracs, axis=0)
    return canonicalize_tiling_result(
        TileGeometry(
            x=merged_coords[:, 0],
            y=merged_coords[:, 1],
            tissue_fractions=merged_fracs,
            requested_tile_size_px=requested_tile_size_px,
            requested_spacing_um=requested_spacing_um,
            read_level=level_sel.level,
            read_tile_size_px=read_tile_size_px,
            read_spacing_um=level_sel.read_spacing_um,
            tile_size_lv0=tile_size_lv0,
            is_within_tolerance=level_sel.is_within_tolerance,
            tissue_mask=contours.mask,
            base_spacing_um=base_spacing_um,
            slide_dimensions=list(slide_dimensions),
            level_downsamples=list(normalized_downsamples),
            overlap=overlap,
            min_tissue_fraction=min_tissue_fraction,
            grid=grid,
        )
    )


def _tiles_for_contour(
    *,
    contour: np.ndarray,
    contour_holes: list[np.ndarray],
    tissue_mask: np.ndarray,
    slide_dimensions: tuple[int, int],
    tile_size_lv0: int,
    step_lv0: int,
    min_tissue_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    slide_w, slide_h = slide_dimensions
    x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)
    x_start = max(x_cont, 0)
    y_start = max(y_cont, 0)
    x_end = min(x_cont + w_cont, slide_w)
    y_end = min(y_cont + h_cont, slide_h)

    if x_end <= x_start or y_end <= y_start:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)

    xs = np.arange(x_start, x_end, step_lv0, dtype=np.int64)
    ys = np.arange(y_start, y_end, step_lv0, dtype=np.int64)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    candidates = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    mask_h, mask_w = tissue_mask.shape[:2]
    mask_x0, mask_y0, mask_x1, mask_y1 = _contour_mask_box(
        contour, slide_dimensions=slide_dimensions, mask_dimensions=(mask_w, mask_h)
    )
    contour_mask = _build_contour_tissue_mask(
        contour=contour,
        contour_holes=contour_holes,
        tissue_mask=tissue_mask,
        slide_dimensions=slide_dimensions,
        mask_bounds=(mask_x0, mask_y0, mask_x1, mask_y1),
    )
    fractions = _compute_tile_coverage(
        candidates=candidates,
        binary_mask=contour_mask,
        tile_size_lv0=tile_size_lv0,
        slide_dimensions=slide_dimensions,
        mask_dimensions=(mask_w, mask_h),
        mask_origin=(mask_x0, mask_y0),
    )
    keep = fractions >= min_tissue_fraction
    return candidates[keep], fractions[keep]


def _contour_mask_box(
    contour: np.ndarray,
    *,
    slide_dimensions: tuple[int, int],
    mask_dimensions: tuple[int, int],
) -> tuple[int, int, int, int]:
    """The ``(x0, y0, x1, y1)`` box on the mask grid that holds every pixel ``contour``
    fills.

    Rounds on the full mask grid before any crop. The inclusive contour extrema bound every
    filled pixel, so pixels outside the box never need an integral.
    """
    slide_w, slide_h = slide_dimensions
    mask_w, mask_h = mask_dimensions
    scale_x, scale_y = mask_w / slide_w, mask_h / slide_h
    x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)
    return (
        min(mask_w, max(0, round(x_cont * scale_x))),
        min(mask_h, max(0, round(y_cont * scale_y))),
        min(mask_w, max(0, round((x_cont + w_cont - 1) * scale_x) + 1)),
        min(mask_h, max(0, round((y_cont + h_cont - 1) * scale_y) + 1)),
    )


def _tiles_on_slide_grid(
    *,
    contours: ContourResult,
    slide_dimensions: tuple[int, int],
    tile_size_lv0: int,
    step_lv0: int,
    min_tissue_fraction: float,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tiles on one lattice from the slide's top-left corner, measured only near tissue.

    Coverage is measured on the union of the contours' tissue, so a tile spanning two
    contours counts both. Only lattice cells whose footprint meets some contour's mask box
    are measured: every other cell covers no tissue and could never be kept, so the result
    is the one measuring the whole lattice would give, at a cost that scales with tissue.
    """
    slide_w, slide_h = slide_dimensions
    tissue_mask = contours.mask
    mask_h, mask_w = tissue_mask.shape[:2]
    scale_x, scale_y = mask_w / slide_w, mask_h / slide_h
    boxes = [
        _contour_mask_box(
            contour, slide_dimensions=slide_dimensions, mask_dimensions=(mask_w, mask_h)
        )
        for contour in contours.contours
    ]

    def _draw(idx: int) -> np.ndarray:
        return _build_contour_tissue_mask(
            contour=contours.contours[idx],
            contour_holes=contours.holes[idx],
            tissue_mask=tissue_mask,
            slide_dimensions=slide_dimensions,
            mask_bounds=boxes[idx],
        )

    # The lattice one axis at a time: level-0 origins, and where each lands on the mask
    # grid, projected and sized as the coverage measurement does it.
    xs = np.arange(0, slide_w, step_lv0, dtype=np.int64)
    ys = np.arange(0, slide_h, step_lv0, dtype=np.int64)
    mask_xs = project_discrete_grid_origins(
        np.column_stack((xs, np.zeros_like(xs))), scale_x=scale_x, scale_y=scale_y
    )[:, 0]
    mask_ys = project_discrete_grid_origins(
        np.column_stack((np.zeros_like(ys), ys)), scale_x=scale_x, scale_y=scale_y
    )[:, 1]
    tile_w_mask = max(1, round(tile_size_lv0 * scale_x))
    tile_h_mask = max(1, round(tile_size_lv0 * scale_y))
    union = np.zeros((mask_h, mask_w), dtype=np.uint8)

    def _measure(box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        x0, y0, x1, y1 = box
        cols = np.flatnonzero((mask_xs < x1) & (mask_xs + tile_w_mask > x0))
        rows = np.flatnonzero((mask_ys < y1) & (mask_ys + tile_h_mask > y0))
        if cols.size == 0 or rows.size == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs[cols], ys[rows], indexing="ij")
        candidates = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        # The union crop that holds every measured footprint whole.
        crop_x0, crop_y0 = int(mask_xs[cols[0]]), int(mask_ys[rows[0]])
        crop_x1 = min(mask_w, int(mask_xs[cols[-1]]) + tile_w_mask)
        crop_y1 = min(mask_h, int(mask_ys[rows[-1]]) + tile_h_mask)
        fractions = _compute_tile_coverage(
            candidates=candidates,
            binary_mask=union[crop_y0:crop_y1, crop_x0:crop_x1],
            tile_size_lv0=tile_size_lv0,
            slide_dimensions=slide_dimensions,
            mask_dimensions=(mask_w, mask_h),
            mask_origin=(crop_x0, crop_y0),
        )
        keep = (fractions > 0) & (fractions >= min_tissue_fraction)
        return candidates[keep], fractions[keep]

    indices = range(len(contours.contours))
    if num_workers > 1 and len(boxes) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            drawn = list(executor.map(_draw, indices))
            for (x0, y0, x1, y1), contour_mask in zip(boxes, drawn):
                union[y0:y1, x0:x1] |= contour_mask
            measured = list(executor.map(_measure, boxes))
    else:
        for idx in indices:
            x0, y0, x1, y1 = boxes[idx]
            union[y0:y1, x0:x1] |= _draw(idx)
        measured = [_measure(box) for box in boxes]
    return (
        np.concatenate([coords for coords, _ in measured], axis=0),
        np.concatenate([fracs for _, fracs in measured], axis=0),
    )


def _build_contour_tissue_mask(
    contour: np.ndarray,
    contour_holes: list[np.ndarray],
    tissue_mask: np.ndarray,
    slide_dimensions: tuple[int, int],
    *,
    mask_bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    mask_h, mask_w = tissue_mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = mask_w / slide_w
    scale_y = mask_h / slide_h

    x0, y0, x1, y1 = mask_bounds if mask_bounds is not None else (0, 0, mask_w, mask_h)
    contour_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    if contour_mask.size == 0:
        return contour_mask
    contour_mask_scaled = contour.astype(np.float64)
    contour_mask_scaled[:, 0, 0] *= scale_x
    contour_mask_scaled[:, 0, 1] *= scale_y
    contour_mask_scaled = np.round(contour_mask_scaled).astype(np.int32)
    cv2.drawContours(
        contour_mask, [contour_mask_scaled], -1, 1, thickness=-1, offset=(-x0, -y0)
    )

    if contour_holes:
        holes_scaled = []
        for hole in contour_holes:
            hole_scaled = hole.astype(np.float64)
            hole_scaled[:, 0, 0] *= scale_x
            hole_scaled[:, 0, 1] *= scale_y
            holes_scaled.append(np.round(hole_scaled).astype(np.int32))
        # A hole contour runs through the foreground pixels bordering the hole, so a
        # filled draw clears that ring of tissue along with the hole. The ring is the
        # contour itself: redraw it as a one-pixel line. The source-mask intersection
        # below keeps everything honest, while the fill still excludes any island of
        # tissue nested inside the hole, which is its own contour.
        cv2.drawContours(
            contour_mask, holes_scaled, -1, 0, thickness=-1, offset=(-x0, -y0)
        )
        cv2.drawContours(
            contour_mask, holes_scaled, -1, 1, thickness=1, offset=(-x0, -y0)
        )

    return contour_mask * (tissue_mask[y0:y1, x0:x1] > 0).astype(np.uint8)


__all__ = [
    "canonicalize_tiling_result",
    "generate_tiles",
    "resolve_base_spacing_um",
]
