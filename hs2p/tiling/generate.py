from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import cv2
import numpy as np

from hs2p.tiling.contours import _normalize_level_downsamples
from hs2p.tiling.coverage import compute_tile_coverage
from hs2p.tiling.result import ContourResult, TileGeometry, TilingResult
from hs2p.wsi.reader import select_level


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
) -> TileGeometry:
    normalized_downsamples = _normalize_level_downsamples(level_downsamples)
    if requested_spacing_um < base_spacing_um:
        relative_diff = abs(base_spacing_um - requested_spacing_um) / requested_spacing_um
        if relative_diff > tolerance:
            raise ValueError(
                f"Desired spacing ({requested_spacing_um}) is smaller than the "
                f"whole-slide image starting spacing ({base_spacing_um}) and does not "
                f"fall within tolerance ({tolerance:.0%})"
            )

    level_sel = select_level(
        requested_spacing_um=requested_spacing_um,
        level0_spacing_um=base_spacing_um,
        level_downsamples=[
            (float(downsample), float(downsample))
            for downsample in normalized_downsamples
        ],
        tolerance=tolerance,
    )
    if level_sel.is_within_tolerance:
        read_tile_size_px = requested_tile_size_px
    else:
        read_tile_size_px = round(
            requested_tile_size_px * requested_spacing_um / level_sel.read_spacing_um
        )
    tile_size_lv0 = round(
        read_tile_size_px * level_sel.read_spacing_um / base_spacing_um
    )
    step_lv0 = max(1, round(tile_size_lv0 * (1.0 - overlap)))

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

    if num_workers > 1 and len(contours.contours) > 1:
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
        )
    )


def _tiles_for_contour(
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
    contour_mask = _build_contour_tissue_mask(
        contour=contour,
        contour_holes=contour_holes,
        tissue_mask=tissue_mask,
        slide_dimensions=slide_dimensions,
    )
    fractions = compute_tile_coverage(
        candidates,
        contour_mask,
        tile_size_lv0,
        slide_dimensions,
    )
    keep = fractions >= min_tissue_fraction
    return candidates[keep], fractions[keep]


def _build_contour_tissue_mask(
    contour: np.ndarray,
    contour_holes: list[np.ndarray],
    tissue_mask: np.ndarray,
    slide_dimensions: tuple[int, int],
) -> np.ndarray:
    mask_h, mask_w = tissue_mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = mask_w / slide_w
    scale_y = mask_h / slide_h

    contour_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    contour_mask_scaled = contour.copy().astype(np.float64)
    contour_mask_scaled[:, 0, 0] *= scale_x
    contour_mask_scaled[:, 0, 1] *= scale_y
    contour_mask_scaled = np.round(contour_mask_scaled).astype(np.int32)
    cv2.drawContours(contour_mask, [contour_mask_scaled], -1, 1, thickness=-1)

    if contour_holes:
        holes_scaled = []
        for hole in contour_holes:
            hole_scaled = hole.copy().astype(np.float64)
            hole_scaled[:, 0, 0] *= scale_x
            hole_scaled[:, 0, 1] *= scale_y
            holes_scaled.append(np.round(hole_scaled).astype(np.int32))
        cv2.drawContours(contour_mask, holes_scaled, -1, 0, thickness=-1)

    return contour_mask * (tissue_mask > 0).astype(np.uint8)


__all__ = [
    "canonicalize_tiling_result",
    "generate_tiles",
    "resolve_base_spacing_um",
]
