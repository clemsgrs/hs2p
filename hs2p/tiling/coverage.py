"""Integral-image tile coverage computation."""

import cv2
import numpy as np

from hs2p.wsi.geometry import project_discrete_grid_origins


def compute_tile_coverage(
    candidates: np.ndarray,
    binary_mask: np.ndarray,
    tile_size_lv0: int,
    slide_dimensions: tuple[int, int],
) -> np.ndarray:
    """Compute the tissue-covered fraction for each candidate tile.

    Uses an integral image for O(1) per-tile summation.

    Args:
        candidates: (N, 2) int64 array of (x, y) tile origins in level-0 pixel space.
        binary_mask: 2-D uint8 mask in mask/segmentation space. Any non-zero pixel
            counts as tissue.
        tile_size_lv0: tile side length in level-0 pixels (square tiles assumed).
        slide_dimensions: (width, height) of the slide in level-0 pixels.

    Returns:
        (N,) float32 array with values in [0, 1].
    """
    mask_h, mask_w = binary_mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = mask_w / slide_w
    scale_y = mask_h / slide_h

    binary = (binary_mask > 0).astype(np.float64)
    integral = cv2.integral(binary)

    tile_w_mask = max(1, round(tile_size_lv0 * scale_x))
    tile_h_mask = max(1, round(tile_size_lv0 * scale_y))
    tile_area = tile_w_mask * tile_h_mask

    mask_origins = project_discrete_grid_origins(
        candidates,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    x1 = np.clip(mask_origins[:, 0], 0, mask_w)
    y1 = np.clip(mask_origins[:, 1], 0, mask_h)
    x2 = np.clip(mask_origins[:, 0] + tile_w_mask, 0, mask_w)
    y2 = np.clip(mask_origins[:, 1] + tile_h_mask, 0, mask_h)

    tissue_sum = (
        integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    )
    return np.clip((tissue_sum / tile_area).astype(np.float32), 0.0, 1.0)


def summarize_annotation_coverage(
    *,
    slide,
    resolved_masks,
    min_coverage: dict[str, float | None] | None,
    requested_tile_size_px: int,
    requested_spacing_um: float,
    overlap: float = 0.0,
) -> dict[str, dict[str, float | int | None]]:
    """Per-class annotation coverage summary at the resolved ``seg_downsample``.

    Returns ``{class_name: {"area_mm2", "frac", "est_tiles"}}`` where:

    - ``area_mm2`` — class foreground area, from the seg-space pixel count scaled by the
      seg-level spacing (``(seg_spacing_um / 1000) ** 2`` mm² per pixel).
    - ``frac`` — class area divided by the total area of the classes of interest, i.e. the
      class's share among the classes given a ``min_coverage`` threshold (sums to 1 across
      those classes). ``None`` for a class without a threshold. No label name is special:
      ``min_coverage`` is the only signal for "which classes matter", so a declared-but-
      unthresholded label (e.g. the value reserved for unannotated pixels) is excluded from
      both the numerator set and the denominator.
    - ``est_tiles`` — number of non-overlapping tile footprints whose class coverage is at
      least ``min_coverage[class]`` (``None`` when no threshold is given). This is an
      *estimate*: it reuses :func:`compute_tile_coverage` over a regular level-0 grid and
      deliberately ignores tissue filtering and tile overlap.

    ``area_mm2`` is reported for every declared class; ``frac``/``est_tiles`` only for the
    thresholded classes. Reuses hs2p's existing coverage primitive (the controllable
    ``seg_downsample`` is the precision/speed knob) rather than re-scanning the mask.
    """
    seg_spacing_um = float(resolved_masks.seg_spacing_um)
    mm2_per_pixel = (seg_spacing_um / 1000.0) ** 2
    areas_px = {
        name: float(np.count_nonzero(mask)) for name, mask in resolved_masks.masks.items()
    }
    total_of_interest = sum(
        area
        for name, area in areas_px.items()
        if min_coverage is not None and min_coverage.get(name) is not None
    )

    base_spacing_um = float(slide.spacing)
    tile_size_lv0 = max(1, round(requested_tile_size_px * requested_spacing_um / base_spacing_um))
    step = max(1, round(tile_size_lv0 * (1.0 - overlap)))
    slide_w, slide_h = int(slide.dimensions[0]), int(slide.dimensions[1])
    xs = np.arange(0, max(1, slide_w), step, dtype=np.int64)
    ys = np.arange(0, max(1, slide_h), step, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    candidates = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.int64)

    summary: dict[str, dict[str, float | int | None]] = {}
    for name, binary_mask in resolved_masks.masks.items():
        area_px = areas_px[name]
        threshold = None if min_coverage is None else min_coverage.get(name)
        if threshold is None:
            est_tiles: int | None = None
            frac: float | None = None
        else:
            coverage = compute_tile_coverage(
                candidates, binary_mask, tile_size_lv0, (slide_w, slide_h)
            )
            est_tiles = int(np.count_nonzero(coverage >= float(threshold)))
            frac = (area_px / total_of_interest) if total_of_interest > 0 else 0.0
        summary[name] = {
            "area_mm2": area_px * mm2_per_pixel,
            "frac": frac,
            "est_tiles": est_tiles,
        }
    return summary


__all__ = ["compute_tile_coverage", "summarize_annotation_coverage"]
