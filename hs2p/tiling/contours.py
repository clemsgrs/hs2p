from __future__ import annotations

import cv2
import numpy as np

from hs2p.tiling.result import ContourResult
from hs2p.wsi.reader import select_level


def _normalize_level_downsamples(
    level_downsamples: list[float] | list[tuple[float, float]],
) -> list[float]:
    normalized: list[float] = []
    for downsample in level_downsamples:
        if isinstance(downsample, tuple):
            normalized.append(float(downsample[0]))
        else:
            normalized.append(float(downsample))
    return normalized


def detect_contours(
    tissue_mask: np.ndarray,
    *,
    slide_dimensions: tuple[int, int],
    ref_tile_size_px: int = 16,
    requested_spacing_um: float = 0.5,
    a_t: int = 4,
    base_spacing_um: float | None = None,
    level_downsamples: list[float] | list[tuple[float, float]] | None = None,
    tolerance: float = 0.05,
) -> ContourResult:
    """Detect tissue contours and holes, scaled into level-0 pixel space."""
    if tissue_mask.max() == 0:
        return ContourResult(contours=[], holes=[], mask=tissue_mask)

    mask_h, mask_w = tissue_mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = slide_w / mask_w
    scale_y = slide_h / mask_h

    min_fg_area = 0
    if a_t > 0:
        if base_spacing_um is None or level_downsamples is None:
            raise ValueError(
                "base_spacing_um and level_downsamples are required when a_t > 0 "
                "so contour filtering can use actual slide geometry."
            )
        level_sel = select_level(
            requested_spacing_um=requested_spacing_um,
            level0_spacing_um=base_spacing_um,
            level_downsamples=[
                (float(downsample), float(downsample))
                for downsample in _normalize_level_downsamples(level_downsamples)
            ],
            tolerance=tolerance,
        )
        current_scale = level_sel.read_spacing_um / base_spacing_um
        ref_tile_mask_w = ref_tile_size_px * current_scale / scale_x
        ref_tile_mask_h = ref_tile_size_px * current_scale / scale_y
        scaled_ref_tile_area = int(ref_tile_mask_w * ref_tile_mask_h)
        min_fg_area = a_t * scaled_ref_tile_area

    if tissue_mask.ndim == 3:
        tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2GRAY)
    raw_contours, hierarchy = cv2.findContours(
        tissue_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )

    if hierarchy is None or len(raw_contours) == 0:
        return ContourResult(contours=[], holes=[], mask=tissue_mask)

    hierarchy = hierarchy[0]
    filtered_contours = []
    filtered_holes: list[list[np.ndarray]] = []
    for fg_idx, h in enumerate(hierarchy):
        if h[3] != -1:
            continue
        child_hole_indices = np.flatnonzero(hierarchy[:, 3] == fg_idx)
        area = cv2.contourArea(raw_contours[fg_idx])
        if child_hole_indices.size > 0:
            hole_areas = [cv2.contourArea(raw_contours[idx]) for idx in child_hole_indices]
            area -= float(np.sum(hole_areas))
        if area == 0 or area <= min_fg_area:
            continue

        contour_lv0 = raw_contours[fg_idx].copy().astype(np.float64)
        contour_lv0[:, 0, 0] *= scale_x
        contour_lv0[:, 0, 1] *= scale_y
        filtered_contours.append(contour_lv0.astype(np.int32))

        hole_contours_lv0 = []
        for hole_idx in child_hole_indices.tolist():
            hole_lv0 = raw_contours[hole_idx].copy().astype(np.float64)
            hole_lv0[:, 0, 0] *= scale_x
            hole_lv0[:, 0, 1] *= scale_y
            hole_contours_lv0.append(hole_lv0.astype(np.int32))
        filtered_holes.append(hole_contours_lv0)

    return ContourResult(
        contours=filtered_contours,
        holes=filtered_holes,
        mask=tissue_mask,
    )


__all__ = ["detect_contours"]
