from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from hs2p.wsi.geometry import select_level
from hs2p.wsi.reader import SlideReader


def segment_tissue(
    *,
    reader: SlideReader,
    segment_params,
    seg_level: int,
) -> dict[str, np.ndarray]:
    img = np.asarray(reader.read_level(seg_level))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if segment_params.use_hsv:
        lower = np.array([90, 8, 103])
        upper = np.array([180, 255, 255])
        img_thresh = cv2.inRange(img_hsv, lower, upper)
    else:
        img_med = cv2.medianBlur(img_hsv[:, :, 1], segment_params.mthresh)
        if segment_params.use_otsu:
            _, img_thresh = cv2.threshold(
                img_med,
                0,
                segment_params.sthresh_up,
                cv2.THRESH_OTSU + cv2.THRESH_BINARY,
            )
        else:
            _, img_thresh = cv2.threshold(
                img_med,
                segment_params.sthresh,
                segment_params.sthresh_up,
                cv2.THRESH_BINARY,
            )

    if segment_params.close > 0:
        kernel = np.ones((segment_params.close, segment_params.close), np.uint8)
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

    return {"tissue": img_thresh}


def filter_contours(contours, hierarchy, filter_params):
    filtered = []
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []
    for cont_idx in hierarchy_1:
        cont = contours[cont_idx]
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        area = cv2.contourArea(cont)
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        area = area - np.array(hole_areas).sum()
        if area == 0:
            continue
        if area > filter_params.a_t:
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]
    hole_contours = []
    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]
        unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        unfiltered_holes = unfiltered_holes[: filter_params.max_n_holes]
        filtered_holes = []
        for hole in unfiltered_holes:
            if cv2.contourArea(hole) > filter_params.a_h:
                filtered_holes.append(hole)
        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours


def scale_contour_dim(contours, scale):
    return [np.array(cont * scale, dtype="int32") for cont in contours]


def scale_holes_dim(contours, scale):
    return [[np.array(hole * scale, dtype="int32") for hole in holes] for holes in contours]


def detect_contours(
    *,
    annotation_mask: dict[str, np.ndarray],
    annotation: str | None,
    target_spacing: float,
    tolerance: float,
    filter_params,
    spacings: list[float],
    level_downsamples: list[tuple[float, float]],
    seg_level: int,
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    spacing_selection = select_level(
        requested_spacing_um=target_spacing,
        level0_spacing_um=spacings[0],
        level_downsamples=level_downsamples,
        tolerance=tolerance,
    )
    current_scale = level_downsamples[spacing_selection.level]
    target_scale = level_downsamples[seg_level]
    scale = tuple(a / b for a, b in zip(target_scale, current_scale))
    ref_tile_size = (filter_params.ref_tile_size, filter_params.ref_tile_size)
    ref_tile_size_at_target_scale = tuple(a / b for a, b in zip(ref_tile_size, scale))
    scaled_ref_tile_area = int(ref_tile_size_at_target_scale[0] * ref_tile_size_at_target_scale[1])

    adjusted_filter_params = replace(
        filter_params,
        a_t=filter_params.a_t * scaled_ref_tile_area,
        a_h=filter_params.a_h * scaled_ref_tile_area,
    )

    mask = annotation_mask["tissue"] if annotation is None else annotation_mask[annotation]
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        return [], []

    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    foreground_contours, hole_contours = filter_contours(
        contours,
        hierarchy,
        adjusted_filter_params,
    )
    contours_level0 = scale_contour_dim(foreground_contours, target_scale)
    holes_level0 = scale_holes_dim(hole_contours, target_scale)
    return contours_level0, holes_level0

