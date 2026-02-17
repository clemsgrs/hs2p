from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

import wholeslidedata as wsd


HSV_LOWER = np.array([90, 8, 103], dtype=np.uint8)
HSV_UPPER = np.array([180, 255, 255], dtype=np.uint8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a tissue mask using HSV thresholding from a whole-slide image "
            "and save it as a pyramidal TIFF."
        )
    )
    parser.add_argument(
        "--wsi",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Input whole-slide image path(s) or glob pattern(s). "
            "Examples: /path/slide.tif or '/path/slides/*.tif'"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pyramidal TIFF path for single-slide mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for multi-slide mode. Each output is written as "
            "{output-dir}/{slide_stem}.tif"
        ),
    )
    parser.add_argument(
        "--spacing",
        type=float,
        required=True,
        help="Requested level-0 spacing for the output mask in microns per pixel.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        required=True,
        help="Tolerance deciding how much a natural spacing can deviate from target spacing when selecting the best level for reading (expressed as a fraction of the target spacing, e.g. 0.1 for 10%%).",
    )
    parser.add_argument(
        "--downsample-per-level",
        type=float,
        default=2.0,
        help="Pyramid downsample factor between successive levels (must be > 1).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="asap",
        help="WholeSlideData backend for reading the WSI.",
    )
    parser.add_argument(
        "--spacing-at-level-0",
        type=float,
        default=None,
        help="Optional override for the input WSI level-0 spacing in microns per pixel.",
    )
    parser.add_argument(
        "--min-component-area-um2",
        type=float,
        default=2000.0,
        help=(
            "Remove tissue connected components smaller than this area in square microns "
            "(default: 2000um2 ; 0 to disable)."
        ),
    )
    parser.add_argument(
        "--min-hole-area-um2",
        type=float,
        default=5000.0,
        help=(
            "Fill holes inside tissue up to this area in square microns "
            "(default: 5000um2 ; 0 to disable)."
        ),
    )
    parser.add_argument(
        "--open-radius-um",
        type=float,
        default=1.0,
        help="Morphological opening radius in microns (default: 1um ; 0 to disable).",
    )
    parser.add_argument(
        "--gaussian-sigma-um",
        type=float,
        default=1,
        help="Optional Gaussian blur sigma in microns before HSV thresholding (default: 1um ; 0 to disable).",
    )
    parser.add_argument(
        "--close-radius-um",
        type=float,
        default=12.0,
        help="Morphological closing radius in microns (default: 12um ; 0 to disable).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="deflate",
        help="TIFF compression codec for output mask.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size for writing TIFF levels when dimensions allow tiling.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Stop pyramid generation before creating a level with min(height, width) < min-size.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-level pyramid details and output path.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache-based skipping and force recomputation for all slides.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of worker processes for slide processing (default: min(4, CPU count)).",
    )
    parser.add_argument(
        "--disable-coarse-roi-shortcut",
        action="store_true",
        help=(
            "Disable coarse-to-fine ROI shortcut and force full-frame loading at target spacing. "
            "By default, the shortcut is enabled."
        ),
    )
    parser.add_argument(
        "--coarse-spacing",
        type=float,
        default=4.0,
        help="Spacing used for coarse ROI pre-pass in microns per pixel (default: 4.0).",
    )
    parser.add_argument(
        "--coarse-roi-margin-um",
        type=float,
        default=64.0,
        help="ROI expansion margin in microns in target mask coordinates (default: 64).",
    )
    parser.add_argument(
        "--processing-tile-size",
        type=int,
        default=2048,
        help="Tile size (pixels) for high-resolution ROI processing (default: 2048).",
    )
    return parser.parse_args()


def get_downsamples(wsi: wsd.WholeSlideImage) -> list[tuple[float, float]]:
    """
    Calculate the downsample factors for each level of the image pyramid.

    This method computes the downsample factors for each level in the image
    pyramid relative to the base level (level 0). The downsample factor for
    each level is represented as a tuple of two values, corresponding to the
    downsampling in the width and height dimensions.

    Returns:
        list of tuple: A list of tuples where each tuple contains two float
        values representing the downsample factors (width_factor, height_factor)
        for each level relative to the base level.
    """
    level_downsamples = []
    dim_0 = wsi.shapes[0]
    for dim in wsi.shapes:
        level_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples.append(level_downsample)
    return level_downsamples


def get_best_level_for_downsample_custom(downsamples: list[tuple[int, int]], target_downsample: int):
    """
    Determines the best level for a given downsample factor based on the available
    level downsample values.

    Args:
        downsample (float): Target downsample factor.

    Returns:
        int: Index of the best matching level for the given downsample factor.
    """
    level = int(np.argmin([abs(x - target_downsample) for x, _ in downsamples]))
    return level


def get_best_level_for_spacing(
    wsi: wsd.WholeSlideImage, target_spacing: float, tolerance: float
):
    """
    Determines the best level in a multi-resolution image pyramid for a given target spacing.

    Ensures that the spacing of the returned level is either within the specified tolerance of the target
    spacing or smaller than the target spacing to avoid upsampling.

    Args:
        target_spacing (float): Desired spacing.
        tolerance (float, optional): Tolerance for matching the spacing, deciding how much
            spacing can deviate from those specified in the slide metadata.

    Returns:
        level (int): Index of the best matching level in the image pyramid.
    """
    spacing_at_0 = wsi.spacings[0]
    target_downsample = target_spacing / spacing_at_0
    level_downsamples = get_downsamples(wsi)
    level = get_best_level_for_downsample_custom(level_downsamples, target_downsample)
    level_spacing = wsi.spacings[level]

    # check if the level_spacing is within the tolerance of the target_spacing
    is_within_tolerance = False
    if abs(level_spacing - target_spacing) / target_spacing <= tolerance:
        is_within_tolerance = True
        return level, is_within_tolerance

    # otherwise, look for a spacing smaller than or equal to the target_spacing
    else:
        while level > 0 and level_spacing > target_spacing:
            level -= 1
            level_spacing = wsi.spacings[level]
            if abs(level_spacing - target_spacing) / target_spacing <= tolerance:
                is_within_tolerance = True
                break

    assert (
        level_spacing <= target_spacing
        or abs(level_spacing - target_spacing) / target_spacing <= tolerance
    ), f"Unable to find a spacing less than or equal to the target spacing ({target_spacing}) or within {int(tolerance * 100)}% of the target spacing."
    return level, is_within_tolerance


def load_wsi_at_spacing(
    *,
    wsi_path: Path,
    target_spacing: float,
    tolerance: float,
    backend: str,
    spacing_at_level_0: float | None = None,
) -> tuple[np.ndarray, float]:

    wsi = wsd.WholeSlideImage(wsi_path, backend=backend)
    base_spacings = list(wsi.spacings)

    if spacing_at_level_0 is None:
        spacings = base_spacings
    else:
        spacings = [spacing_at_level_0 * sp / base_spacings[0] for sp in base_spacings]

    level, is_within_tolerance = get_best_level_for_spacing(wsi, target_spacing, tolerance)
    wsi_arr = wsi.get_slide(spacing=spacings[level])

    if not is_within_tolerance:
        # means the selected level's spacing is smaller than the target_spacing
        # find power of 2 downsample factor to apply to the selected level to get as close as possible to the target_spacing
        # we want the end spacing to be within the tolerance of the target_spacing
        power = int(np.ceil(np.log2(target_spacing / spacings[level])))
        final_spacing = spacings[level] * (2**power)
        if abs(final_spacing - target_spacing) / target_spacing > tolerance:
            raise ValueError(
                f"Unable to achieve target spacing within tolerance after downsampling. Closest achievable spacing is {final_spacing:.2f} mpp, which is {abs(final_spacing - target_spacing) / target_spacing:.2%} away from the target spacing. Consider increasing the tolerance or adjusting the target spacing."
            )
        wsi_arr = cv2.resize(
            wsi_arr,
            dsize=None,
            fx=final_spacing / spacings[level],
            fy=final_spacing / spacings[level],
            interpolation=cv2.INTER_AREA,
        )
        effective_spacing = final_spacing
    else:
        effective_spacing = spacings[level]

    return wsi_arr, effective_spacing


def _get_spacings(
    *,
    wsi: wsd.WholeSlideImage,
    spacing_at_level_0: float | None,
) -> list[float]:
    base_spacings = list(wsi.spacings)
    if spacing_at_level_0 is None:
        return base_spacings
    return [spacing_at_level_0 * sp / base_spacings[0] for sp in base_spacings]


def _resolve_spacing_plan(
    *,
    wsi: wsd.WholeSlideImage,
    target_spacing: float,
    tolerance: float,
    spacing_at_level_0: float | None,
) -> tuple[int, float, float, bool]:
    spacings = _get_spacings(wsi=wsi, spacing_at_level_0=spacing_at_level_0)
    target_downsample = target_spacing / spacings[0]
    level_downsamples = get_downsamples(wsi)
    level = get_best_level_for_downsample_custom(level_downsamples, target_downsample)
    level_spacing = spacings[level]

    if abs(level_spacing - target_spacing) / target_spacing <= tolerance:
        return level, level_spacing, level_spacing, False

    while level > 0 and level_spacing > target_spacing:
        level -= 1
        level_spacing = spacings[level]
        if abs(level_spacing - target_spacing) / target_spacing <= tolerance:
            return level, level_spacing, level_spacing, False

    power = int(np.ceil(np.log2(target_spacing / level_spacing)))
    effective_spacing = level_spacing * (2**power)
    if abs(effective_spacing - target_spacing) / target_spacing > tolerance:
        raise ValueError(
            f"Unable to achieve target spacing within tolerance after downsampling. Closest achievable spacing is {effective_spacing:.2f} mpp, which is {abs(effective_spacing - target_spacing) / target_spacing:.2%} away from the target spacing. Consider increasing the tolerance or adjusting the target spacing."
        )

    return level, level_spacing, effective_spacing, True


def segment_tissue_hsv(
    wsi_arr: np.ndarray,
    lower: np.ndarray = HSV_LOWER,
    upper: np.ndarray = HSV_UPPER,
    gaussian_sigma_px: float = 0.0,
) -> np.ndarray:
    if gaussian_sigma_px > 0:
        wsi_arr = cv2.GaussianBlur(wsi_arr, (0, 0), sigmaX=gaussian_sigma_px, sigmaY=gaussian_sigma_px)

    img_hsv = cv2.cvtColor(wsi_arr, cv2.COLOR_RGB2HSV)
    mask = (cv2.inRange(img_hsv, lower, upper) > 0).astype(np.uint8)

    return mask


def _area_um2_to_px(area_um2: float, spacing_um_per_px: float) -> int:
    if area_um2 <= 0:
        return 0
    px_area = area_um2 / (spacing_um_per_px**2)
    return max(1, int(round(px_area)))


def _radius_um_to_px(radius_um: float, spacing_um_per_px: float) -> int:
    if radius_um <= 0:
        return 0
    px_radius = radius_um / spacing_um_per_px
    return max(1, int(round(px_radius)))


def _sigma_um_to_px(sigma_um: float, spacing_um_per_px: float) -> float:
    if sigma_um <= 0:
        return 0.0
    return float(sigma_um / spacing_um_per_px)


def _remove_small_tissue_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    keep = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            keep[labels == label] = 1
    return keep


def _fill_small_holes(mask: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    if max_hole_area_px <= 0:
        return mask

    h, w = mask.shape[:2]
    inv = (1 - mask).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num_labels <= 1:
        return mask

    filled = mask.copy().astype(np.uint8)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        touches_border = (x == 0) or (y == 0) or (x + bw >= w) or (y + bh >= h)
        if touches_border:
            continue
        if area <= max_hole_area_px:
            filled[labels == label] = 1

    return filled


def _extract_roi_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w, y + h))
    return boxes


def _boxes_intersect(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> bool:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _coarse_boxes_to_target_boxes(
    *,
    coarse_boxes: list[tuple[int, int, int, int]],
    coarse_shape: tuple[int, int],
    target_shape: tuple[int, int],
    margin_px: int,
) -> list[tuple[int, int, int, int]]:
    coarse_h, coarse_w = coarse_shape
    target_h, target_w = target_shape
    if coarse_h == 0 or coarse_w == 0:
        return []

    scale_x = target_w / float(coarse_w)
    scale_y = target_h / float(coarse_h)
    target_boxes: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in coarse_boxes:
        tx0 = max(0, int(np.floor(x0 * scale_x)) - margin_px)
        ty0 = max(0, int(np.floor(y0 * scale_y)) - margin_px)
        tx1 = min(target_w, int(np.ceil(x1 * scale_x)) + margin_px)
        ty1 = min(target_h, int(np.ceil(y1 * scale_y)) + margin_px)
        if tx1 > tx0 and ty1 > ty0:
            target_boxes.append((tx0, ty0, tx1, ty1))
    return target_boxes


def _compute_level0_mask_with_coarse_roi_shortcut(
    *,
    wsi_path: Path,
    target_spacing: float,
    tolerance: float,
    backend: str,
    spacing_at_level_0: float | None,
    coarse_spacing: float,
    coarse_roi_margin_um: float,
    processing_tile_size: int,
    min_component_area_um2: float,
    min_hole_area_um2: float,
    gaussian_sigma_um: float,
    open_radius_um: float,
    close_radius_um: float,
) -> tuple[np.ndarray, float, str]:
    wsi = wsd.WholeSlideImage(wsi_path, backend=backend)
    spacing_at_l0 = _get_spacings(wsi=wsi, spacing_at_level_0=spacing_at_level_0)[0]
    level, read_spacing, effective_spacing, needs_resampling = _resolve_spacing_plan(
        wsi=wsi,
        target_spacing=target_spacing,
        tolerance=tolerance,
        spacing_at_level_0=spacing_at_level_0,
    )

    coarse_arr, coarse_effective_spacing = load_wsi_at_spacing(
        wsi_path=wsi_path,
        target_spacing=coarse_spacing,
        tolerance=tolerance,
        backend=backend,
        spacing_at_level_0=spacing_at_level_0,
    )
    coarse_sigma_px = _sigma_um_to_px(gaussian_sigma_um, coarse_effective_spacing)
    coarse_mask = segment_tissue_hsv(wsi_arr=coarse_arr, gaussian_sigma_px=coarse_sigma_px)
    coarse_mask = postprocess_mask(
        mask=coarse_mask,
        spacing_um_per_px=coarse_effective_spacing,
        min_component_area_um2=min_component_area_um2,
        min_hole_area_um2=min_hole_area_um2,
        open_radius_um=open_radius_um,
        close_radius_um=close_radius_um,
    )

    level_w, level_h = wsi.shapes[level]
    spacing_ratio = read_spacing / effective_spacing
    target_w = max(1, int(round(level_w * spacing_ratio)))
    target_h = max(1, int(round(level_h * spacing_ratio)))
    target_shape = (target_h, target_w)
    margin_px = _radius_um_to_px(coarse_roi_margin_um, effective_spacing)
    target_boxes = _coarse_boxes_to_target_boxes(
        coarse_boxes=_extract_roi_boxes(coarse_mask),
        coarse_shape=coarse_mask.shape[:2],
        target_shape=target_shape,
        margin_px=margin_px,
    )

    tmp_file = tempfile.NamedTemporaryFile(prefix="hs2p-mask-", suffix=".mmap", delete=False)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    try:
        mask_l0 = np.memmap(tmp_path, dtype=np.uint8, mode="w+", shape=target_shape)
        mask_l0[:] = 0

        sigma_px = _sigma_um_to_px(gaussian_sigma_um, effective_spacing)
        halo_px = max(
            _radius_um_to_px(open_radius_um, effective_spacing),
            _radius_um_to_px(close_radius_um, effective_spacing),
            int(np.ceil(3.0 * sigma_px)),
        )

        for y0 in range(0, target_shape[0], processing_tile_size):
            y1 = min(target_shape[0], y0 + processing_tile_size)
            for x0 in range(0, target_shape[1], processing_tile_size):
                x1 = min(target_shape[1], x0 + processing_tile_size)

                rx0 = max(0, x0 - halo_px)
                ry0 = max(0, y0 - halo_px)
                rx1 = min(target_shape[1], x1 + halo_px)
                ry1 = min(target_shape[0], y1 + halo_px)

                rx0_l0 = int(round(rx0 * (effective_spacing / spacing_at_l0)))
                ry0_l0 = int(round(ry0 * (effective_spacing / spacing_at_l0)))

                tile = wsi.get_patch(
                    rx0_l0,
                    ry0_l0,
                    rx1 - rx0,
                    ry1 - ry0,
                    spacing=effective_spacing,
                    center=False,
                )
                tile_mask = segment_tissue_hsv(wsi_arr=tile, gaussian_sigma_px=sigma_px)

                crop_x0 = x0 - rx0
                crop_y0 = y0 - ry0
                crop_x1 = crop_x0 + (x1 - x0)
                crop_y1 = crop_y0 + (y1 - y0)
                mask_l0[y0:y1, x0:x1] = tile_mask[crop_y0:crop_y1, crop_x0:crop_x1]

        mask_l0 = postprocess_mask(
            mask=np.asarray(mask_l0),
            spacing_um_per_px=effective_spacing,
            min_component_area_um2=min_component_area_um2,
            min_hole_area_um2=min_hole_area_um2,
            open_radius_um=open_radius_um,
            close_radius_um=close_radius_um,
        )

        mode = "coarse-roi-resampled" if needs_resampling else "coarse-roi"
        return np.asarray(mask_l0), effective_spacing, mode
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def postprocess_mask(
    *,
    mask: np.ndarray,
    spacing_um_per_px: float,
    min_component_area_um2: float = 0.0,
    min_hole_area_um2: float = 0.0,
    open_radius_um: float = 0.0,
    close_radius_um: float = 0.0,
) -> np.ndarray:
    processed = mask.astype(np.uint8)

    min_component_area_px = _area_um2_to_px(min_component_area_um2, spacing_um_per_px)
    max_hole_area_px = _area_um2_to_px(min_hole_area_um2, spacing_um_per_px)
    open_radius_px = _radius_um_to_px(open_radius_um, spacing_um_per_px)
    close_radius_px = _radius_um_to_px(close_radius_um, spacing_um_per_px)

    processed = _remove_small_tissue_components(processed, min_component_area_px)
    processed = _fill_small_holes(processed, max_hole_area_px)

    if close_radius_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * close_radius_px + 1, 2 * close_radius_px + 1),
        )
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    if open_radius_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * open_radius_px + 1, 2 * open_radius_px + 1),
        )
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

    return (processed > 0).astype(np.uint8)


def build_mask_pyramid(
    level0_mask: np.ndarray,
    downsample_per_level: float,
    min_size: int,
) -> list[np.ndarray]:
    if downsample_per_level <= 1:
        raise ValueError("downsample_per_level must be > 1")
    if min_size < 1:
        raise ValueError("min_size must be >= 1")

    levels = [level0_mask.astype(np.uint8)]
    while True:
        previous = levels[-1]
        prev_h, prev_w = previous.shape[:2]
        next_w = max(1, int(round(prev_w / downsample_per_level)))
        next_h = max(1, int(round(prev_h / downsample_per_level)))
        if min(next_h, next_w) < min_size:
            break
        next_level = cv2.resize(previous, (next_w, next_h), interpolation=cv2.INTER_NEAREST)
        levels.append(next_level)

    return levels


def _spacing_to_pixels_per_cm(spacing_um_per_px: float) -> float:
    return 10000.0 / spacing_um_per_px


def resolve_wsi_paths(wsi_inputs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw_input in wsi_inputs:
        matches = [Path(match) for match in glob.glob(raw_input, recursive=True)]
        if not matches:
            matches = [Path(raw_input)]

        for candidate in sorted(matches):
            if not candidate.is_file():
                continue
            key = candidate.resolve()
            if key in seen:
                continue
            seen.add(key)
            resolved.append(candidate)

    if not resolved:
        raise ValueError("No input slides were found from --wsi paths/patterns")

    return resolved


def build_output_mapping(
    *,
    input_wsi_paths: list[Path],
    output_path: Path | None,
    output_dir: Path | None,
) -> list[tuple[Path, Path]]:
    if len(input_wsi_paths) == 1:
        if output_path is not None and output_dir is not None:
            raise ValueError("For single-slide mode, provide only one of --output or --output-dir")
        if output_path is None and output_dir is None:
            raise ValueError("For single-slide mode, provide --output or --output-dir")
        slide_path = input_wsi_paths[0]
        final_output = output_path if output_path is not None else output_dir / f"{slide_path.stem}.tif"
        return [(slide_path, final_output)]

    if output_path is not None:
        raise ValueError("For multi-slide mode, use --output-dir (not --output)")
    if output_dir is None:
        raise ValueError("For multi-slide mode, --output-dir is required")

    used_names: set[str] = set()
    mapping: list[tuple[Path, Path]] = []
    for slide_path in input_wsi_paths:
        out_name = f"{slide_path.stem}.tif"
        if out_name in used_names:
            raise ValueError(
                "Duplicate slide stem detected while using --output-dir: "
                f"{out_name}. Rename files or process separately."
            )
        used_names.add(out_name)
        mapping.append((slide_path, output_dir / out_name))
    return mapping


def write_pyramidal_mask_tiff(
    *,
    levels: list[np.ndarray],
    output_path: Path,
    level0_spacing: float,
    downsample_per_level: float,
    compression: str,
    tile_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_level(
        tif: tifffile.TiffWriter,
        level_arr: np.ndarray,
        spacing: float,
        *,
        subifds: int | None = None,
        subfiletype: int | None = None,
    ) -> None:
        height, width = level_arr.shape[:2]
        write_kwargs = {
            "data": level_arr,
            "photometric": "minisblack",
            "compression": compression,
            "resolution": (_spacing_to_pixels_per_cm(spacing), _spacing_to_pixels_per_cm(spacing)),
            "resolutionunit": "CENTIMETER",
            "metadata": None,
        }
        if height >= tile_size and width >= tile_size:
            write_kwargs["tile"] = (tile_size, tile_size)
        if subifds is not None:
            write_kwargs["subifds"] = subifds
        if subfiletype is not None:
            write_kwargs["subfiletype"] = subfiletype
        tif.write(**write_kwargs)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        _write_level(tif, levels[0], level0_spacing)

        for idx, level in enumerate(levels[1:], start=1):
            spacing = level0_spacing * (downsample_per_level ** idx)
            _write_level(
                tif,
                level,
                spacing,
                subfiletype=1,
            )


def build_command_signature(
    *,
    spacing: float,
    tolerance: float,
    downsample_per_level: float,
    backend: str,
    spacing_at_level_0: float | None,
    min_component_area_um2: float,
    min_hole_area_um2: float,
    gaussian_sigma_um: float,
    open_radius_um: float,
    close_radius_um: float,
    disable_coarse_roi_shortcut: bool,
    coarse_spacing: float,
    coarse_roi_margin_um: float,
    processing_tile_size: int,
    compression: str,
    tile_size: int,
    min_size: int | None,
) -> str:
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {
        "spacing": spacing,
        "tolerance": tolerance,
        "downsample_per_level": downsample_per_level,
        "backend": backend,
        "spacing_at_level_0": spacing_at_level_0,
        "min_component_area_um2": min_component_area_um2,
        "min_hole_area_um2": min_hole_area_um2,
        "gaussian_sigma_um": gaussian_sigma_um,
        "open_radius_um": open_radius_um,
        "close_radius_um": close_radius_um,
        "disable_coarse_roi_shortcut": disable_coarse_roi_shortcut,
        "coarse_spacing": coarse_spacing,
        "coarse_roi_margin_um": coarse_roi_margin_um,
        "processing_tile_size": processing_tile_size,
        "compression": compression,
        "tile_size": tile_size,
        "min_size": min_size,
        "script_hash": script_hash,
    }
    payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload_bytes).hexdigest()


def get_file_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_cache_manifest(cache_manifest_path: Path) -> dict[str, object]:
    if not cache_manifest_path.is_file():
        return {"command_signature": None, "entries": {}}
    try:
        with cache_manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"command_signature": None, "entries": {}}

    if not isinstance(payload, dict):
        return {"command_signature": None, "entries": {}}

    entries = payload.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {
        "command_signature": payload.get("command_signature"),
        "entries": entries,
    }


def save_cache_manifest(
    cache_manifest_path: Path,
    *,
    command_signature: str,
    entries: dict[str, dict[str, object]],
) -> None:
    cache_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command_signature": command_signature,
        "entries": entries,
    }
    with cache_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def is_valid_output_tiff(output_path: Path) -> bool:
    if not output_path.is_file():
        return False
    try:
        with tifffile.TiffFile(output_path) as tf:
            return len(tf.pages) > 0
    except Exception:
        return False


def can_skip_slide(
    *,
    input_wsi: Path,
    output_path: Path,
    cached_entry: dict[str, object] | None,
) -> bool:
    if cached_entry is None:
        return False
    if not is_valid_output_tiff(output_path):
        return False

    expected_output_path = str(output_path.resolve())
    if cached_entry.get("output_path") != expected_output_path:
        return False

    expected_input_fingerprint = cached_entry.get("input_fingerprint")
    expected_output_fingerprint = cached_entry.get("output_fingerprint")
    if not isinstance(expected_input_fingerprint, dict) or not isinstance(expected_output_fingerprint, dict):
        return False

    current_input_fingerprint = get_file_fingerprint(input_wsi)
    current_output_fingerprint = get_file_fingerprint(output_path)
    return (
        current_input_fingerprint == expected_input_fingerprint
        and current_output_fingerprint == expected_output_fingerprint
    )


def _process_slide_job(job: dict[str, object]) -> dict[str, object]:
    input_wsi = Path(str(job["input_wsi"]))
    output_path = Path(str(job["output_path"]))
    downsample_per_level = float(job["downsample_per_level"])

    try:
        if bool(job.get("disable_coarse_roi_shortcut", False)):
            wsi_arr, effective_spacing = load_wsi_at_spacing(
                wsi_path=input_wsi,
                target_spacing=float(job["spacing"]),
                tolerance=float(job["tolerance"]),
                backend=str(job["backend"]),
                spacing_at_level_0=job["spacing_at_level_0"],
            )
            gaussian_sigma_px = _sigma_um_to_px(float(job["gaussian_sigma_um"]), effective_spacing)
            mask_l0 = segment_tissue_hsv(wsi_arr=wsi_arr, gaussian_sigma_px=gaussian_sigma_px)
            mask_l0 = postprocess_mask(
                mask=mask_l0,
                spacing_um_per_px=effective_spacing,
                min_component_area_um2=float(job["min_component_area_um2"]),
                min_hole_area_um2=float(job["min_hole_area_um2"]),
                open_radius_um=float(job["open_radius_um"]),
                close_radius_um=float(job["close_radius_um"]),
            )
            processing_mode = "full"
        else:
            mask_l0, effective_spacing, processing_mode = _compute_level0_mask_with_coarse_roi_shortcut(
                wsi_path=input_wsi,
                target_spacing=float(job["spacing"]),
                tolerance=float(job["tolerance"]),
                backend=str(job["backend"]),
                spacing_at_level_0=job["spacing_at_level_0"],
                coarse_spacing=float(job["coarse_spacing"]),
                coarse_roi_margin_um=float(job["coarse_roi_margin_um"]),
                processing_tile_size=int(job["processing_tile_size"]),
                min_component_area_um2=float(job["min_component_area_um2"]),
                min_hole_area_um2=float(job["min_hole_area_um2"]),
                gaussian_sigma_um=float(job["gaussian_sigma_um"]),
                open_radius_um=float(job["open_radius_um"]),
                close_radius_um=float(job["close_radius_um"]),
            )

        resolved_min_size = job["min_size"] if job["min_size"] is not None else int(job["tile_size"])
        levels = build_mask_pyramid(
            level0_mask=mask_l0,
            downsample_per_level=downsample_per_level,
            min_size=int(resolved_min_size),
        )

        write_pyramidal_mask_tiff(
            levels=levels,
            output_path=output_path,
            level0_spacing=effective_spacing,
            downsample_per_level=downsample_per_level,
            compression=str(job["compression"]),
            tile_size=int(job["tile_size"]),
        )

        level_info = []
        for idx, level in enumerate(levels):
            level_spacing = effective_spacing * (downsample_per_level ** idx)
            height, width = level.shape[:2]
            level_info.append(
                {
                    "level": idx,
                    "width": int(width),
                    "height": int(height),
                    "spacing": float(level_spacing),
                }
            )

        return {
            "index": int(job["index"]),
            "slide_path": str(input_wsi),
            "output_path": str(output_path),
            "status": "success",
            "traceback": "",
            "cache_key": str(input_wsi.resolve()),
            "cache_entry": {
                "slide_path": str(input_wsi.resolve()),
                "output_path": str(output_path.resolve()),
                "input_fingerprint": get_file_fingerprint(input_wsi),
                "output_fingerprint": get_file_fingerprint(output_path),
            },
            "level_info": level_info,
            "processing_mode": processing_mode,
        }
    except Exception:
        return {
            "index": int(job["index"]),
            "slide_path": str(input_wsi),
            "output_path": str(output_path),
            "status": "failed",
            "traceback": traceback.format_exc(),
            "cache_key": str(input_wsi.resolve()),
            "cache_entry": None,
            "level_info": [],
            "processing_mode": "failed",
        }


def process_slides(
    *,
    output_mapping: list[tuple[Path, Path]],
    spacing: float,
    tolerance: float,
    backend: str,
    spacing_at_level_0: float | None,
    downsample_per_level: float,
    min_size: int | None,
    compression: str,
    tile_size: int,
    verbose: bool,
    command_signature: str,
    cache_manifest_path: Path,
    no_cache: bool = False,
    num_workers: int = 1,
    min_component_area_um2: float = 0.0,
    min_hole_area_um2: float = 0.0,
    gaussian_sigma_um: float = 0.0,
    open_radius_um: float = 0.0,
    close_radius_um: float = 0.0,
    disable_coarse_roi_shortcut: bool = False,
    coarse_spacing: float = 4.0,
    coarse_roi_margin_um: float = 64.0,
    processing_tile_size: int = 2048,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    cache_manifest = load_cache_manifest(cache_manifest_path) if not no_cache else {
        "command_signature": None,
        "entries": {},
    }
    cached_command_signature = cache_manifest.get("command_signature")
    cached_entries_raw = cache_manifest.get("entries", {})
    cached_entries = cached_entries_raw if isinstance(cached_entries_raw, dict) else {}

    if cached_command_signature == command_signature:
        updated_entries: dict[str, dict[str, object]] = {
            k: v for k, v in cached_entries.items() if isinstance(v, dict)
        }
    else:
        updated_entries = {}

    jobs: list[dict[str, object]] = []
    progress_bar = tqdm(total=len(output_mapping), desc="Generating tissue masks", unit="slide")

    for index, (input_wsi, output_path) in enumerate(output_mapping, start=1):
        traceback_text = ""
        status = "success"
        cache_key = str(input_wsi.resolve())

        if (not no_cache) and cached_command_signature == command_signature and can_skip_slide(
            input_wsi=input_wsi,
            output_path=output_path,
            cached_entry=updated_entries.get(cache_key),
        ):
            status = "skipped"
            if verbose:
                print(f"[{index}/{len(output_mapping)}] Skipped (cache hit): {input_wsi}")
            results.append(
                {
                    "slide_path": str(input_wsi),
                    "output_path": str(output_path),
                    "status": status,
                    "traceback": traceback_text,
                }
            )
            progress_bar.update(1)
            continue

        jobs.append(
            {
                "index": index,
                "input_wsi": str(input_wsi),
                "output_path": str(output_path),
                "spacing": spacing,
                "tolerance": tolerance,
                "backend": backend,
                "spacing_at_level_0": spacing_at_level_0,
                "downsample_per_level": downsample_per_level,
                "min_size": min_size,
                "compression": compression,
                "tile_size": tile_size,
                "min_component_area_um2": min_component_area_um2,
                "min_hole_area_um2": min_hole_area_um2,
                "gaussian_sigma_um": gaussian_sigma_um,
                "open_radius_um": open_radius_um,
                "close_radius_um": close_radius_um,
                "disable_coarse_roi_shortcut": disable_coarse_roi_shortcut,
                "coarse_spacing": coarse_spacing,
                "coarse_roi_margin_um": coarse_roi_margin_um,
                "processing_tile_size": processing_tile_size,
            }
        )

    def _handle_job_result(result: dict[str, object]) -> None:
        index = int(result["index"])
        slide_path = str(result["slide_path"])
        output_path = str(result["output_path"])
        status = str(result["status"])
        traceback_text = str(result["traceback"])
        cache_key = str(result["cache_key"])

        if status == "success":
            cache_entry = result.get("cache_entry")
            if isinstance(cache_entry, dict):
                updated_entries[cache_key] = cache_entry
            if verbose:
                print(f"[{index}/{len(output_mapping)}] {slide_path}")
                level_info = result.get("level_info", [])
                processing_mode = result.get("processing_mode", "unknown")
                print(f"Processing mode: {processing_mode}")
                if isinstance(level_info, list):
                    print(f"Generated mask pyramid with {len(level_info)} level(s):")
                    for info in level_info:
                        print(
                            f"  L{info['level']}: {info['width']}x{info['height']} px @ {info['spacing']:.6f} um/px"
                        )
                print(f"Saved pyramidal tissue mask to {output_path}")
        else:
            updated_entries.pop(cache_key, None)
            print(f"[{index}/{len(output_mapping)}] Failed: {slide_path}")

        results.append(
            {
                "slide_path": slide_path,
                "output_path": output_path,
                "status": status,
                "traceback": traceback_text,
            }
        )

    try:
        if jobs:
            if num_workers > 1:
                worker_count = min(num_workers, len(jobs))
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=worker_count) as pool:
                    for result in pool.imap_unordered(_process_slide_job, jobs):
                        _handle_job_result(result)
                        progress_bar.update(1)
            else:
                for job in jobs:
                    result = _process_slide_job(job)
                    _handle_job_result(result)
                    progress_bar.update(1)
    finally:
        progress_bar.close()

    save_cache_manifest(
        cache_manifest_path,
        command_signature=command_signature,
        entries=updated_entries,
    )

    return results


def write_summary_csv(summary_csv_path: Path, results: list[dict[str, str]]) -> None:
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["slide_path", "output_path", "status", "traceback"]
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def get_summary_csv_path(*, output_path: Path | None, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir / "summary.csv"
    assert output_path is not None
    return output_path.parent / "summary.csv"


def get_cache_manifest_path(*, output_path: Path | None, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir / "cache_manifest.json"
    assert output_path is not None
    return output_path.parent / "cache_manifest.json"


def main() -> None:
    args = _parse_args()

    if args.spacing <= 0:
        raise ValueError("spacing must be > 0")
    if args.downsample_per_level <= 1:
        raise ValueError("downsample-per-level must be > 1")
    if args.num_workers < 1:
        raise ValueError("num-workers must be >= 1")
    if args.min_component_area_um2 < 0:
        raise ValueError("min-component-area-um2 must be >= 0")
    if args.min_hole_area_um2 < 0:
        raise ValueError("min-hole-area-um2 must be >= 0")
    if args.gaussian_sigma_um < 0:
        raise ValueError("gaussian-sigma-um must be >= 0")
    if args.coarse_spacing <= 0:
        raise ValueError("coarse-spacing must be > 0")
    if args.coarse_roi_margin_um < 0:
        raise ValueError("coarse-roi-margin-um must be >= 0")
    if args.processing_tile_size < 1:
        raise ValueError("processing-tile-size must be >= 1")
    if args.open_radius_um < 0:
        raise ValueError("open-radius-um must be >= 0")
    if args.close_radius_um < 0:
        raise ValueError("close-radius-um must be >= 0")

    input_wsi_paths = resolve_wsi_paths(args.wsi)
    output_mapping = build_output_mapping(
        input_wsi_paths=input_wsi_paths,
        output_path=args.output,
        output_dir=args.output_dir,
    )
    command_signature = build_command_signature(
        spacing=args.spacing,
        tolerance=args.tolerance,
        downsample_per_level=args.downsample_per_level,
        backend=args.backend,
        spacing_at_level_0=args.spacing_at_level_0,
        min_component_area_um2=args.min_component_area_um2,
        min_hole_area_um2=args.min_hole_area_um2,
        gaussian_sigma_um=args.gaussian_sigma_um,
        open_radius_um=args.open_radius_um,
        close_radius_um=args.close_radius_um,
        disable_coarse_roi_shortcut=args.disable_coarse_roi_shortcut,
        coarse_spacing=args.coarse_spacing,
        coarse_roi_margin_um=args.coarse_roi_margin_um,
        processing_tile_size=args.processing_tile_size,
        compression=args.compression,
        tile_size=args.tile_size,
        min_size=args.min_size,
    )
    cache_manifest_path = get_cache_manifest_path(output_path=args.output, output_dir=args.output_dir)

    results = process_slides(
        output_mapping=output_mapping,
        spacing=args.spacing,
        tolerance=args.tolerance,
        backend=args.backend,
        spacing_at_level_0=args.spacing_at_level_0,
        downsample_per_level=args.downsample_per_level,
        min_size=args.min_size,
        compression=args.compression,
        tile_size=args.tile_size,
        verbose=args.verbose,
        command_signature=command_signature,
        cache_manifest_path=cache_manifest_path,
        no_cache=args.no_cache,
        num_workers=args.num_workers,
        min_component_area_um2=args.min_component_area_um2,
        min_hole_area_um2=args.min_hole_area_um2,
        gaussian_sigma_um=args.gaussian_sigma_um,
        open_radius_um=args.open_radius_um,
        close_radius_um=args.close_radius_um,
        disable_coarse_roi_shortcut=args.disable_coarse_roi_shortcut,
        coarse_spacing=args.coarse_spacing,
        coarse_roi_margin_um=args.coarse_roi_margin_um,
        processing_tile_size=args.processing_tile_size,
    )

    summary_csv_path = get_summary_csv_path(output_path=args.output, output_dir=args.output_dir)
    write_summary_csv(summary_csv_path, results)

    success_count = sum(row["status"] == "success" for row in results)
    skipped_count = sum(row["status"] == "skipped" for row in results)
    failed_count = sum(row["status"] == "failed" for row in results)
    print(
        f"Processing summary: {success_count} succeeded, {skipped_count} skipped, {failed_count} failed"
    )
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
