from pathlib import Path
from dataclasses import dataclass

import cv2
import tqdm
import numpy as np
from PIL import Image, ImageOps
from collections import defaultdict

from .wsi import (
    FilterParameters,
    SegmentationParameters,
    _SupportsTilingParams,
    SamplingParameters,
    WholeSlideImage,
)

DEFAULT_TISSUE_PIXEL_MAPPING = {"background": 0, "tissue": 1}
DEFAULT_TISSUE_COLOR_MAPPING = {
    "background": None,
    "tissue": [157, 219, 129],
}


@dataclass(init=False)
class CoordinateExtractionResult:
    """Low-level coordinate extraction output for one slide.

    Attributes:
        coordinates: Tile origin coordinates as ``(x, y)`` pairs in level-0 pixels.
            This is reconstructed lazily from ``x`` and ``y``.
        contour_indices: Contour index associated with each retained tile.
        tissue_percentages: Per-tile tissue coverage values measured during extraction.
        x: Tile origin x-coordinates in level-0 pixels.
        y: Tile origin y-coordinates in level-0 pixels.
        read_level: Pyramid level actually read from the slide.
        read_spacing_um: Native spacing of the pyramid level that was read.
        read_tile_size_px: Tile width and height at the read level.
        resize_factor: Resize factor between the read level and requested spacing.
        tile_size_lv0: Tile width and height expressed in level-0 pixels.
    """

    contour_indices: list[int]
    tissue_percentages: list[float]
    x: np.ndarray
    y: np.ndarray
    read_level: int
    read_spacing_um: float
    read_tile_size_px: int
    resize_factor: float
    tile_size_lv0: int

    def __init__(
        self,
        *,
        contour_indices: list[int],
        tissue_percentages: list[float],
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        read_level: int,
        read_spacing_um: float,
        read_tile_size_px: int,
        resize_factor: float,
        tile_size_lv0: int,
        coordinates: list[tuple[int, int]] | None = None,
    ):
        if x is None or y is None:
            if coordinates is None:
                raise TypeError("Provide x and y arrays or coordinates")
            x = np.array([coord[0] for coord in coordinates], dtype=np.int64)
            y = np.array([coord[1] for coord in coordinates], dtype=np.int64)
        else:
            x = np.asarray(x, dtype=np.int64)
            y = np.asarray(y, dtype=np.int64)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError("x and y must be one-dimensional arrays of equal length")
        if coordinates is not None:
            expected_x = np.array([coord[0] for coord in coordinates], dtype=np.int64)
            expected_y = np.array([coord[1] for coord in coordinates], dtype=np.int64)
            if not np.array_equal(x, expected_x) or not np.array_equal(y, expected_y):
                raise ValueError("coordinates must match the provided x and y arrays")

        self.contour_indices = list(contour_indices)
        self.tissue_percentages = list(tissue_percentages)
        self.x = x
        self.y = y
        self.read_level = read_level
        self.read_spacing_um = read_spacing_um
        self.read_tile_size_px = read_tile_size_px
        self.resize_factor = resize_factor
        self.tile_size_lv0 = tile_size_lv0

    @property
    def coordinates(self) -> list[tuple[int, int]]:
        return list(zip(self.x.tolist(), self.y.tolist()))


def sort_coordinates_with_tissue(coordinates, tissue_percentages, contour_indices):
    """
    Deduplicate coordinates, then sort deterministically in column-major order.

    The final order is numeric ``x`` first, then numeric ``y`` within each
    shared ``x`` value. In other words, tiles are grouped by column before row.
    This ordering defines the saved artifact row order for ``x``, ``y``, and
    aligned arrays such as ``tile_index`` / ``tissue_fraction``.
    """
    seen = set()
    dedup_coordinates = []
    dedup_tissue_percentages = []
    dedup_contour_indices = []
    # deduplicate
    for (x, y), tissue, contour_idx in zip(
        coordinates, tissue_percentages, contour_indices
    ):
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        dedup_coordinates.append((x, y))
        dedup_tissue_percentages.append(tissue)
        dedup_contour_indices.append(contour_idx)
    if not dedup_coordinates:
        return [], [], []
    coord_array = np.asarray(dedup_coordinates, dtype=np.int64)
    order = np.lexsort((coord_array[:, 1], coord_array[:, 0]))
    sorted_coordinates = [dedup_coordinates[idx] for idx in order]
    sorted_tissue_percentages = [dedup_tissue_percentages[idx] for idx in order]
    sorted_contour_indices = [dedup_contour_indices[idx] for idx in order]
    return sorted_coordinates, sorted_tissue_percentages, sorted_contour_indices


def get_mask_coverage(mask: np.ndarray, val: int):
    """
    Determine the percentage of a mask that is equal to value `val`.
    input:
        mask: mask as a numpy array
        val: given value we want to check for
    output:
        percentage of the numpy array that is equal to val
    """
    binary_mask = mask == val
    mask_percentage = np.sum(binary_mask) / np.size(mask)
    return mask_percentage


def _build_palette(
    *,
    pixel_mapping: dict[str, int],
    color_mapping: dict[str, list[int] | None],
) -> np.ndarray:
    palette = np.zeros(shape=768, dtype=np.uint8)
    for annotation, label_value in pixel_mapping.items():
        color = color_mapping.get(annotation)
        if color is None:
            continue
        palette[label_value * 3 : label_value * 3 + 3] = np.asarray(
            color,
            dtype=np.uint8,
        )
    return palette


def _resolve_overlay_style(
    *,
    palette: np.ndarray | None,
    pixel_mapping: dict[str, int] | None,
    color_mapping: dict[str, list[int] | None] | None,
) -> tuple[np.ndarray, dict[str, int], dict[str, list[int] | None]]:
    if palette is None and pixel_mapping is None and color_mapping is None:
        pixel_mapping = DEFAULT_TISSUE_PIXEL_MAPPING
        color_mapping = DEFAULT_TISSUE_COLOR_MAPPING
        palette = _build_palette(
            pixel_mapping=pixel_mapping,
            color_mapping=color_mapping,
        )
        return palette, pixel_mapping, color_mapping
    if palette is None or pixel_mapping is None or color_mapping is None:
        raise ValueError(
            "Provide either all of palette, pixel_mapping, and color_mapping, or none of them"
        )
    return palette, pixel_mapping, color_mapping


def _normalize_tissue_mask(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    return (mask_arr > 0).astype(np.uint8)


def _mask_level_downsamples(mask_obj) -> list[tuple[float, float]]:
    if hasattr(mask_obj, "level_downsamples"):
        return list(mask_obj.level_downsamples)
    if hasattr(mask_obj, "downsamplings"):
        return [(float(d), float(d)) for d in mask_obj.downsamplings]
    raise AttributeError("Mask object must expose level_downsamples or downsamplings")


def _read_aligned_mask(
    *,
    mask_obj,
    slide_spacing: float,
    slide_dimensions: tuple[int, int],
) -> np.ndarray:
    mask_spacings = list(mask_obj.spacings)
    mask_level_downsamples = _mask_level_downsamples(mask_obj)
    mask_spacing_at_level_0 = mask_spacings[0]
    mask_downsample = slide_spacing / mask_spacing_at_level_0
    mask_level = int(
        np.argmin([abs(ds[0] - mask_downsample) for ds in mask_level_downsamples])
    )
    mask_spacing = mask_spacings[mask_level]
    scale = slide_spacing / mask_spacing
    while scale < 1 and mask_level > 0:
        mask_level -= 1
        mask_spacing = mask_spacings[mask_level]
        scale = slide_spacing / mask_spacing

    mask_arr = mask_obj.get_slide(spacing=mask_spacing)
    target_width, target_height = slide_dimensions
    return cv2.resize(
        mask_arr.astype(np.uint8),
        (int(target_width), int(target_height)),
        interpolation=cv2.INTER_NEAREST,
    )


def _extract_padded_crop(
    arr: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    if arr.ndim == 2:
        crop = np.zeros((height, width), dtype=arr.dtype)
        src = arr[y : y + height, x : x + width]
        crop[: src.shape[0], : src.shape[1]] = src
        return crop
    crop = np.zeros((height, width, arr.shape[2]), dtype=arr.dtype)
    src = arr[y : y + height, x : x + width, :]
    crop[: src.shape[0], : src.shape[1], :] = src
    return crop


def _compose_overlay_mask_from_annotations(
    *,
    annotation_mask: dict[str, np.ndarray],
    pixel_mapping: dict[str, int],
) -> np.ndarray:
    mask = np.full_like(
        _normalize_tissue_mask(annotation_mask["tissue"]),
        fill_value=pixel_mapping.get("background", 0),
        dtype=np.uint8,
    )
    for annotation, label_value in pixel_mapping.items():
        if annotation == "background":
            continue
        if annotation not in annotation_mask:
            continue
        mask[_normalize_tissue_mask(annotation_mask[annotation]) > 0] = label_value
    return mask


def _save_overlay_preview(
    *,
    wsi_path: Path,
    backend: str,
    mask_arr: np.ndarray,
    mask_visu_path: Path,
    downsample: int = 32,
    palette: np.ndarray | None = None,
    pixel_mapping: dict[str, int] | None = None,
    color_mapping: dict[str, list[int] | None] | None = None,
    tile_size_lv0: int | None = None,
) -> None:
    mask_visu_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_mask_on_slide(
        wsi_path=wsi_path,
        annotation_mask_path=None,
        mask_arr=mask_arr,
        downsample=downsample,
        backend=backend,
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        tile_size_lv0=tile_size_lv0,
    )
    overlay.save(mask_visu_path)


def _build_default_tissue_sampling_params(
    tiling_params: _SupportsTilingParams,
) -> SamplingParameters:
    return SamplingParameters(
        pixel_mapping=DEFAULT_TISSUE_PIXEL_MAPPING,
        color_mapping={"background": None, "tissue": None},
        tissue_percentage={
            "background": None,
            "tissue": tiling_params.min_tissue_percentage,
        },
    )


def extract_coordinates(
    *,
    wsi_path: Path,
    mask_path: Path | None,
    backend: str,
    segment_params: SegmentationParameters,
    tiling_params: _SupportsTilingParams,
    filter_params: FilterParameters,
    sampling_params: SamplingParameters | None = None,
    mask_visu_path: Path | None = None,
    preview_downsample: int = 32,
    preview_palette: np.ndarray | None = None,
    preview_pixel_mapping: dict[str, int] | None = None,
    preview_color_mapping: dict[str, list[int] | None] | None = None,
    disable_tqdm: bool = False,
    num_workers: int = 1,
):
    if sampling_params is None:
        sampling_params = _build_default_tissue_sampling_params(tiling_params)
    wsi = WholeSlideImage(
        path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        segment=True,
        segment_params=segment_params,
        sampling_params=sampling_params,
    )
    tolerance = tiling_params.tolerance
    starting_spacing = wsi.spacings[0]
    target_spacing = tiling_params.spacing
    if target_spacing < starting_spacing:
        relative_diff = abs(starting_spacing - target_spacing) / target_spacing
        if relative_diff > tolerance:
            raise ValueError(
                f"Desired spacing ({target_spacing}) is smaller than the whole-slide image starting spacing ({starting_spacing}) and does not fall within tolerance ({tolerance})"
            )
    (
        coordinates,
        tissue_percentages,
        contour_indices,
        tile_level,
        resize_factor,
        tile_size_lv0,
    ) = wsi.get_tile_coordinates(
        tiling_params,
        filter_params,
        disable_tqdm=disable_tqdm,
        num_workers=num_workers,
    )
    sorted_coordinates, sorted_tissue_percentages, sorted_contour_indices = (
        sort_coordinates_with_tissue(coordinates, tissue_percentages, contour_indices)
    )
    if mask_visu_path is not None:
        preview_mask_arr = _normalize_tissue_mask(wsi.annotation_mask["tissue"])
        if preview_pixel_mapping is not None and preview_color_mapping is not None:
            preview_mask_arr = _compose_overlay_mask_from_annotations(
                annotation_mask=wsi.annotation_mask,
                pixel_mapping=preview_pixel_mapping,
            )
        _save_overlay_preview(
            wsi_path=wsi_path,
            backend=backend,
            mask_arr=preview_mask_arr,
            mask_visu_path=mask_visu_path,
            downsample=preview_downsample,
            palette=preview_palette,
            pixel_mapping=preview_pixel_mapping,
            color_mapping=preview_color_mapping,
            tile_size_lv0=tile_size_lv0,
        )
    tile_spacing = wsi.get_level_spacing(tile_level)
    read_tile_size_px = int(round(tiling_params.tile_size * resize_factor, 0))
    x = np.array([x for x, _ in sorted_coordinates], dtype=np.int64)
    y = np.array([y for _, y in sorted_coordinates], dtype=np.int64)
    return CoordinateExtractionResult(
        contour_indices=sorted_contour_indices,
        tissue_percentages=sorted_tissue_percentages,
        x=x,
        y=y,
        read_level=tile_level,
        read_spacing_um=tile_spacing,
        read_tile_size_px=read_tile_size_px,
        resize_factor=resize_factor,
        tile_size_lv0=tile_size_lv0,
    )


def sample_coordinates(
    *,
    wsi_path: Path,
    mask_path: Path | None,
    backend: str,
    segment_params: SegmentationParameters,
    tiling_params: _SupportsTilingParams,
    filter_params: FilterParameters,
    sampling_params: SamplingParameters,
    annotation: str,
    mask_visu_path: Path | None = None,
    preview_downsample: int = 32,
    disable_tqdm: bool = False,
    num_workers: int = 1,
):
    wsi = WholeSlideImage(
        path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        segment=True,
        segment_params=segment_params,
        sampling_params=sampling_params,
    )
    tolerance = tiling_params.tolerance
    starting_spacing = wsi.spacings[0]
    target_spacing = tiling_params.spacing
    if target_spacing < starting_spacing:
        relative_diff = abs(starting_spacing - target_spacing) / target_spacing
        if relative_diff > tolerance:
            raise ValueError(
                f"Desired spacing ({target_spacing}) is smaller than the whole-slide image starting spacing ({starting_spacing}) and does not fall within tolerance ({tolerance})"
            )
    (
        coordinates,
        tissue_percentages,
        contour_indices,
        tile_level,
        resize_factor,
        tile_size_lv0,
    ) = wsi.get_tile_coordinates(
        tiling_params,
        filter_params,
        annotation=annotation,
        disable_tqdm=disable_tqdm,
        num_workers=num_workers,
    )
    sorted_coordinates, sorted_tissue_percentages, sorted_contour_indices = (
        sort_coordinates_with_tissue(coordinates, tissue_percentages, contour_indices)
    )
    if mask_visu_path is not None:
        _save_overlay_preview(
            wsi_path=wsi_path,
            backend=backend,
            mask_arr=_normalize_tissue_mask(wsi.annotation_mask["tissue"]),
            mask_visu_path=mask_visu_path,
            downsample=preview_downsample,
            tile_size_lv0=tile_size_lv0,
        )
    tile_spacing = wsi.get_level_spacing(tile_level)
    read_tile_size_px = int(round(tiling_params.tile_size * resize_factor, 0))
    x = np.array([x for x, _ in sorted_coordinates], dtype=np.int64)
    y = np.array([y for _, y in sorted_coordinates], dtype=np.int64)
    return CoordinateExtractionResult(
        contour_indices=sorted_contour_indices,
        tissue_percentages=sorted_tissue_percentages,
        x=x,
        y=y,
        read_level=tile_level,
        read_spacing_um=tile_spacing,
        read_tile_size_px=read_tile_size_px,
        resize_factor=resize_factor,
        tile_size_lv0=tile_size_lv0,
    )


def filter_coordinates(
    *,
    wsi_path: Path,
    mask_path: Path | None,
    backend: str,
    coordinates: list[tuple[int, int]],
    contour_indices: list[int],
    tile_level: int,
    segment_params: SegmentationParameters,
    tiling_params: _SupportsTilingParams,
    sampling_params: SamplingParameters,
    disable_tqdm: bool = False,
):
    if mask_path is None:
        raise ValueError("mask_path is required for filter_coordinates()")
    wsi = WholeSlideImage(
        path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        segment=True,
        segment_params=segment_params,
        sampling_params=sampling_params,
    )
    tile_spacing = wsi.get_level_spacing(tile_level)
    resize_factor = tiling_params.spacing / tile_spacing
    tile_size_resized = int(round(tiling_params.tile_size * resize_factor, 0))
    combined = list(zip(coordinates, contour_indices))
    slide_downsample_x, slide_downsample_y = wsi.level_downsamples[tile_level]
    mask = _read_aligned_mask(
        mask_obj=wsi.mask,
        slide_spacing=tile_spacing,
        slide_dimensions=wsi.level_dimensions[tile_level],
    )
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = np.squeeze(mask, axis=-1)

    filtered_coordinates = defaultdict(list)
    filtered_contour_indices = defaultdict(list)
    for annotation, pct in sampling_params.tissue_percentage.items():
        if pct is None:
            continue
        if annotation not in sampling_params.pixel_mapping:
            continue
        with tqdm.tqdm(
            combined,
            desc=f"Filtering coordinates for annotation '{annotation}' (min coverage {pct}%) on slide {wsi_path.stem}",
            unit=" tile",
            total=len(coordinates),
            disable=disable_tqdm,
        ) as t:
            for coord, contour_idx in t:
                x, y = coord
                x_level = int(round(x / slide_downsample_x, 0))
                y_level = int(round(y / slide_downsample_y, 0))
                masked_tile = _extract_padded_crop(
                    mask,
                    x=x_level,
                    y=y_level,
                    width=tile_size_resized,
                    height=tile_size_resized,
                )
                if masked_tile.ndim == 3 and masked_tile.shape[-1] == 1:
                    masked_tile = np.squeeze(masked_tile, axis=-1)
                mask_pct = get_mask_coverage(
                    masked_tile, sampling_params.pixel_mapping[annotation]
                )
                if mask_pct >= pct:
                    filtered_coordinates[annotation].append(coord)
                    filtered_contour_indices[annotation].append(contour_idx)
    return filtered_coordinates, filtered_contour_indices


def overlay_mask_on_tile(
    tile: Image.Image,
    mask: Image.Image,
    palette: dict[str, int],
    pixel_mapping: dict[str, int],
    color_mapping: dict[str, list[int] | None],
    alpha=0.5,
):

    mask_arr = np.array(mask)
    alpha_content = _build_overlay_alpha(
        mask_arr=mask_arr,
        alpha=alpha,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )

    mask.putpalette(data=palette.tolist())
    mask_rgb = mask.convert(mode="RGB")

    overlayed_image = Image.composite(image1=tile, image2=mask_rgb, mask=alpha_content)
    return overlayed_image


def _build_overlay_alpha(
    *,
    mask_arr: np.ndarray,
    alpha: float,
    pixel_mapping: dict[str, int],
    color_mapping: dict[str, list[int] | None],
) -> Image.Image:
    alpha_int = int(round(255 * alpha))
    active_labels = set()
    for annotation, label_value in pixel_mapping.items():
        if color_mapping.get(annotation) is not None:
            active_labels.add(label_value)

    overlay_mask = np.isin(mask_arr, list(active_labels)).astype("uint8")
    alpha_content = np.less(overlay_mask, 1).astype("uint8") * alpha_int + (
        255 - alpha_int
    )
    return Image.fromarray(alpha_content)


def overlay_mask_on_slide(
    wsi_path: Path,
    annotation_mask_path: Path | None,
    downsample: int,
    backend: str,
    palette: np.ndarray | None = None,
    pixel_mapping: dict[str, int] | None = None,
    color_mapping: dict[str, list[int] | None] | None = None,
    alpha: float = 0.5,
    mask_arr: np.ndarray | None = None,
    tile_size_lv0: int | None = None,
):
    """
    Show a mask overlayed on a slide
    """

    palette, pixel_mapping, color_mapping = _resolve_overlay_style(
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )

    wsi_object = WholeSlideImage(path=wsi_path, backend=backend)

    vis_level = wsi_object.get_best_level_for_downsample_custom(downsample)
    vis_spacing = wsi_object.spacings[vis_level]
    wsi_arr = wsi_object.get_slide(spacing=vis_spacing)
    height, width, _ = wsi_arr.shape

    wsi = Image.fromarray(wsi_arr).convert("RGBA")
    if tile_size_lv0 is not None:
        tile_size_at_vis_level = tuple(
            (
                np.array((tile_size_lv0, tile_size_lv0))
                / np.array(wsi_object.level_downsamples[vis_level])
            ).astype(np.int32)
        )
        wsi = pad_to_patch_size(wsi.convert("RGB"), tile_size_at_vis_level).convert(
            "RGBA"
        )
        width, height = wsi.size
    if annotation_mask_path is not None:
        mask_object = WholeSlideImage(path=annotation_mask_path, backend=backend)
        mask_width_at_level_0, _ = mask_object.level_dimensions[0]
        mask_downsample = mask_width_at_level_0 / width
        mask_level = int(
            np.argmin(
                [abs(x - mask_downsample) for x, _ in mask_object.level_downsamples]
            )
        )
        mask_spacing = mask_object.spacings[mask_level]
        mask_width, _ = mask_object.level_dimensions[mask_level]

        scale = mask_width / width
        while scale < 1 and mask_level > 0:
            mask_level -= 1
            mask_spacing = mask_object.spacings[mask_level]
            mask_width, _ = mask_object.level_dimensions[mask_level]
            scale = mask_width / width

        mask_arr = mask_object.get_slide(spacing=mask_spacing)
        mask_arr = mask_arr[:, :, 0]
        mask_height, mask_width = mask_arr.shape
        mask_arr = cv2.resize(
            mask_arr.astype(np.uint8),
            (int(round(mask_width / scale, 0)), int(round(mask_height / scale, 0))),
            interpolation=cv2.INTER_NEAREST,
        )
    elif mask_arr is not None:
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask_arr = cv2.resize(
            mask_arr.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        raise ValueError(
            "Provide annotation_mask_path or mask_arr to overlay_mask_on_slide()"
        )

    mask = Image.fromarray(mask_arr)

    alpha_content = _build_overlay_alpha(
        mask_arr=mask_arr,
        alpha=alpha,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )

    mask.putpalette(data=palette.tolist())
    mask_rgb = mask.convert(mode="RGB")

    overlayed_image = Image.composite(image1=wsi, image2=mask_rgb, mask=alpha_content)
    return overlayed_image


def draw_grid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(
        img,
        tuple(np.maximum([0, 0], coord - thickness // 2)),
        tuple(coord - thickness // 2 + np.array(shape)),
        color,
        thickness=thickness,
    )
    return img


def draw_grid_from_coordinates(
    canvas,
    wsi,
    coords,
    tile_size_at_0,
    vis_level: int,
    thickness: int = 2,
    indices: list[int] | None = None,
    mask=None,
    palette: dict[str, int] | None = None,
    pixel_mapping: dict[str, int] | None = None,
    color_mapping: dict[str, list[int] | None] | None = None,
):
    downsamples = wsi.level_downsamples[vis_level]
    source_canvas = canvas[:, :, :3].copy()
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)

    tile_size = tuple(
        np.ceil((np.array(tile_size_at_0) / np.array(downsamples))).astype(np.int32)
    )  # defined w.r.t vis_level

    wsi_width_at_0, wsi_height_at_0 = wsi.level_dimensions[
        0
    ]  # retrieve slide dimension at level 0

    vis_spacing = wsi.get_level_spacing(vis_level)
    aligned_mask = None
    if mask is not None:
        aligned_mask = _read_aligned_mask(
            mask_obj=mask,
            slide_spacing=vis_spacing,
            slide_dimensions=wsi.level_dimensions[vis_level],
        )
        if aligned_mask.ndim == 3 and aligned_mask.shape[-1] == 1:
            aligned_mask = np.squeeze(aligned_mask, axis=-1)

    for idx in range(total):
        tile_id = indices[idx]
        coord = coords[tile_id]
        x, y = coord

        width, height = tile_size
        tile = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        # compute valid tile area
        if x + tile_size_at_0[0] > wsi_width_at_0:
            valid_width_at_0 = max(
                0, wsi_width_at_0 - x
            )  # how much of the tile width is inside the wsi
            valid_width = int(valid_width_at_0 / downsamples[0])
        else:
            valid_width = width

        if y + tile_size_at_0[1] > wsi_height_at_0:
            valid_height_at_0 = max(
                0, wsi_height_at_0 - y
            )  # how much of the tile height is inside the wsi
            valid_height = int(valid_height_at_0 / downsamples[1])
        else:
            valid_height = height

        coord = np.ceil(
            tuple(coord[i] / downsamples[i] for i in range(len(coord)))
        ).astype(np.int32)

        # extract only the valid portion of the tile
        if valid_width > 0 and valid_height > 0:
            valid_tile = _extract_padded_crop(
                source_canvas,
                x=int(coord[0]),
                y=int(coord[1]),
                width=valid_width,
                height=valid_height,
            )
            valid_tile = Image.fromarray(valid_tile).convert("RGB")

            if aligned_mask is not None:
                if palette is None or pixel_mapping is None or color_mapping is None:
                    raise ValueError(
                        "palette, pixel_mapping, and color_mapping are required when mask overlay is enabled"
                    )
                masked_tile = _extract_padded_crop(
                    aligned_mask,
                    x=int(coord[0]),
                    y=int(coord[1]),
                    width=valid_width,
                    height=valid_height,
                )
                if masked_tile.ndim == 3 and masked_tile.shape[-1] == 1:
                    masked_tile = np.squeeze(masked_tile, axis=-1)
                masked_tile = Image.fromarray(masked_tile)
                masked_tile = masked_tile.split()[0]
                masked_tile = masked_tile.resize(
                    (valid_width, valid_height),
                    Image.NEAREST,
                )
                # masked_tile = cv2.resize(
                #     masked_tile.astype(np.uint8),
                #     (valid_width, valid_height),
                #     interpolation=cv2.INTER_NEAREST,
                # )
                overlayed_tile = overlay_mask_on_tile(
                    valid_tile,
                    masked_tile,
                    palette,
                    pixel_mapping,
                    color_mapping,
                )

                # paste the valid part into the white tile
                tile[:valid_height, :valid_width, :] = overlayed_tile

            else:
                valid_tile = np.array(valid_tile)
                # paste the valid part into the white tile
                tile[:valid_height, :valid_width, :] = valid_tile

        canvas_crop_shape = canvas[
            coord[1] : coord[1] + tile_size[1],
            coord[0] : coord[0] + tile_size[0],
            :3,
        ].shape[:2]
        canvas[
            coord[1] : coord[1] + tile_size[1],
            coord[0] : coord[0] + tile_size[0],
            :3,
        ] = tile[: canvas_crop_shape[0], : canvas_crop_shape[1], :]
        draw_grid(canvas, coord, tile_size, thickness=thickness)

    return Image.fromarray(canvas)


def pad_to_patch_size(canvas: Image.Image, patch_size: tuple[int, int]) -> Image.Image:
    width, height = canvas.size
    # compute amount of padding required for width and height
    pad_width = (patch_size[0] - (width % patch_size[0])) % patch_size[0]
    pad_height = (patch_size[1] - (height % patch_size[1])) % patch_size[1]
    # apply the padding to canvas
    padded_canvas = ImageOps.expand(
        canvas, (0, 0, pad_width, pad_height), fill=(255, 255, 255)
    )  # white padding
    return padded_canvas


def visualize_coordinates(
    *,
    wsi_path: Path,
    coordinates: list[tuple[int, int]],
    tile_size_lv0: int,
    save_dir: Path,
    backend: str,
    sample_id: str | None = None,
    downsample: int = 64,
    grid_thickness: int = 1,
    mask_path: Path | None = None,
    annotation: str | None = None,
    palette: dict[str, int] | None = None,
    pixel_mapping: dict[str, int] | None = None,
    color_mapping: dict[str, list[int] | None] | None = None,
):
    wsi = WholeSlideImage(wsi_path, backend=backend)
    vis_level = wsi.get_best_level_for_downsample_custom(downsample)
    vis_spacing = wsi.spacings[vis_level]

    if mask_path is not None:
        mask = WholeSlideImage(mask_path, backend=backend)
    else:
        mask = None

    canvas = wsi.get_slide(spacing=vis_spacing)
    canvas = Image.fromarray(canvas).convert("RGB")
    if len(coordinates) == 0:
        return canvas

    w, h = wsi.level_dimensions[vis_level]
    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError(
            f"Visualization downsample ({downsample}) is too large"
        )

    tile_size_at_0 = (tile_size_lv0, tile_size_lv0)
    tile_size_at_vis_level = tuple(
        (np.array(tile_size_at_0) / np.array(wsi.level_downsamples[vis_level])).astype(
            np.int32
        )
    )  # defined w.r.t vis_level

    canvas = pad_to_patch_size(canvas, tile_size_at_vis_level)
    canvas = np.array(canvas)
    canvas = draw_grid_from_coordinates(
        canvas,
        wsi,
        coordinates,
        tile_size_at_0,
        vis_level,
        indices=None,
        thickness=grid_thickness,
        mask=mask,
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )
    wsi_name = sample_id if sample_id is not None else wsi_path.stem.replace(" ", "_")
    if annotation is not None:
        save_dir = Path(save_dir, annotation)
        save_dir.mkdir(parents=True, exist_ok=True)
    visu_path = Path(save_dir, f"{wsi_name}.jpg")
    canvas.save(visu_path)
