from pathlib import Path
from dataclasses import dataclass
from typing import Any

import cv2
import tqdm
import numpy as np
from PIL import Image
from collections import defaultdict

from hs2p.configs import FilterConfig, SegmentationConfig, TilingConfig
from hs2p.progress import emit_progress_log
from .masks import (
    compose_overlay_mask_from_annotations as _compose_overlay_mask_from_annotations,
    extract_padded_crop as _extract_padded_crop,
    mask_level_downsamples as _mask_level_downsamples,
    normalize_tissue_mask as _normalize_tissue_mask,
    pad_array_to_shape as _pad_array_to_shape,
    read_aligned_mask as _read_aligned_mask,
)
from .preview import (
    build_overlay_alpha as _build_overlay_alpha,
    build_palette as _build_palette,
    draw_grid,
    draw_grid_from_coordinates,
    overlay_mask_on_tile,
    pad_to_patch_size,
)
from .wsi import (
    ResolvedSamplingSpec,
    WholeSlideImage,
)

DEFAULT_TISSUE_PIXEL_MAPPING = {"background": 0, "tissue": 1}
DEFAULT_TISSUE_COLOR_MAPPING = {
    "background": None,
    "tissue": [157, 219, 129],
}


class CoordinateSelectionStrategy:
    MERGED_DEFAULT_TILING = "merged_default_tiling"
    JOINT_SAMPLING = "joint_sampling"
    INDEPENDENT_SAMPLING = "independent_sampling"


class CoordinateOutputMode:
    SINGLE_OUTPUT = "single_output"
    PER_ANNOTATION = "per_annotation"

@dataclass(frozen=True)
class UnifiedCoordinateRequest:
    wsi_path: Path
    mask_path: Path | None
    backend: str
    segment_params: SegmentationConfig
    tiling_params: TilingConfig
    filter_params: FilterConfig
    sampling_spec: ResolvedSamplingSpec | None
    selection_strategy: str
    output_mode: str
    mask_preview_path: Path | None = None
    mask_preview_paths_by_annotation: dict[str, Path] | None = None
    preview_downsample: int = 32
    preview_palette: np.ndarray | None = None
    preview_pixel_mapping: dict[str, int] | None = None
    preview_color_mapping: dict[str, list[int] | None] | None = None
    spacing_at_level_0: float | None = None
    disable_tqdm: bool = False
    num_workers: int = 1


@dataclass(frozen=True)
class UnifiedCoordinateResponse:
    merged_result: "CoordinateExtractionResult | None" = None
    per_annotation_results: "dict[str, CoordinateExtractionResult] | None" = None


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
        read_step_px: Step between neighboring tiles at the read level.
        resize_factor: Resize factor between the read level and requested spacing.
        step_px_lv0: Step between neighboring tiles in level-0 pixels.
        tile_size_lv0: Tile width and height expressed in level-0 pixels.
    """

    contour_indices: list[int]
    tissue_percentages: list[float]
    x: np.ndarray
    y: np.ndarray
    read_level: int
    read_spacing_um: float
    read_tile_size_px: int
    read_step_px: int
    resize_factor: float
    tile_size_lv0: int
    step_px_lv0: int

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
        read_step_px: int,
        resize_factor: float,
        tile_size_lv0: int,
        step_px_lv0: int,
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
        if int(read_step_px) <= 0:
            raise ValueError(f"read_step_px must be > 0, got {read_step_px}")
        if int(step_px_lv0) <= 0:
            raise ValueError(f"step_px_lv0 must be > 0, got {step_px_lv0}")

        self.contour_indices = list(contour_indices)
        self.tissue_percentages = list(tissue_percentages)
        self.x = x
        self.y = y
        self.read_level = read_level
        self.read_spacing_um = read_spacing_um
        self.read_tile_size_px = read_tile_size_px
        self.read_step_px = int(read_step_px)
        self.resize_factor = resize_factor
        self.tile_size_lv0 = tile_size_lv0
        self.step_px_lv0 = int(step_px_lv0)

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


def _compute_stride_metadata_from_geometry(
    *,
    tiling_params: TilingConfig,
    read_tile_size_px: int,
    tile_level: int,
    level_downsamples: list[tuple[float, float]],
) -> tuple[int, int]:
    read_step_px = max(1, int(read_tile_size_px * (1.0 - tiling_params.overlap)))
    tile_downsample = level_downsamples[tile_level]
    step_x_lv0 = int(round(read_step_px * tile_downsample[0], 0))
    step_y_lv0 = int(round(read_step_px * tile_downsample[1], 0))
    if step_x_lv0 != step_y_lv0:
        raise ValueError(
            "anisotropic level-0 step is not supported for scalar step_px_lv0 metadata: "
            f"x={step_x_lv0}, y={step_y_lv0}"
        )
    return read_step_px, step_x_lv0


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


def _save_overlay_preview(
    *,
    wsi_path: Path,
    backend: str,
    mask_arr: np.ndarray,
    mask_preview_path: Path,
    downsample: int = 32,
    palette: np.ndarray | None = None,
    pixel_mapping: dict[str, int] | None = None,
    color_mapping: dict[str, list[int] | None] | None = None,
    tile_size_lv0: int | None = None,
) -> None:
    mask_preview_path.parent.mkdir(parents=True, exist_ok=True)
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
    overlay.save(mask_preview_path)


def _backend_name(wsi, fallback: str) -> str:
    return str(getattr(wsi, "backend", fallback))


def _build_default_tissue_sampling_spec(
    tiling_params: TilingConfig,
) -> ResolvedSamplingSpec:
    return ResolvedSamplingSpec(
        pixel_mapping=DEFAULT_TISSUE_PIXEL_MAPPING,
        color_mapping={"background": None, "tissue": None},
        tissue_percentage={
            "background": None,
            "tissue": tiling_params.tissue_threshold,
        },
        active_annotations=("tissue",),
    )


def _validate_requested_spacing(
    *,
    wsi: WholeSlideImage,
    tiling_params: TilingConfig,
) -> None:
    tolerance = tiling_params.tolerance
    starting_spacing = wsi.spacings[0]
    target_spacing = tiling_params.target_spacing_um
    if target_spacing < starting_spacing:
        relative_diff = abs(starting_spacing - target_spacing) / target_spacing
        if relative_diff > tolerance:
            raise ValueError(
                f"Desired spacing ({target_spacing}) is smaller than the whole-slide image starting spacing ({starting_spacing}) and does not fall within tolerance ({tolerance:.0%})"
            )


def _extract_coordinate_result_from_wsi(
    *,
    wsi: WholeSlideImage,
    tiling_params: TilingConfig,
    filter_params: FilterConfig,
    annotation: str | None = None,
    disable_tqdm: bool = False,
    num_workers: int = 1,
) -> CoordinateExtractionResult:
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
    tile_spacing = wsi.get_level_spacing(tile_level)
    read_tile_size_px = int(
        round(tiling_params.target_tile_size_px * resize_factor, 0)
    )
    x = np.array([x for x, _ in sorted_coordinates], dtype=np.int64)
    y = np.array([y for _, y in sorted_coordinates], dtype=np.int64)
    read_step_px, step_px_lv0 = _compute_stride_metadata_from_geometry(
        tiling_params=tiling_params,
        read_tile_size_px=read_tile_size_px,
        tile_level=tile_level,
        level_downsamples=wsi.level_downsamples,
    )
    return CoordinateExtractionResult(
        contour_indices=sorted_contour_indices,
        tissue_percentages=sorted_tissue_percentages,
        x=x,
        y=y,
        read_level=tile_level,
        read_spacing_um=tile_spacing,
        read_tile_size_px=read_tile_size_px,
        read_step_px=read_step_px,
        resize_factor=resize_factor,
        tile_size_lv0=tile_size_lv0,
        step_px_lv0=step_px_lv0,
    )


def _save_request_preview(
    *,
    wsi_path: Path,
    backend: str,
    wsi: WholeSlideImage,
    preview_path: Path,
    preview_downsample: int,
    preview_palette: np.ndarray | None,
    preview_pixel_mapping: dict[str, int] | None,
    preview_color_mapping: dict[str, list[int] | None] | None,
    tile_size_lv0: int,
) -> None:
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
        mask_preview_path=preview_path,
        downsample=preview_downsample,
        palette=preview_palette,
        pixel_mapping=preview_pixel_mapping,
        color_mapping=preview_color_mapping,
        tile_size_lv0=tile_size_lv0,
    )


def _filter_coordinates_for_sampling_with_wsi(
    *,
    wsi: WholeSlideImage,
    coordinates: list[tuple[int, int]],
    contour_indices: list[int],
    tissue_percentages: list[float],
    tile_level: int,
    tiling_params: TilingConfig,
    sampling_spec: ResolvedSamplingSpec,
):
    mask = _read_aligned_mask(
        mask_obj=wsi.mask,
        slide_spacing=wsi.get_level_spacing(tile_level),
        slide_dimensions=wsi.level_dimensions[tile_level],
    )
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = np.squeeze(mask, axis=-1)

    coord_array = np.asarray(coordinates, dtype=np.int64)
    contour_index_array = np.asarray(contour_indices)
    tissue_percentage_array = np.asarray(tissue_percentages, dtype=np.float32)
    if coord_array.size == 0:
        return defaultdict(list), defaultdict(list), defaultdict(list)

    tile_spacing = wsi.get_level_spacing(tile_level)
    resize_factor = tiling_params.target_spacing_um / tile_spacing
    tile_size_resized = int(
        round(tiling_params.target_tile_size_px * resize_factor, 0)
    )
    slide_downsample_x, slide_downsample_y = wsi.level_downsamples[tile_level]
    tile_area = float(tile_size_resized * tile_size_resized)
    x_level = np.rint(coord_array[:, 0] / slide_downsample_x).astype(np.int64)
    y_level = np.rint(coord_array[:, 1] / slide_downsample_y).astype(np.int64)
    mask_height, mask_width = mask.shape[:2]
    x1 = np.clip(x_level, 0, mask_width)
    y1 = np.clip(y_level, 0, mask_height)
    x2 = np.clip(x1 + tile_size_resized, 0, mask_width)
    y2 = np.clip(y1 + tile_size_resized, 0, mask_height)

    filtered_coordinates = defaultdict(list)
    filtered_contour_indices = defaultdict(list)
    filtered_tissue_percentages = defaultdict(list)
    for annotation, pct in sampling_spec.tissue_percentage.items():
        if pct is None:
            continue
        if annotation not in sampling_spec.pixel_mapping:
            continue
        label_value = sampling_spec.pixel_mapping[annotation]
        integral = cv2.integral((mask == label_value).astype(np.uint8), sdepth=cv2.CV_32S)
        label_area = (
            integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        ).astype(np.float32)
        keep = (label_area / tile_area) >= float(pct)
        kept_indices = np.flatnonzero(keep)
        filtered_coordinates[annotation].extend(
            tuple(int(v) for v in coord_array[idx]) for idx in kept_indices
        )
        filtered_contour_indices[annotation].extend(
            contour_index_array[kept_indices].tolist()
        )
        filtered_tissue_percentages[annotation].extend(
            tissue_percentage_array[kept_indices].tolist()
        )
    return filtered_coordinates, filtered_contour_indices, filtered_tissue_percentages


def execute_coordinate_request(
    request: UnifiedCoordinateRequest,
) -> UnifiedCoordinateResponse:
    wsi = WholeSlideImage(
        path=request.wsi_path,
        mask_path=request.mask_path,
        backend=request.backend,
        segment=True,
        segment_params=request.segment_params,
        sampling_spec=request.sampling_spec,
        spacing_at_level_0=request.spacing_at_level_0,
    )
    _validate_requested_spacing(wsi=wsi, tiling_params=request.tiling_params)

    if request.selection_strategy == CoordinateSelectionStrategy.INDEPENDENT_SAMPLING:
        if request.sampling_spec is None:
            raise ValueError(
                "sampling_spec is required for independent_sampling requests"
            )
        per_annotation_results: dict[str, CoordinateExtractionResult] = {}
        for annotation in request.sampling_spec.active_annotations:
            extraction = _extract_coordinate_result_from_wsi(
                wsi=wsi,
                tiling_params=request.tiling_params,
                filter_params=request.filter_params,
                annotation=annotation,
                disable_tqdm=request.disable_tqdm,
                num_workers=request.num_workers,
            )
            preview_path = None
            if request.mask_preview_paths_by_annotation is not None:
                preview_path = request.mask_preview_paths_by_annotation.get(annotation)
            elif request.mask_preview_path is not None:
                preview_path = request.mask_preview_path
            if preview_path is not None:
                _save_request_preview(
                    wsi_path=request.wsi_path,
                    backend=_backend_name(wsi, request.backend),
                    wsi=wsi,
                    preview_path=preview_path,
                    preview_downsample=request.preview_downsample,
                    preview_palette=request.preview_palette,
                    preview_pixel_mapping=request.preview_pixel_mapping,
                    preview_color_mapping=request.preview_color_mapping,
                    tile_size_lv0=extraction.tile_size_lv0,
                )
            per_annotation_results[annotation] = extraction
        return UnifiedCoordinateResponse(per_annotation_results=per_annotation_results)

    merged_result = _extract_coordinate_result_from_wsi(
        wsi=wsi,
        tiling_params=request.tiling_params,
        filter_params=request.filter_params,
        annotation=None,
        disable_tqdm=request.disable_tqdm,
        num_workers=request.num_workers,
    )
    if request.mask_preview_path is not None:
        _save_request_preview(
            wsi_path=request.wsi_path,
            backend=_backend_name(wsi, request.backend),
            wsi=wsi,
            preview_path=request.mask_preview_path,
            preview_downsample=request.preview_downsample,
            preview_palette=request.preview_palette,
            preview_pixel_mapping=request.preview_pixel_mapping,
            preview_color_mapping=request.preview_color_mapping,
            tile_size_lv0=merged_result.tile_size_lv0,
        )

    if request.output_mode == CoordinateOutputMode.SINGLE_OUTPUT:
        return UnifiedCoordinateResponse(merged_result=merged_result)

    if request.sampling_spec is None or request.mask_path is None:
        raise ValueError("sampling_spec and mask_path are required for sampling output")

    filtered_coordinates, filtered_contour_indices, filtered_tissue_percentages = (
        _filter_coordinates_for_sampling_with_wsi(
            wsi=wsi,
            coordinates=merged_result.coordinates,
            contour_indices=merged_result.contour_indices,
            tissue_percentages=merged_result.tissue_percentages,
            tile_level=merged_result.read_level,
            tiling_params=request.tiling_params,
            sampling_spec=request.sampling_spec,
        )
    )
    per_annotation_results = {}
    for annotation in request.sampling_spec.active_annotations:
        coordinates = filtered_coordinates.get(annotation, [])
        contour_indices = filtered_contour_indices.get(annotation, [])
        tissue_percentages = filtered_tissue_percentages.get(annotation, [])
        per_annotation_results[annotation] = CoordinateExtractionResult(
            coordinates=coordinates,
            contour_indices=contour_indices,
            tissue_percentages=tissue_percentages,
            read_level=merged_result.read_level,
            read_spacing_um=merged_result.read_spacing_um,
            read_tile_size_px=merged_result.read_tile_size_px,
            read_step_px=merged_result.read_step_px,
            resize_factor=merged_result.resize_factor,
            tile_size_lv0=merged_result.tile_size_lv0,
            step_px_lv0=merged_result.step_px_lv0,
        )
    return UnifiedCoordinateResponse(per_annotation_results=per_annotation_results)


def extract_coordinates(
    *,
    wsi_path: Path,
    mask_path: Path | None,
    backend: str,
    segment_params: SegmentationConfig,
    tiling_params: TilingConfig,
    filter_params: FilterConfig,
    sampling_spec: ResolvedSamplingSpec | None = None,
    mask_preview_path: Path | None = None,
    preview_downsample: int = 32,
    preview_palette: np.ndarray | None = None,
    preview_pixel_mapping: dict[str, int] | None = None,
    preview_color_mapping: dict[str, list[int] | None] | None = None,
    spacing_at_level_0: float | None = None,
    disable_tqdm: bool = False,
    num_workers: int = 1,
):
    resolved_sampling_spec = sampling_spec
    if resolved_sampling_spec is None and mask_path is not None:
        resolved_sampling_spec = _build_default_tissue_sampling_spec(tiling_params)
    response = execute_coordinate_request(
        UnifiedCoordinateRequest(
            wsi_path=wsi_path,
            mask_path=mask_path,
            backend=backend,
            segment_params=segment_params,
            tiling_params=tiling_params,
            filter_params=filter_params,
            sampling_spec=resolved_sampling_spec,
            selection_strategy=CoordinateSelectionStrategy.MERGED_DEFAULT_TILING,
            output_mode=CoordinateOutputMode.SINGLE_OUTPUT,
            mask_preview_path=mask_preview_path,
            preview_downsample=preview_downsample,
            preview_palette=preview_palette,
            preview_pixel_mapping=preview_pixel_mapping,
            preview_color_mapping=preview_color_mapping,
            spacing_at_level_0=spacing_at_level_0,
            disable_tqdm=disable_tqdm,
            num_workers=num_workers,
        )
    )
    if response.merged_result is None:
        raise ValueError("coordinate request did not return a merged extraction result")
    return response.merged_result


def sample_coordinates(
    *,
    wsi_path: Path,
    mask_path: Path | None,
    backend: str,
    segment_params: SegmentationConfig,
    tiling_params: TilingConfig,
    filter_params: FilterConfig,
    sampling_spec: ResolvedSamplingSpec | None = None,
    annotation: str,
    mask_preview_path: Path | None = None,
    preview_downsample: int = 32,
    disable_tqdm: bool = False,
    num_workers: int = 1,
):
    if sampling_spec is None:
        raise ValueError("sampling_spec is required for sample_coordinates()")
    response = execute_coordinate_request(
        UnifiedCoordinateRequest(
            wsi_path=wsi_path,
            mask_path=mask_path,
            backend=backend,
            segment_params=segment_params,
            tiling_params=tiling_params,
            filter_params=filter_params,
            sampling_spec=ResolvedSamplingSpec(
                pixel_mapping=sampling_spec.pixel_mapping,
                color_mapping=sampling_spec.color_mapping,
                tissue_percentage=sampling_spec.tissue_percentage,
                active_annotations=(annotation,),
            ),
            selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
            output_mode=CoordinateOutputMode.PER_ANNOTATION,
            mask_preview_paths_by_annotation=(
                {annotation: mask_preview_path}
                if mask_preview_path is not None
                else None
            ),
            preview_downsample=preview_downsample,
            disable_tqdm=disable_tqdm,
            num_workers=num_workers,
        )
    )
    if response.per_annotation_results is None:
        raise ValueError(
            "coordinate request did not return per-annotation extraction results"
        )
    return response.per_annotation_results[annotation]


def filter_coordinates(
    *,
    wsi_path: Path,
    mask_path: Path | None,
    backend: str,
    coordinates: list[tuple[int, int]],
    contour_indices: list[int],
    tile_level: int,
    segment_params: SegmentationConfig,
    tiling_params: TilingConfig,
    sampling_spec: ResolvedSamplingSpec | None = None,
):
    if mask_path is None:
        raise ValueError("mask_path is required for filter_coordinates()")
    if sampling_spec is None:
        raise ValueError("sampling_spec is required for filter_coordinates()")
    wsi = WholeSlideImage(
        path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        segment=True,
        segment_params=segment_params,
        sampling_spec=sampling_spec,
    )
    filtered_coordinates, filtered_contour_indices, _ = (
        _filter_coordinates_for_sampling_with_wsi(
            wsi=wsi,
            coordinates=coordinates,
            contour_indices=contour_indices,
            tissue_percentages=[0.0] * len(coordinates),
            tile_level=tile_level,
            tiling_params=tiling_params,
            sampling_spec=sampling_spec,
        )
    )
    return filtered_coordinates, filtered_contour_indices


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
    wsi_arr = wsi_object.get_slide(vis_level)
    height, width, _ = wsi_arr.shape
    base_width, base_height = width, height

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
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask_arr = cv2.resize(
            mask_arr.astype(np.uint8),
            (base_width, base_height),
            interpolation=cv2.INTER_NEAREST,
        )
    elif mask_arr is not None:
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask_arr = cv2.resize(
            mask_arr.astype(np.uint8),
            (base_width, base_height),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        raise ValueError(
            "Provide annotation_mask_path or mask_arr to overlay_mask_on_slide()"
        )

    if tile_size_lv0 is not None and (width != base_width or height != base_height):
        mask_arr = _pad_array_to_shape(
            mask_arr,
            target_width=width,
            target_height=height,
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


def write_coordinate_preview(
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
    if mask_path is not None:
        mask = WholeSlideImage(mask_path, backend=backend)
    else:
        mask = None

    canvas = wsi.get_slide(vis_level)
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
    preview_path = Path(save_dir, f"{wsi_name}.jpg")
    canvas.save(preview_path)
