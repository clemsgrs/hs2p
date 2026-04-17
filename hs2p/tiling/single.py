from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from hs2p.tiling.contours import _normalize_level_downsamples, detect_contours
from hs2p.tiling.coverage import compute_tile_coverage
from hs2p.tiling.generate import generate_tiles
from hs2p.tiling.result import ResolvedAnnotationMasks, ResolvedTissueMask, TilingResult
from hs2p.tiling.mask import resolve_tissue_mask
from hs2p.tile_qc import filter_coordinate_tiles, needs_pixel_qc
from hs2p.wsi.reader import open_slide


def build_tiling_result_from_mask(
    *,
    slide,
    resolved_mask: ResolvedTissueMask,
    image_path: str | Path,
    backend: str,
    requested_backend: str,
    sample_id: str | None = None,
    requested_tile_size_px: int = 256,
    requested_spacing_um: float = 0.5,
    min_tissue_fraction: float = 0.5,
    overlap: float = 0.0,
    tolerance: float = 0.05,
    seg_sthresh: int = 8,
    seg_sthresh_up: int = 255,
    seg_mthresh: int = 7,
    seg_close: int = 4,
    ref_tile_size_px: int = 16,
    a_t: int = 4,
    a_h: int = 0,
    filter_white: bool = False,
    filter_black: bool = False,
    white_threshold: int = 220,
    black_threshold: int = 25,
    fraction_threshold: float = 0.9,
    filter_grayspace: bool = False,
    grayspace_saturation_threshold: float = 0.05,
    grayspace_fraction_threshold: float = 0.6,
    filter_blur: bool = False,
    blur_threshold: float = 50.0,
    qc_spacing_um: float = 2.0,
    num_workers: int = 1,
    annotation: str | None = None,
    selection_strategy: str | None = None,
    output_mode: str | None = None,
) -> TilingResult:
    normalized_downsamples = _normalize_level_downsamples(slide.level_downsamples)
    seg_level = resolved_mask.seg_level
    seg_spacing_um = resolved_mask.seg_spacing_um
    mask = resolved_mask.tissue_mask
    contours = detect_contours(
        mask,
        slide_dimensions=slide.dimensions,
        ref_tile_size_px=ref_tile_size_px,
        requested_spacing_um=requested_spacing_um,
        a_t=a_t,
        base_spacing_um=float(slide.spacing),
        level_downsamples=normalized_downsamples,
        tolerance=tolerance,
    )
    tiles = generate_tiles(
        slide_dimensions=slide.dimensions,
        contours=contours,
        requested_tile_size_px=requested_tile_size_px,
        requested_spacing_um=requested_spacing_um,
        base_spacing_um=float(slide.spacing),
        level_downsamples=normalized_downsamples,
        overlap=overlap,
        min_tissue_fraction=min_tissue_fraction,
        tolerance=tolerance,
        num_workers=num_workers,
    )
    filter_params = SimpleNamespace(
        filter_white=filter_white,
        filter_black=filter_black,
        white_threshold=white_threshold,
        black_threshold=black_threshold,
        fraction_threshold=fraction_threshold,
        filter_grayspace=filter_grayspace,
        grayspace_saturation_threshold=grayspace_saturation_threshold,
        grayspace_fraction_threshold=grayspace_fraction_threshold,
        filter_blur=filter_blur,
        blur_threshold=blur_threshold,
        qc_spacing_um=qc_spacing_um,
    )
    if needs_pixel_qc(filter_params):
        coord_candidates = np.column_stack((tiles.x, tiles.y))
        keep_flags = filter_coordinate_tiles(
            coord_candidates=coord_candidates,
            keep_flags=np.ones(len(coord_candidates), dtype=np.uint8),
            level_dimensions=slide.level_dimensions,
            level_downsamples=slide.level_downsamples,
            requested_tile_size_px=requested_tile_size_px,
            requested_spacing_um=requested_spacing_um,
            base_spacing_um=float(slide.spacing),
            tolerance=tolerance,
            filter_params=filter_params,
            read_window=lambda x, y, width, height, level: slide.read_region(
                (x, y),
                level,
                (width, height),
            ),
            batch_read_windows=None,
            num_workers=num_workers,
            source_label=str(image_path),
        )
        keep = np.asarray(keep_flags, dtype=bool)
        tiles = replace(
            tiles,
            x=tiles.x[keep],
            y=tiles.y[keep],
            tissue_fractions=tiles.tissue_fractions[keep],
            tile_index=np.arange(int(keep.sum()), dtype=np.int32),
        )
    step_px_lv0 = max(1, round(tiles.tile_size_lv0 * (1.0 - overlap)))
    return TilingResult(
        tiles=tiles,
        sample_id=sample_id,
        image_path=Path(image_path),
        backend=backend,
        requested_backend=requested_backend,
        tolerance=tolerance,
        step_px_lv0=step_px_lv0,
        tissue_method=resolved_mask.tissue_method,
        seg_downsample=resolved_mask.seg_downsample,
        seg_level=seg_level,
        seg_spacing_um=seg_spacing_um,
        seg_sthresh=seg_sthresh,
        seg_sthresh_up=seg_sthresh_up,
        seg_mthresh=seg_mthresh,
        seg_close=seg_close,
        ref_tile_size_px=ref_tile_size_px,
        a_t=a_t,
        a_h=a_h,
        filter_white=filter_white,
        filter_black=filter_black,
        white_threshold=white_threshold,
        black_threshold=black_threshold,
        fraction_threshold=fraction_threshold,
        filter_grayspace=filter_grayspace,
        grayspace_saturation_threshold=grayspace_saturation_threshold,
        grayspace_fraction_threshold=grayspace_fraction_threshold,
        filter_blur=filter_blur,
        blur_threshold=blur_threshold,
        qc_spacing_um=qc_spacing_um,
        mask_path=resolved_mask.mask_path,
        tissue_mask_tissue_value=resolved_mask.tissue_mask_tissue_value,
        mask_level=resolved_mask.mask_level,
        mask_spacing_um=resolved_mask.mask_spacing_um,
        contours=contours,
        annotation=annotation,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
    )


def _annotation_to_resolved_tissue_mask(
    annotation: str,
    resolved_masks: ResolvedAnnotationMasks,
) -> ResolvedTissueMask:
    return ResolvedTissueMask(
        tissue_mask=resolved_masks.masks[annotation],
        tissue_method=resolved_masks.tissue_method,
        seg_downsample=resolved_masks.seg_downsample,
        seg_level=resolved_masks.seg_level,
        seg_spacing_um=resolved_masks.seg_spacing_um,
        mask_path=resolved_masks.mask_path,
        tissue_mask_tissue_value=None,
        mask_level=resolved_masks.mask_level,
        mask_spacing_um=resolved_masks.mask_spacing_um,
    )


def _build_independent_annotation_results(
    *,
    resolved_masks: ResolvedAnnotationMasks,
    sampling_spec: Any,
    selection_strategy: str,
    output_mode: str,
    slide,
    image_path: str | Path,
    backend: str,
    requested_backend: str,
    sample_id: str | None,
    requested_tile_size_px: int,
    requested_spacing_um: float,
    overlap: float,
    tolerance: float,
    seg_sthresh: int,
    seg_sthresh_up: int,
    seg_mthresh: int,
    seg_close: int,
    ref_tile_size_px: int,
    a_t: int,
    a_h: int,
    filter_white: bool,
    filter_black: bool,
    white_threshold: int,
    black_threshold: int,
    fraction_threshold: float,
    filter_grayspace: bool,
    grayspace_saturation_threshold: float,
    grayspace_fraction_threshold: float,
    filter_blur: bool,
    blur_threshold: float,
    qc_spacing_um: float,
    num_workers: int,
) -> "dict[str, TilingResult]":
    results: dict[str, TilingResult] = {}
    for annotation in sampling_spec.active_annotations:
        threshold = float(sampling_spec.tissue_percentage.get(annotation) or 0.0)
        result = build_tiling_result_from_mask(
            slide=slide,
            resolved_mask=_annotation_to_resolved_tissue_mask(annotation, resolved_masks),
            image_path=image_path,
            backend=backend,
            requested_backend=requested_backend,
            sample_id=sample_id,
            requested_tile_size_px=requested_tile_size_px,
            requested_spacing_um=requested_spacing_um,
            min_tissue_fraction=threshold,
            overlap=overlap,
            tolerance=tolerance,
            seg_sthresh=seg_sthresh,
            seg_sthresh_up=seg_sthresh_up,
            seg_mthresh=seg_mthresh,
            seg_close=seg_close,
            ref_tile_size_px=ref_tile_size_px,
            a_t=a_t,
            a_h=a_h,
            filter_white=filter_white,
            filter_black=filter_black,
            white_threshold=white_threshold,
            black_threshold=black_threshold,
            fraction_threshold=fraction_threshold,
            filter_grayspace=filter_grayspace,
            grayspace_saturation_threshold=grayspace_saturation_threshold,
            grayspace_fraction_threshold=grayspace_fraction_threshold,
            filter_blur=filter_blur,
            blur_threshold=blur_threshold,
            qc_spacing_um=qc_spacing_um,
            num_workers=num_workers,
            annotation=annotation,
            selection_strategy=selection_strategy,
            output_mode=output_mode,
        )
        results[annotation] = result
    return results


def _build_joint_annotation_results(
    *,
    resolved_masks: ResolvedAnnotationMasks,
    sampling_spec: Any,
    selection_strategy: str,
    output_mode: str,
    slide,
    image_path: str | Path,
    backend: str,
    requested_backend: str,
    sample_id: str | None,
    requested_tile_size_px: int,
    requested_spacing_um: float,
    overlap: float,
    tolerance: float,
    seg_sthresh: int,
    seg_sthresh_up: int,
    seg_mthresh: int,
    seg_close: int,
    ref_tile_size_px: int,
    a_t: int,
    a_h: int,
    filter_white: bool,
    filter_black: bool,
    white_threshold: int,
    black_threshold: int,
    fraction_threshold: float,
    filter_grayspace: bool,
    grayspace_saturation_threshold: float,
    grayspace_fraction_threshold: float,
    filter_blur: bool,
    blur_threshold: float,
    qc_spacing_um: float,
    num_workers: int,
) -> "dict[str, TilingResult]":
    union_mask = np.zeros_like(next(iter(resolved_masks.masks.values())))
    for binary_mask in resolved_masks.masks.values():
        union_mask = np.where(binary_mask > 0, np.uint8(255), union_mask).astype(np.uint8)

    union_resolved = ResolvedTissueMask(
        tissue_mask=union_mask,
        tissue_method=resolved_masks.tissue_method,
        seg_downsample=resolved_masks.seg_downsample,
        seg_level=resolved_masks.seg_level,
        seg_spacing_um=resolved_masks.seg_spacing_um,
        mask_path=resolved_masks.mask_path,
        tissue_mask_tissue_value=None,
        mask_level=resolved_masks.mask_level,
        mask_spacing_um=resolved_masks.mask_spacing_um,
    )

    base_result = build_tiling_result_from_mask(
        slide=slide,
        resolved_mask=union_resolved,
        image_path=image_path,
        backend=backend,
        requested_backend=requested_backend,
        sample_id=sample_id,
        requested_tile_size_px=requested_tile_size_px,
        requested_spacing_um=requested_spacing_um,
        min_tissue_fraction=0.0,
        overlap=overlap,
        tolerance=tolerance,
        seg_sthresh=seg_sthresh,
        seg_sthresh_up=seg_sthresh_up,
        seg_mthresh=seg_mthresh,
        seg_close=seg_close,
        ref_tile_size_px=ref_tile_size_px,
        a_t=a_t,
        a_h=a_h,
        filter_white=filter_white,
        filter_black=filter_black,
        white_threshold=white_threshold,
        black_threshold=black_threshold,
        fraction_threshold=fraction_threshold,
        filter_grayspace=filter_grayspace,
        grayspace_saturation_threshold=grayspace_saturation_threshold,
        grayspace_fraction_threshold=grayspace_fraction_threshold,
        filter_blur=filter_blur,
        blur_threshold=blur_threshold,
        qc_spacing_um=qc_spacing_um,
        num_workers=num_workers,
        annotation=None,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
    )

    results: dict[str, TilingResult] = {}
    if base_result.num_tiles == 0:
        for annotation in sampling_spec.active_annotations:
            results[annotation] = replace(
                base_result,
                annotation=annotation,
                selection_strategy=selection_strategy,
                output_mode=output_mode,
            )
        return results

    candidates = np.column_stack((base_result.tiles.x, base_result.tiles.y))
    slide_dims = tuple(base_result.tiles.slide_dimensions)

    for annotation in sampling_spec.active_annotations:
        per_anno_fracs = compute_tile_coverage(
            candidates,
            resolved_masks.masks[annotation],
            base_result.tiles.tile_size_lv0,
            slide_dims,
        )
        threshold = float(sampling_spec.tissue_percentage.get(annotation) or 0.0)
        keep = per_anno_fracs >= threshold
        n_keep = int(keep.sum())
        filtered_tiles = replace(
            base_result.tiles,
            x=base_result.tiles.x[keep],
            y=base_result.tiles.y[keep],
            tissue_fractions=per_anno_fracs[keep],
            tile_index=np.arange(n_keep, dtype=np.int32),
            min_tissue_fraction=threshold,
        )
        results[annotation] = replace(
            base_result,
            tiles=filtered_tiles,
            annotation=annotation,
            selection_strategy=selection_strategy,
            output_mode=output_mode,
        )
    return results


def build_per_annotation_tiling_results(
    *,
    slide,
    resolved_masks: ResolvedAnnotationMasks,
    sampling_spec: Any,
    selection_strategy: str,
    image_path: str | Path,
    backend: str,
    requested_backend: str,
    sample_id: str | None = None,
    requested_tile_size_px: int = 256,
    requested_spacing_um: float = 0.5,
    overlap: float = 0.0,
    tolerance: float = 0.05,
    seg_sthresh: int = 8,
    seg_sthresh_up: int = 255,
    seg_mthresh: int = 7,
    seg_close: int = 4,
    ref_tile_size_px: int = 16,
    a_t: int = 4,
    a_h: int = 0,
    filter_white: bool = False,
    filter_black: bool = False,
    white_threshold: int = 220,
    black_threshold: int = 25,
    fraction_threshold: float = 0.9,
    filter_grayspace: bool = False,
    grayspace_saturation_threshold: float = 0.05,
    grayspace_fraction_threshold: float = 0.6,
    filter_blur: bool = False,
    blur_threshold: float = 50.0,
    qc_spacing_um: float = 2.0,
    num_workers: int = 1,
    output_mode: str | None = None,
) -> "dict[str, TilingResult]":
    """Tile a slide for each active annotation in sampling_spec.

    INDEPENDENT_SAMPLING: one tiling pass per annotation using that annotation's binary mask.
    JOINT_SAMPLING: one pass on the union mask, then per-annotation post-filter by coverage.
    """
    from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy

    if output_mode is None:
        output_mode = CoordinateOutputMode.PER_ANNOTATION

    _shared = dict(
        resolved_masks=resolved_masks,
        sampling_spec=sampling_spec,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
        slide=slide,
        image_path=image_path,
        backend=backend,
        requested_backend=requested_backend,
        sample_id=sample_id,
        requested_tile_size_px=requested_tile_size_px,
        requested_spacing_um=requested_spacing_um,
        overlap=overlap,
        tolerance=tolerance,
        seg_sthresh=seg_sthresh,
        seg_sthresh_up=seg_sthresh_up,
        seg_mthresh=seg_mthresh,
        seg_close=seg_close,
        ref_tile_size_px=ref_tile_size_px,
        a_t=a_t,
        a_h=a_h,
        filter_white=filter_white,
        filter_black=filter_black,
        white_threshold=white_threshold,
        black_threshold=black_threshold,
        fraction_threshold=fraction_threshold,
        filter_grayspace=filter_grayspace,
        grayspace_saturation_threshold=grayspace_saturation_threshold,
        grayspace_fraction_threshold=grayspace_fraction_threshold,
        filter_blur=filter_blur,
        blur_threshold=blur_threshold,
        qc_spacing_um=qc_spacing_um,
        num_workers=num_workers,
    )

    if selection_strategy == CoordinateSelectionStrategy.INDEPENDENT_SAMPLING:
        return _build_independent_annotation_results(**_shared)
    elif selection_strategy == CoordinateSelectionStrategy.JOINT_SAMPLING:
        return _build_joint_annotation_results(**_shared)
    else:
        raise ValueError(
            f"selection_strategy must be INDEPENDENT_SAMPLING or JOINT_SAMPLING, "
            f"got {selection_strategy!r}"
        )


def preprocess_slide(
    *,
    image_path: str | Path,
    sample_id: str | None = None,
    tissue_mask_path: str | Path | None = None,
    tissue_mask_tissue_value: int = 1,
    backend: str = "auto",
    spacing_override: float | None = None,
    requested_tile_size_px: int = 256,
    requested_spacing_um: float = 0.5,
    tissue_method: str = "hsv",
    sthresh: int = 8,
    sthresh_up: int = 255,
    mthresh: int = 7,
    close: int = 4,
    min_tissue_fraction: float = 0.1,
    overlap: float = 0.0,
    seg_downsample: int = 64,
    sam2_checkpoint_path: str | Path | None = None,
    sam2_config_path: str | Path | None = None,
    sam2_device: str = "cpu",
    tolerance: float = 0.05,
    ref_tile_size_px: int = 16,
    a_t: int = 4,
    a_h: int = 0,
    filter_white: bool = False,
    filter_black: bool = False,
    white_threshold: int = 220,
    black_threshold: int = 25,
    fraction_threshold: float = 0.9,
    filter_grayspace: bool = False,
    grayspace_saturation_threshold: float = 0.05,
    grayspace_fraction_threshold: float = 0.6,
    filter_blur: bool = False,
    blur_threshold: float = 50.0,
    qc_spacing_um: float = 2.0,
    num_workers: int = 1,
    annotation: str | None = None,
    selection_strategy: str | None = None,
    output_mode: str | None = None,
) -> TilingResult:
    slide = open_slide(
        image_path,
        backend=backend,
        spacing_override=spacing_override,
    )
    try:
        resolved_mask = resolve_tissue_mask(
            slide=slide,
            tissue_mask_path=tissue_mask_path,
            tissue_mask_tissue_value=tissue_mask_tissue_value,
            tissue_method=tissue_method,
            sthresh=sthresh,
            sthresh_up=sthresh_up,
            mthresh=mthresh,
            close=close,
            seg_downsample=seg_downsample,
            sam2_checkpoint_path=sam2_checkpoint_path,
            sam2_config_path=sam2_config_path,
            sam2_device=sam2_device,
        )
        return build_tiling_result_from_mask(
            slide=slide,
            resolved_mask=resolved_mask,
            image_path=image_path,
            backend=slide.backend_name,
            requested_backend=backend,
            sample_id=sample_id,
            requested_tile_size_px=requested_tile_size_px,
            requested_spacing_um=requested_spacing_um,
            min_tissue_fraction=min_tissue_fraction,
            overlap=overlap,
            tolerance=tolerance,
            seg_sthresh=sthresh,
            seg_sthresh_up=sthresh_up,
            seg_mthresh=mthresh,
            seg_close=close,
            ref_tile_size_px=ref_tile_size_px,
            a_t=a_t,
            a_h=a_h,
            filter_white=filter_white,
            filter_black=filter_black,
            white_threshold=white_threshold,
            black_threshold=black_threshold,
            fraction_threshold=fraction_threshold,
            filter_grayspace=filter_grayspace,
            grayspace_saturation_threshold=grayspace_saturation_threshold,
            grayspace_fraction_threshold=grayspace_fraction_threshold,
            filter_blur=filter_blur,
            blur_threshold=blur_threshold,
            qc_spacing_um=qc_spacing_um,
            num_workers=num_workers,
            annotation=annotation,
            selection_strategy=selection_strategy,
            output_mode=output_mode,
        )
    finally:
        slide.close()


__all__ = [
    "build_per_annotation_tiling_results",
    "build_tiling_result_from_mask",
    "preprocess_slide",
]
