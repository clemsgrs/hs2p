from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from hs2p.configs.models import FilterConfig, SegmentationConfig, TilingConfig
from hs2p.configs.resolvers import validate_pixel_mapping, validate_sampling_spec
from hs2p.mask import AnnotationLabels, Mask
from hs2p.tiling.contours import _normalize_level_downsamples, detect_contours
from hs2p.tiling.coverage import compute_tile_coverage
from hs2p.tiling.generate import generate_tiles
from hs2p.tiling.result import ResolvedAnnotationMasks, ResolvedTissueMask, TilingResult
from hs2p.tiling.mask import (
    open_tissue_mask,
    resolve_annotation_masks,
    resolve_tissue_mask,
)
from hs2p.tile_qc import filter_coordinate_tiles, needs_pixel_qc
from hs2p.wsi.geometry import resolve_tile_stride
from hs2p.wsi.reader import AUTO_BACKEND, open_slide
from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy, PixelMapping
from hs2p.wsi.visualization import _combine_label_masks, save_overlay_preview


@dataclass(frozen=True)
class MaskPreviewRequest:
    """Where and how to render the per-slide multi-label annotation mask preview."""

    mask_preview_path: Path
    color_mapping: dict[str, list[int] | None] | None
    downsample: int = 32
    alpha: float = 0.5


def _render_annotation_mask_preview(
    *,
    request: MaskPreviewRequest,
    resolved_masks: ResolvedAnnotationMasks,
    image_path: str | Path,
    backend: str,
    spacing_at_level_0: float | None,
) -> None:
    """Render one filled, semi-transparent multi-label overlay from the *resolved* per-label
    binary masks (never a re-read of the raw mask file), once per slide. Labels with a null
    color are omitted by the shared palette/alpha builders."""
    if request.color_mapping is None:
        return
    label_arr = _combine_label_masks(
        masks=resolved_masks.masks,
        pixel_mapping=resolved_masks.pixel_mapping,
    )
    save_overlay_preview(
        wsi_path=Path(image_path),
        backend=backend,
        spacing_at_level_0=spacing_at_level_0,
        mask_arr=label_arr,
        mask_preview_path=Path(request.mask_preview_path),
        downsample=request.downsample,
        pixel_mapping=dict(resolved_masks.pixel_mapping),
        color_mapping=dict(request.color_mapping),
        alpha=request.alpha,
    )


def build_tiling_result_from_mask(
    *,
    slide,
    resolved_mask: ResolvedTissueMask,
    image_path: str | Path,
    backend: str,
    requested_backend: str,
    spacing_at_level_0: float | None = None,
    sam2_checkpoint_path: str | Path | None = None,
    sam2_config_path: str | Path | None = None,
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
    return _build_tiling_result_from_mask(
        slide=slide,
        resolved_mask=resolved_mask,
        image_path=image_path,
        backend=backend,
        requested_backend=requested_backend,
        spacing_at_level_0=spacing_at_level_0,
        sample_id=sample_id,
        tiling=TilingConfig(
            requested_spacing_um=requested_spacing_um,
            requested_tile_size_px=requested_tile_size_px,
            tolerance=tolerance,
            overlap=overlap,
            min_coverage={"tissue": min_tissue_fraction},
        ),
        min_tissue_fraction=min_tissue_fraction,
        segmentation=SegmentationConfig(
            method=resolved_mask.tissue_method,
            downsample=resolved_mask.requested_seg_downsample,
            sthresh=seg_sthresh,
            sthresh_up=seg_sthresh_up,
            mthresh=seg_mthresh,
            close=seg_close,
            sam2_checkpoint_path=sam2_checkpoint_path,
            sam2_config_path=sam2_config_path,
        ),
        filtering=FilterConfig(
            ref_tile_size=ref_tile_size_px,
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
        ),
        num_workers=num_workers,
        annotation=annotation,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
    )


def _build_tiling_result_from_mask(
    *,
    slide,
    resolved_mask: ResolvedTissueMask,
    image_path: str | Path,
    backend: str,
    requested_backend: str,
    spacing_at_level_0: float | None,
    sample_id: str | None,
    tiling: TilingConfig,
    min_tissue_fraction: float,
    segmentation: SegmentationConfig | None,
    filtering: FilterConfig,
    num_workers: int,
    annotation: str | None = None,
    selection_strategy: str | None = None,
    output_mode: str | None = None,
) -> TilingResult:
    """Tile ``slide`` over ``resolved_mask``: the typed core of
    :func:`build_tiling_result_from_mask`.

    ``tiling`` supplies the physical tile geometry. The coverage gate is passed on its own
    because every pass picks its own: the configured tissue threshold, one annotation's
    threshold, or ``0.0`` for the joint-sampling union. ``backend``/``requested_backend``
    are the slide reader that actually opened and the one asked for. ``segmentation``
    supplies only the persisted thresholds and SAM2 paths, and is recorded only when it
    produced ``resolved_mask``: a precomputed tissue mask records ``None`` thresholds
    whatever ``segmentation`` says, and annotation sampling, which never segments, passes
    ``None``. The mask's own ``tissue_method`` decides whether the SAM2 paths are recorded.
    """
    # The resolved mask, not the caller's config, says whether segmentation ran.
    applied = (
        None if resolved_mask.tissue_method == "precomputed_mask" else segmentation
    )
    normalized_downsamples = _normalize_level_downsamples(slide.level_downsamples)
    seg_level = resolved_mask.seg_level
    seg_spacing_um = resolved_mask.seg_spacing_um
    mask = resolved_mask.tissue_mask
    contours = detect_contours(
        mask,
        slide_dimensions=slide.dimensions,
        ref_tile_size_px=filtering.ref_tile_size,
        requested_spacing_um=tiling.requested_spacing_um,
        a_t=filtering.a_t,
        base_spacing_um=float(slide.spacing),
        level_downsamples=normalized_downsamples,
        tolerance=tiling.tolerance,
    )
    tiles = generate_tiles(
        slide_dimensions=slide.dimensions,
        contours=contours,
        requested_tile_size_px=tiling.requested_tile_size_px,
        requested_spacing_um=tiling.requested_spacing_um,
        base_spacing_um=float(slide.spacing),
        level_downsamples=normalized_downsamples,
        overlap=tiling.overlap,
        min_tissue_fraction=min_tissue_fraction,
        tolerance=tiling.tolerance,
        num_workers=num_workers,
    )
    if needs_pixel_qc(filtering):
        coord_candidates = np.column_stack((tiles.x, tiles.y))
        keep_flags = filter_coordinate_tiles(
            coord_candidates=coord_candidates,
            keep_flags=np.ones(len(coord_candidates), dtype=np.uint8),
            level_dimensions=slide.level_dimensions,
            level_downsamples=slide.level_downsamples,
            requested_tile_size_px=tiling.requested_tile_size_px,
            requested_spacing_um=tiling.requested_spacing_um,
            base_spacing_um=float(slide.spacing),
            tolerance=tiling.tolerance,
            filter_params=filtering,
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
    step_px_lv0 = resolve_tile_stride(
        read_tile_size_px=tiles.read_tile_size_px,
        tile_size_lv0=tiles.tile_size_lv0,
        overlap=tiling.overlap,
    ).step_px_lv0
    is_sam2 = applied is not None and str(resolved_mask.tissue_method).lower() == "sam2"
    return TilingResult(
        tiles=tiles,
        sample_id=sample_id,
        image_path=Path(image_path),
        backend=backend,
        requested_backend=requested_backend,
        spacing_at_level_0=spacing_at_level_0,
        tolerance=tiling.tolerance,
        step_px_lv0=step_px_lv0,
        tissue_method=resolved_mask.tissue_method,
        requested_seg_downsample=resolved_mask.requested_seg_downsample,
        seg_downsample=resolved_mask.seg_downsample,
        seg_level=seg_level,
        seg_spacing_um=seg_spacing_um,
        seg_sthresh=applied.sthresh if applied is not None else None,
        seg_sthresh_up=applied.sthresh_up if applied is not None else None,
        seg_mthresh=applied.mthresh if applied is not None else None,
        seg_close=applied.close if applied is not None else None,
        sam2_checkpoint_path=applied.sam2_checkpoint_path if is_sam2 else None,
        sam2_config_path=applied.sam2_config_path if is_sam2 else None,
        ref_tile_size_px=filtering.ref_tile_size,
        a_t=filtering.a_t,
        a_h=filtering.a_h,
        filter_white=filtering.filter_white,
        filter_black=filtering.filter_black,
        white_threshold=filtering.white_threshold,
        black_threshold=filtering.black_threshold,
        fraction_threshold=filtering.fraction_threshold,
        filter_grayspace=filtering.filter_grayspace,
        grayspace_saturation_threshold=filtering.grayspace_saturation_threshold,
        grayspace_fraction_threshold=filtering.grayspace_fraction_threshold,
        filter_blur=filtering.filter_blur,
        blur_threshold=filtering.blur_threshold,
        qc_spacing_um=filtering.qc_spacing_um,
        mask_path=resolved_mask.mask_path,
        tissue_mask_tissue_value=resolved_mask.tissue_mask_tissue_value,
        mask_level=resolved_mask.mask_level,
        mask_spacing_um=resolved_mask.mask_spacing_um,
        mask_backend=resolved_mask.mask_backend,
        requested_mask_backend=resolved_mask.requested_mask_backend,
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
        requested_seg_downsample=resolved_masks.requested_seg_downsample,
        seg_downsample=resolved_masks.seg_downsample,
        seg_level=resolved_masks.seg_level,
        seg_spacing_um=resolved_masks.seg_spacing_um,
        mask_path=resolved_masks.mask_path,
        tissue_mask_tissue_value=None,
        mask_level=resolved_masks.mask_level,
        mask_spacing_um=resolved_masks.mask_spacing_um,
        mask_backend=resolved_masks.mask_backend,
        requested_mask_backend=resolved_masks.requested_mask_backend,
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
    spacing_at_level_0: float | None,
    sample_id: str | None,
    tiling: TilingConfig,
    filtering: FilterConfig,
    num_workers: int,
) -> "dict[str, TilingResult]":
    results: dict[str, TilingResult] = {}
    for annotation in sampling_spec.active_annotations:
        threshold = float(sampling_spec.tissue_percentage.get(annotation) or 0.0)
        result = _build_tiling_result_from_mask(
            slide=slide,
            resolved_mask=_annotation_to_resolved_tissue_mask(annotation, resolved_masks),
            image_path=image_path,
            backend=backend,
            requested_backend=requested_backend,
            spacing_at_level_0=spacing_at_level_0,
            sample_id=sample_id,
            tiling=tiling,
            min_tissue_fraction=threshold,
            segmentation=None,
            filtering=filtering,
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
    spacing_at_level_0: float | None,
    sample_id: str | None,
    tiling: TilingConfig,
    filtering: FilterConfig,
    num_workers: int,
) -> "dict[str, TilingResult]":
    # Union over the classes actually being sampled (active_annotations) — never the full
    # mask set, which may include declared-but-unsampled labels (e.g. the value reserved for
    # unannotated pixels) that would otherwise swallow the whole slide.
    active = sampling_spec.active_annotations
    union_mask = np.zeros_like(resolved_masks.masks[active[0]])
    for annotation in active:
        union_mask = np.where(
            resolved_masks.masks[annotation] > 0, np.uint8(255), union_mask
        ).astype(np.uint8)

    union_resolved = ResolvedTissueMask(
        tissue_mask=union_mask,
        tissue_method=resolved_masks.tissue_method,
        requested_seg_downsample=resolved_masks.requested_seg_downsample,
        seg_downsample=resolved_masks.seg_downsample,
        seg_level=resolved_masks.seg_level,
        seg_spacing_um=resolved_masks.seg_spacing_um,
        mask_path=resolved_masks.mask_path,
        tissue_mask_tissue_value=None,
        mask_level=resolved_masks.mask_level,
        mask_spacing_um=resolved_masks.mask_spacing_um,
        mask_backend=resolved_masks.mask_backend,
        requested_mask_backend=resolved_masks.requested_mask_backend,
    )

    base_result = _build_tiling_result_from_mask(
        slide=slide,
        resolved_mask=union_resolved,
        image_path=image_path,
        backend=backend,
        requested_backend=requested_backend,
        spacing_at_level_0=spacing_at_level_0,
        sample_id=sample_id,
        tiling=tiling,
        min_tissue_fraction=0.0,
        segmentation=None,
        filtering=filtering,
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
            candidates=candidates,
            binary_mask=resolved_masks.masks[annotation],
            tile_size_lv0=base_result.tiles.tile_size_lv0,
            slide_dimensions=slide_dims,
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


def _merge_annotation_results_to_single(
    results: "dict[str, TilingResult]",
) -> "TilingResult":
    """Collapse a per-annotation result dict into one merged result for MERGED.

    The merged result is the **union** of every tile that passes *any* active
    annotation's coverage threshold (dedup'd over the shared candidate grid), with
    ``annotation=None``. Each spatial tile therefore appears once — the dense
    segmentation contract, where one tile is encoded once and its full multi-class
    mask is attached downstream. ``tissue_fractions`` carries the max per-class
    coverage seen for each kept tile.
    """
    result_list = list(results.values())
    if not result_list:
        raise ValueError(
            "MERGED merge requires at least one annotation result; got none "
            "(no active annotations in the sampling spec?)"
        )
    template = result_list[0]
    all_x = np.concatenate([np.asarray(r.tiles.x) for r in result_list])
    all_y = np.concatenate([np.asarray(r.tiles.y) for r in result_list])
    all_fracs = np.concatenate(
        [np.asarray(r.tiles.tissue_fractions) for r in result_list]
    )
    if all_x.size:
        coords = np.stack([all_x, all_y], axis=1)
        # np.unique(axis=0) returns lexicographically-sorted unique rows → deterministic.
        uniq, inverse = np.unique(coords, axis=0, return_inverse=True)
        merged_x = uniq[:, 0].astype(template.tiles.x.dtype)
        merged_y = uniq[:, 1].astype(template.tiles.y.dtype)
        merged_fracs = np.zeros(len(uniq), dtype=all_fracs.dtype)
        np.maximum.at(merged_fracs, inverse.ravel(), all_fracs)
    else:
        merged_x = np.asarray(template.tiles.x)[:0]
        merged_y = np.asarray(template.tiles.y)[:0]
        merged_fracs = np.asarray(template.tiles.tissue_fractions)[:0]

    merged_tiles = replace(
        template.tiles,
        x=merged_x,
        y=merged_y,
        tissue_fractions=merged_fracs,
        tile_index=np.arange(len(merged_x), dtype=np.int32),
        min_tissue_fraction=0.0,
    )
    return replace(
        template,
        tiles=merged_tiles,
        annotation=None,
        output_mode=CoordinateOutputMode.MERGED,
    )


def build_per_annotation_tiling_results(
    *,
    slide,
    resolved_masks: ResolvedAnnotationMasks,
    sampling_spec: Any,
    selection_strategy: str,
    image_path: str | Path,
    backend: str,
    requested_backend: str,
    spacing_at_level_0: float | None = None,
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

    The ``seg_*`` thresholds are accepted but unused: an annotation mask is never
    segmented, so every result records ``None`` thresholds.
    """
    return _build_per_annotation_tiling_results(
        slide=slide,
        resolved_masks=resolved_masks,
        sampling_spec=sampling_spec,
        selection_strategy=selection_strategy,
        image_path=image_path,
        backend=backend,
        requested_backend=requested_backend,
        spacing_at_level_0=spacing_at_level_0,
        sample_id=sample_id,
        tiling=TilingConfig(
            requested_spacing_um=requested_spacing_um,
            requested_tile_size_px=requested_tile_size_px,
            tolerance=tolerance,
            overlap=overlap,
            # Annotation sampling gates on the sampling spec's per-label thresholds.
            min_coverage={},
        ),
        filtering=FilterConfig(
            ref_tile_size=ref_tile_size_px,
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
        ),
        num_workers=num_workers,
        output_mode=output_mode,
    )


def _build_per_annotation_tiling_results(
    *,
    slide,
    resolved_masks: ResolvedAnnotationMasks,
    sampling_spec: Any,
    selection_strategy: str,
    image_path: str | Path,
    backend: str,
    requested_backend: str,
    spacing_at_level_0: float | None,
    sample_id: str | None,
    tiling: TilingConfig,
    filtering: FilterConfig,
    num_workers: int,
    output_mode: str | None,
) -> "dict[str, TilingResult]":
    """The typed core of :func:`build_per_annotation_tiling_results`. ``tiling`` supplies
    the tile geometry; each label is gated by its ``sampling_spec`` threshold."""
    validate_sampling_spec(sampling_spec)
    validate_pixel_mapping(resolved_masks.pixel_mapping)
    if output_mode is None:
        output_mode = CoordinateOutputMode.PER_ANNOTATION
    # Validate here (the shared chokepoint for both the CLI and the public tile_slide/
    # tile_slides API), so an invalid value fails fast instead of silently falling through
    # to per-annotation output and recording a bogus mode in metadata/process_list.
    if output_mode not in (
        CoordinateOutputMode.PER_ANNOTATION,
        CoordinateOutputMode.MERGED,
    ):
        raise ValueError(
            f"output_mode must be PER_ANNOTATION or MERGED, got {output_mode!r}"
        )

    _shared = dict(
        resolved_masks=resolved_masks,
        sampling_spec=sampling_spec,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
        slide=slide,
        image_path=image_path,
        backend=backend,
        requested_backend=requested_backend,
        spacing_at_level_0=spacing_at_level_0,
        sample_id=sample_id,
        tiling=tiling,
        filtering=filtering,
        num_workers=num_workers,
    )

    if selection_strategy == CoordinateSelectionStrategy.INDEPENDENT_SAMPLING:
        results = _build_independent_annotation_results(**_shared)
    elif selection_strategy == CoordinateSelectionStrategy.JOINT_SAMPLING:
        results = _build_joint_annotation_results(**_shared)
    else:
        raise ValueError(
            f"selection_strategy must be INDEPENDENT_SAMPLING or JOINT_SAMPLING, "
            f"got {selection_strategy!r}"
        )

    if output_mode == CoordinateOutputMode.MERGED:
        # One merged result per slide (union of tiles passing any class threshold),
        # keyed by None so tile_slide/tile_slides persist a single per-slide artifact.
        return {None: _merge_annotation_results_to_single(results)}
    return results


def preprocess_slide(
    *,
    image_path: str | Path,
    sample_id: str | None = None,
    tissue_mask_path: str | Path | None = None,
    pixel_mapping: PixelMapping | None = None,
    backend: str = "auto",
    requested_backend: str | None = None,
    mask_backend: str | None = None,
    requested_mask_backend: str | None = None,
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
    return _preprocess_slide(
        image_path=image_path,
        sample_id=sample_id,
        tissue_mask_path=tissue_mask_path,
        pixel_mapping=pixel_mapping,
        backend=backend,
        requested_backend=requested_backend,
        mask_backend=mask_backend,
        requested_mask_backend=requested_mask_backend,
        spacing_override=spacing_override,
        tiling=TilingConfig(
            requested_spacing_um=requested_spacing_um,
            requested_tile_size_px=requested_tile_size_px,
            tolerance=tolerance,
            overlap=overlap,
            min_coverage={"tissue": min_tissue_fraction},
        ),
        min_tissue_fraction=min_tissue_fraction,
        segmentation=SegmentationConfig(
            method=tissue_method,
            downsample=seg_downsample,
            sthresh=sthresh,
            sthresh_up=sthresh_up,
            mthresh=mthresh,
            close=close,
            sam2_checkpoint_path=sam2_checkpoint_path,
            sam2_config_path=sam2_config_path,
            sam2_device=sam2_device,
        ),
        filtering=FilterConfig(
            ref_tile_size=ref_tile_size_px,
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
        ),
        num_workers=num_workers,
        annotation=annotation,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
    )


def _preprocess_slide(
    *,
    image_path: str | Path,
    sample_id: str | None,
    tissue_mask_path: str | Path | None,
    pixel_mapping: PixelMapping | None,
    backend: str,
    requested_backend: str | None,
    mask_backend: str | None,
    requested_mask_backend: str | None,
    spacing_override: float | None,
    tiling: TilingConfig,
    min_tissue_fraction: float,
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
    num_workers: int,
    annotation: str | None = None,
    selection_strategy: str | None = None,
    output_mode: str | None = None,
) -> TilingResult:
    """Open the slide, resolve its tissue mask and tile it: the typed core of
    :func:`preprocess_slide`.

    The backends stay explicit rather than coming from ``tiling`` because the scalar API
    accepts spellings (``None``, any case) that :class:`TilingConfig` rejects; ``None``
    requests default to the backend they qualify.
    """
    slide = open_slide(
        image_path,
        backend=backend,
        spacing_override=spacing_override,
    )
    try:
        if mask_backend is None:
            mask_backend = AUTO_BACKEND
        # The tissue mask lives only for its one full read.
        with open_tissue_mask(
            tissue_mask_path, pixel_mapping=pixel_mapping, backend=mask_backend
        ) as mask:
            resolved_mask = resolve_tissue_mask(
                slide=slide,
                sample_id=sample_id,
                mask=mask,
                segmentation=segmentation,
                requested_mask_backend=(
                    requested_mask_backend
                    if requested_mask_backend is not None
                    else mask_backend
                ),
            )
        return _build_tiling_result_from_mask(
            slide=slide,
            resolved_mask=resolved_mask,
            image_path=image_path,
            backend=slide.backend_name,
            requested_backend=requested_backend if requested_backend is not None else backend,
            spacing_at_level_0=spacing_override,
            sample_id=sample_id,
            tiling=tiling,
            min_tissue_fraction=min_tissue_fraction,
            segmentation=segmentation,
            filtering=filtering,
            num_workers=num_workers,
            annotation=annotation,
            selection_strategy=selection_strategy,
            output_mode=output_mode,
        )
    finally:
        slide.close()


def preprocess_slide_per_annotation(
    *,
    image_path: str | Path,
    mask_path: str | Path,
    pixel_mapping: PixelMapping,
    sampling_spec: Any,
    selection_strategy: str,
    sample_id: str | None = None,
    backend: str = "auto",
    requested_backend: str | None = None,
    mask_backend: str | None = None,
    requested_mask_backend: str | None = None,
    spacing_override: float | None = None,
    requested_tile_size_px: int = 256,
    requested_spacing_um: float = 0.5,
    overlap: float = 0.0,
    seg_downsample: int = 64,
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
    output_mode: str | None = None,
    mask_preview: MaskPreviewRequest | None = None,
) -> "dict[str, TilingResult]":
    """Annotation-aware tiling: open the slide, resolve its annotation mask into per-class
    binaries (:func:`resolve_annotation_masks`), then sample per active annotation.

    The annotation counterpart of :func:`preprocess_slide` — it returns one
    :class:`TilingResult` per active annotation (keyed by name) instead of a single tissue
    result. ``selection_strategy`` selects INDEPENDENT vs JOINT sampling; ``output_mode``
    selects single vs per-annotation coordinate output.

    When ``mask_preview`` is given, one filled multi-label overlay is rendered here — once per
    slide, from the just-resolved per-label binary masks — before any sampling.
    """
    return _preprocess_slide_per_annotation(
        image_path=image_path,
        mask_path=mask_path,
        pixel_mapping=pixel_mapping,
        sampling_spec=sampling_spec,
        selection_strategy=selection_strategy,
        sample_id=sample_id,
        backend=backend,
        requested_backend=requested_backend,
        mask_backend=mask_backend,
        requested_mask_backend=requested_mask_backend,
        spacing_override=spacing_override,
        tiling=TilingConfig(
            requested_spacing_um=requested_spacing_um,
            requested_tile_size_px=requested_tile_size_px,
            tolerance=tolerance,
            overlap=overlap,
            # Annotation sampling gates on the sampling spec's per-label thresholds.
            min_coverage={},
        ),
        seg_downsample=seg_downsample,
        filtering=FilterConfig(
            ref_tile_size=ref_tile_size_px,
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
        ),
        num_workers=num_workers,
        output_mode=output_mode,
        mask_preview=mask_preview,
    )


def _preprocess_slide_per_annotation(
    *,
    image_path: str | Path,
    mask_path: str | Path,
    pixel_mapping: PixelMapping,
    sampling_spec: Any,
    selection_strategy: str,
    sample_id: str | None,
    backend: str,
    requested_backend: str | None,
    mask_backend: str | None,
    requested_mask_backend: str | None,
    spacing_override: float | None,
    tiling: TilingConfig,
    seg_downsample: int,
    filtering: FilterConfig,
    num_workers: int,
    output_mode: str | None,
    mask_preview: MaskPreviewRequest | None = None,
) -> "dict[str, TilingResult]":
    """The typed core of :func:`preprocess_slide_per_annotation`. ``seg_downsample`` picks
    the grid the annotation mask is resolved on; backends stay explicit as in
    :func:`_preprocess_slide`."""
    validate_pixel_mapping(pixel_mapping)
    validate_sampling_spec(sampling_spec)
    slide = open_slide(image_path, backend=backend, spacing_override=spacing_override)
    try:
        if mask_backend is None:
            mask_backend = AUTO_BACKEND
        # The annotation mask declares the full configured vocabulary and lives only for
        # its one full read.
        with Mask(
            path=mask_path,
            labels=AnnotationLabels(pixel_mapping=pixel_mapping),
            backend=mask_backend,
        ) as mask:
            resolved_masks = resolve_annotation_masks(
                slide=slide,
                mask=mask,
                seg_downsample=seg_downsample,
                requested_mask_backend=(
                    requested_mask_backend
                    if requested_mask_backend is not None
                    else mask_backend
                ),
            )
        if mask_preview is not None:
            _render_annotation_mask_preview(
                request=mask_preview,
                resolved_masks=resolved_masks,
                image_path=image_path,
                backend=slide.backend_name,
                spacing_at_level_0=spacing_override,
            )
        return _build_per_annotation_tiling_results(
            slide=slide,
            resolved_masks=resolved_masks,
            sampling_spec=sampling_spec,
            selection_strategy=selection_strategy,
            image_path=image_path,
            backend=slide.backend_name,
            requested_backend=requested_backend if requested_backend is not None else backend,
            spacing_at_level_0=spacing_override,
            sample_id=sample_id,
            tiling=tiling,
            filtering=filtering,
            num_workers=num_workers,
            output_mode=output_mode,
        )
    finally:
        slide.close()


__all__ = [
    "MaskPreviewRequest",
    "build_per_annotation_tiling_results",
    "build_tiling_result_from_mask",
    "preprocess_slide",
    "preprocess_slide_per_annotation",
]
