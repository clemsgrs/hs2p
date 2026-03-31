from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import seaborn as sns

from hs2p.configs import FilterConfig, SegmentationConfig, TilingConfig
from hs2p.configs.resolvers import resolve_sampling_strategy, validate_color_mapping
from hs2p.preprocessing import TileGeometry, TilingResult
from hs2p.tiling_artifacts import load_tiling_result
from hs2p.wsi import CoordinateExtractionResult
from hs2p.wsi.types import (
    CoordinateOutputMode,
    CoordinateSelectionStrategy,
    ResolvedSamplingSpec,
)


def resolve_sampling_workers(cfg, *, slide_count: int) -> tuple[int, int]:
    if hasattr(cfg.speed, "inner_workers"):
        raise ValueError(
            "cfg.speed.inner_workers is no longer supported; sampling always uses 1 inner worker per slide"
        )

    parallel_workers = min(mp.cpu_count(), int(cfg.speed.num_workers))
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        parallel_workers = min(
            parallel_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        )
    parallel_workers = max(1, parallel_workers)
    outer_workers = max(1, min(parallel_workers, int(slide_count)))
    return outer_workers, 1


def selection_strategy_from_cfg(cfg) -> str:
    return resolve_sampling_strategy(cfg)


def build_sampling_preview_assets(
    resolved_sampling_spec: ResolvedSamplingSpec,
    *,
    save_previews: bool,
):
    if not save_previews:
        return None, None

    preview_palette = np.zeros(shape=768, dtype=int)
    if resolved_sampling_spec.color_mapping is None:
        ncat = len(resolved_sampling_spec.pixel_mapping)
        if ncat <= 10:
            color_palette = sns.color_palette("tab10")[:ncat]
        elif ncat <= 20:
            color_palette = sns.color_palette("tab20")[:ncat]
        else:
            raise ValueError(
                f"Implementation supports up to 20 categories (provided pixel_mapping has {ncat})"
            )
        color_mapping = {
            k: tuple(int(round(255 * x)) for x in color_palette[i])
            for i, k in enumerate(resolved_sampling_spec.pixel_mapping.keys())
        }
    else:
        color_mapping = resolved_sampling_spec.color_mapping
    validate_color_mapping(
        pixel_mapping=resolved_sampling_spec.pixel_mapping,
        color_mapping=color_mapping,
    )
    p = [0] * 3 * len(color_mapping)
    for k, v in resolved_sampling_spec.pixel_mapping.items():
        if color_mapping[k] is not None:
            p[v * 3 : v * 3 + 3] = color_mapping[k]
    preview_palette[0 : len(p)] = np.array(p).astype(int)
    return preview_palette, color_mapping


def save_sampling_coordinates(
    *,
    sample_id: str,
    image_path: Path,
    mask_path: Path | None,
    backend: str,
    cfg: Any,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
    annotation: str,
    coordinates: list[tuple[int, int]],
    extraction: CoordinateExtractionResult,
    resolved_sampling_spec: ResolvedSamplingSpec,
    save_tiling_result,
    selection_strategy: str | None = None,
    output_mode: str = CoordinateOutputMode.PER_ANNOTATION,
):
    if selection_strategy is None:
        selection_strategy = selection_strategy_from_cfg(cfg)
    annotation_threshold = resolved_sampling_spec.tissue_percentage[annotation]
    coordinate_array = np.asarray(coordinates, dtype=np.int64)
    if coordinate_array.size == 0:
        coordinate_array = np.empty((0, 2), dtype=np.int64)
    result = TilingResult(
        tiles=TileGeometry(
            coordinates=coordinate_array,
            tissue_fractions=np.zeros(len(coordinates), dtype=np.float32),
            tile_index=np.arange(len(coordinates), dtype=np.int32),
            requested_tile_size_px=tiling_config.target_tile_size_px,
            requested_spacing_um=tiling_config.target_spacing_um,
            read_level=extraction.read_level,
            effective_tile_size_px=extraction.read_tile_size_px,
            effective_spacing_um=extraction.read_spacing_um,
            tile_size_lv0=extraction.tile_size_lv0,
            is_within_tolerance=True,
            base_spacing_um=extraction.read_spacing_um,
            slide_dimensions=[0, 0],
            level_downsamples=[1.0],
            overlap=tiling_config.overlap,
            min_tissue_fraction=annotation_threshold,
            use_padding=tiling_config.use_padding,
        ),
        sample_id=sample_id,
        image_path=image_path,
        mask_path=mask_path,
        backend=backend,
        requested_backend=backend,
        step_px_lv0=extraction.step_px_lv0,
        tolerance=tiling_config.tolerance,
        tissue_method="unknown",
        seg_downsample=segmentation_config.downsample,
        seg_level=0,
        seg_spacing_um=0.0,
        seg_sthresh=segmentation_config.sthresh,
        seg_sthresh_up=segmentation_config.sthresh_up,
        seg_mthresh=segmentation_config.mthresh,
        seg_close=segmentation_config.close,
        ref_tile_size_px=filter_config.ref_tile_size,
        a_t=filter_config.a_t,
        a_h=filter_config.a_h,
        filter_white=filter_config.filter_white,
        filter_black=filter_config.filter_black,
        white_threshold=filter_config.white_threshold,
        black_threshold=filter_config.black_threshold,
        fraction_threshold=filter_config.fraction_threshold,
        seg_use_otsu=segmentation_config.use_otsu,
        seg_use_hsv=segmentation_config.use_hsv,
        annotation=annotation,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
    )
    annotation_dir = Path(cfg.output_dir, "tiles", annotation)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    return save_tiling_result(
        result, output_dir=cfg.output_dir, tiles_dir=annotation_dir
    )


def build_sampling_process_rows(
    *,
    whole_slides,
    active_annotations: tuple[str, ...],
):
    rows = []
    for slide in whole_slides:
        for annotation in active_annotations:
            rows.append(
                {
                    "sample_id": slide.sample_id,
                    "annotation": annotation,
                    "image_path": str(slide.image_path),
                    "annotation_mask_path": (
                        str(slide.mask_path) if slide.mask_path is not None else None
                    ),
                    "sampling_status": "tbp",
                    "num_tiles": 0,
                    "coordinates_npz_path": np.nan,
                    "coordinates_meta_path": np.nan,
                    "error": np.nan,
                    "traceback": np.nan,
                }
            )
    return rows


def validate_sampling_artifact_row(
    *,
    row: dict[str, object],
    whole_slide,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
    expected_tissue_threshold: float,
    selection_strategy: str,
) -> None:
    if row.get("sampling_status") != "success":
        raise ValueError("sampling row is not successful")
    num_tiles = int(row.get("num_tiles", 0))
    if num_tiles == 0:
        return
    npz_path = Path(str(row["coordinates_npz_path"]))
    meta_path = Path(str(row["coordinates_meta_path"]))
    result = load_tiling_result(npz_path, meta_path)
    if result.sample_id != whole_slide.sample_id:
        raise ValueError("sampling sample_id mismatch")
    if result.image_path != whole_slide.image_path:
        raise ValueError("sampling image_path mismatch")
    if result.mask_path != whole_slide.mask_path:
        raise ValueError("sampling mask_path mismatch")
    if result.backend != tiling_config.backend:
        raise ValueError("sampling backend mismatch")
    if result.requested_spacing_um != tiling_config.target_spacing_um:
        raise ValueError("sampling target_spacing_um mismatch")
    if result.requested_tile_size_px != tiling_config.target_tile_size_px:
        raise ValueError("sampling target_tile_size_px mismatch")
    if result.overlap != tiling_config.overlap:
        raise ValueError("sampling overlap mismatch")
    if result.min_tissue_fraction != expected_tissue_threshold:
        raise ValueError("sampling tissue_threshold mismatch")
    if result.use_padding != tiling_config.use_padding:
        raise ValueError("sampling use_padding mismatch")
    if result.tolerance != tiling_config.tolerance:
        raise ValueError("sampling tolerance mismatch")
    if result.seg_downsample != segmentation_config.downsample:
        raise ValueError("sampling seg_downsample mismatch")
    if result.seg_sthresh != segmentation_config.sthresh:
        raise ValueError("sampling sthresh mismatch")
    if result.seg_sthresh_up != segmentation_config.sthresh_up:
        raise ValueError("sampling sthresh_up mismatch")
    if result.seg_mthresh != segmentation_config.mthresh:
        raise ValueError("sampling mthresh mismatch")
    if result.seg_close != segmentation_config.close:
        raise ValueError("sampling close mismatch")
    if result.seg_use_otsu != segmentation_config.use_otsu:
        raise ValueError("sampling use_otsu mismatch")
    if result.seg_use_hsv != segmentation_config.use_hsv:
        raise ValueError("sampling use_hsv mismatch")
    if result.ref_tile_size_px != filter_config.ref_tile_size:
        raise ValueError("sampling ref_tile_size mismatch")
    if result.a_t != filter_config.a_t:
        raise ValueError("sampling a_t mismatch")
    if result.a_h != filter_config.a_h:
        raise ValueError("sampling a_h mismatch")
    if result.filter_white != filter_config.filter_white:
        raise ValueError("sampling filter_white mismatch")
    if result.filter_black != filter_config.filter_black:
        raise ValueError("sampling filter_black mismatch")
    if result.white_threshold != filter_config.white_threshold:
        raise ValueError("sampling white_threshold mismatch")
    if result.black_threshold != filter_config.black_threshold:
        raise ValueError("sampling black_threshold mismatch")
    if result.fraction_threshold != filter_config.fraction_threshold:
        raise ValueError("sampling fraction_threshold mismatch")
    if result.annotation != row["annotation"]:
        raise ValueError("sampling annotation mismatch")
    if result.selection_strategy != selection_strategy:
        raise ValueError("sampling selection_strategy mismatch")
    if result.output_mode != CoordinateOutputMode.PER_ANNOTATION:
        raise ValueError("sampling output_mode mismatch")


__all__ = [
    "build_sampling_preview_assets",
    "build_sampling_process_rows",
    "resolve_sampling_workers",
    "save_sampling_coordinates",
    "selection_strategy_from_cfg",
    "validate_sampling_artifact_row",
]
