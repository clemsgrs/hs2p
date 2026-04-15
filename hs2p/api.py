import csv
import io
import multiprocessing as mp
import tarfile
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from hs2p.configs import (
    FilterConfig,
    PreviewConfig,
    SegmentationConfig,
    TilingConfig,
)
from hs2p.progress import emit_progress, emit_progress_log
from hs2p.tile_qc import filter_coordinate_tiles, needs_pixel_qc
from hs2p.wsi import (
    CoordinateOutputMode,
    CoordinateSelectionStrategy,
    build_palette,
    iter_tile_records_from_result,
    normalize_tissue_mask,
    open_reader_for_result,
    overlay_mask_on_slide as _overlay_mask_on_slide,
    save_overlay_preview,
    write_coordinate_preview,
)
from hs2p.wsi.backend import resolve_backend
from hs2p.wsi.reader import BatchRegionReader
from hs2p.preprocessing import (
    ResolvedTissueMask,
    TilingResult,
    build_tiling_result_from_mask,
    preprocess_slide,
    resolve_tissue_mask,
    open_slide as open_preprocessing_slide,
)
from hs2p.artifacts import (
    CompatibilitySpec,
    SlideSpec,
    TilingArtifacts,
    load_tiling_result,
    maybe_load_existing_artifacts,
    optional_path,
    save_tiling_result,
    validate_required_columns,
    validate_tiling_artifacts,
    write_process_list,
)


def _write_mask_preview(
    *,
    wsi_path: Path,
    backend: str,
    mask_preview_path: Path | None,
    tissue_mask: np.ndarray | None,
    downsample: int,
    tile_size_lv0: int,
    mask_overlay_color: tuple[int, int, int],
    mask_overlay_alpha: float,
) -> None:
    if mask_preview_path is None or tissue_mask is None:
        return
    pixel_mapping = {"background": 0, "tissue": 1}
    color_mapping = {
        "background": None,
        "tissue": list(mask_overlay_color),
    }
    palette = build_palette(
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )
    save_overlay_preview(
        wsi_path=wsi_path,
        backend=backend,
        mask_arr=normalize_tissue_mask(np.asarray(tissue_mask)),
        mask_preview_path=mask_preview_path,
        downsample=downsample,
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        alpha=mask_overlay_alpha,
        tile_size_lv0=tile_size_lv0,
    )


def _resolve_effective_segmentation(
    *,
    whole_slide: SlideSpec,
    segmentation: SegmentationConfig | None,
) -> SegmentationConfig:
    if segmentation is not None:
        return segmentation
    if whole_slide.mask_path is None:
        raise ValueError(
            "segmentation config is required when no precomputed tissue mask is provided"
        )
    return SegmentationConfig(method="precomputed_mask")


def _validate_segmentation_requirement(
    *,
    whole_slides: Sequence[SlideSpec],
    segmentation: SegmentationConfig | None,
) -> None:
    if segmentation is not None:
        return
    missing_masks = [
        slide.sample_id for slide in whole_slides if slide.mask_path is None
    ]
    if missing_masks:
        raise ValueError(
            "segmentation config is required for slides without a precomputed mask: "
            + ", ".join(missing_masks)
        )


def _compute_tiling_result(
    whole_slide: SlideSpec,
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig | None,
    filtering: FilterConfig,
    mask_preview_path: Path | None,
    preview_downsample: int = 32,
    mask_overlay_color: tuple[int, int, int] = (157, 219, 129),
    mask_overlay_alpha: float = 0.5,
    num_workers: int,
    resolved_mask: ResolvedTissueMask | None = None,
) -> TilingResult:
    has_mask = whole_slide.mask_path is not None
    effective_segmentation = _resolve_effective_segmentation(
        whole_slide=whole_slide,
        segmentation=segmentation,
    )
    if resolved_mask is None:
        return preprocess_slide(
            image_path=whole_slide.image_path,
            sample_id=whole_slide.sample_id,
            tissue_mask_path=whole_slide.mask_path,
            backend=tiling.backend,
            spacing_override=whole_slide.spacing_at_level_0,
            requested_tile_size_px=tiling.requested_tile_size_px,
            requested_spacing_um=tiling.requested_spacing_um,
            tissue_method=effective_segmentation.method,
            sthresh=effective_segmentation.sthresh,
            sthresh_up=effective_segmentation.sthresh_up,
            mthresh=effective_segmentation.mthresh,
            close=effective_segmentation.close,
            sam2_checkpoint_path=effective_segmentation.sam2_checkpoint_path,
            sam2_config_path=effective_segmentation.sam2_config_path,
            sam2_device=effective_segmentation.sam2_device,
            min_tissue_fraction=tiling.tissue_threshold,
            overlap=tiling.overlap,
            seg_downsample=effective_segmentation.downsample,
            tolerance=tiling.tolerance,
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
            num_workers=num_workers,
            annotation=None,
            selection_strategy=(
                CoordinateSelectionStrategy.MERGED_DEFAULT_TILING if has_mask else None
            ),
            output_mode=(
                CoordinateOutputMode.SINGLE_OUTPUT if has_mask else None
            ),
        )

    slide = open_preprocessing_slide(
        whole_slide.image_path,
        backend=tiling.backend,
        spacing_override=whole_slide.spacing_at_level_0,
    )
    try:
        preprocessing_result = build_tiling_result_from_mask(
            slide=slide,
            resolved_mask=resolved_mask,
            image_path=whole_slide.image_path,
            backend=slide.backend_name,
            requested_backend=tiling.requested_backend,
            sample_id=whole_slide.sample_id,
            requested_tile_size_px=tiling.requested_tile_size_px,
            requested_spacing_um=tiling.requested_spacing_um,
            min_tissue_fraction=tiling.tissue_threshold,
            overlap=tiling.overlap,
            tolerance=tiling.tolerance,
            seg_sthresh=effective_segmentation.sthresh,
            seg_sthresh_up=effective_segmentation.sthresh_up,
            seg_mthresh=effective_segmentation.mthresh,
            seg_close=effective_segmentation.close,
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
            num_workers=num_workers,
            selection_strategy=(
                CoordinateSelectionStrategy.MERGED_DEFAULT_TILING if has_mask else None
            ),
            output_mode=(
                CoordinateOutputMode.SINGLE_OUTPUT if has_mask else None
            ),
        )
        _write_mask_preview(
            wsi_path=preprocessing_result.image_path,
            backend=preprocessing_result.backend,
            mask_preview_path=mask_preview_path,
            tissue_mask=preprocessing_result.tissue_mask,
            downsample=preview_downsample,
            tile_size_lv0=preprocessing_result.tile_size_lv0,
            mask_overlay_color=mask_overlay_color,
            mask_overlay_alpha=mask_overlay_alpha,
        )
        return preprocessing_result
    finally:
        slide.close()


def tile_slide(
    whole_slide: SlideSpec,
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig | None = None,
    filtering: FilterConfig = FilterConfig(),
    num_workers: int = 1,
) -> TilingResult:
    backend_selection = resolve_backend(
        tiling.requested_backend,
        wsi_path=whole_slide.image_path,
        mask_path=whole_slide.mask_path,
    )
    if backend_selection.reason is not None:
        emit_progress(
            "backend.selected",
            sample_id=whole_slide.sample_id,
            backend=backend_selection.backend,
            reason=backend_selection.reason,
        )
    effective_tiling = (
        tiling
    if backend_selection.backend == tiling.requested_backend
    else replace(tiling, backend=backend_selection.backend)
    )
    effective_segmentation = _resolve_effective_segmentation(
        whole_slide=whole_slide,
        segmentation=segmentation,
    )
    return preprocess_slide(
        image_path=whole_slide.image_path,
        sample_id=whole_slide.sample_id,
        tissue_mask_path=whole_slide.mask_path,
        backend=effective_tiling.backend,
        spacing_override=whole_slide.spacing_at_level_0,
        requested_tile_size_px=effective_tiling.requested_tile_size_px,
        requested_spacing_um=effective_tiling.requested_spacing_um,
        tissue_method=effective_segmentation.method,
        sthresh=effective_segmentation.sthresh,
        sthresh_up=effective_segmentation.sthresh_up,
        mthresh=effective_segmentation.mthresh,
        close=effective_segmentation.close,
        sam2_checkpoint_path=effective_segmentation.sam2_checkpoint_path,
        sam2_config_path=effective_segmentation.sam2_config_path,
        sam2_device=effective_segmentation.sam2_device,
        min_tissue_fraction=effective_tiling.tissue_threshold,
        overlap=effective_tiling.overlap,
        seg_downsample=effective_segmentation.downsample,
        tolerance=effective_tiling.tolerance,
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
        num_workers=num_workers,
        selection_strategy=(
            CoordinateSelectionStrategy.MERGED_DEFAULT_TILING
            if whole_slide.mask_path is not None
            else None
        ),
        output_mode=(
            CoordinateOutputMode.SINGLE_OUTPUT
            if whole_slide.mask_path is not None
            else None
        ),
    )


def extract_tiles_to_tar(
    result: TilingResult,
    output_dir: Path,
    *,
    jpeg_quality: int = 90,
    jpeg_backend: str = "turbojpeg",
    supertile_sizes: Sequence[int] | None = None,
    tiles_dir: Path | None = None,
    filter_params: FilterConfig | None = None,
    num_workers: int = 4,
    gpu_decode: bool = False,
    phase_recorder: Any | None = None,
) -> tuple[Path, TilingResult]:
    """Extract tile images from a WSI and save them as a JPEG tar archive.

    When *filter_params* requests pixel QC, candidate coordinates are filtered
    first at the configured QC spacing so rejected tiles are never read at the
    final extraction spacing. The returned ``TilingResult`` has its coordinate
    arrays trimmed to the surviving tiles.
    """
    jpeg_backend = str(jpeg_backend)
    _jpeg_encoder = None
    if jpeg_backend == "turbojpeg":
        import turbojpeg

        _jpeg_encoder = turbojpeg.TurboJPEG()

    tiles_dir = Path(tiles_dir) if tiles_dir is not None else Path(output_dir) / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tar_path = tiles_dir / f"{result.sample_id}.tiles.tar"
    manifest_path = tiles_dir / f"{result.sample_id}.tiles.manifest.csv"

    if filter_params is not None and needs_pixel_qc(filter_params):
        result = _apply_qc_filtering_to_result(
            result=result,
            filter_params=filter_params,
            num_workers=num_workers,
            gpu_decode=gpu_decode,
        )

    kept_indices: list[int] = []

    temp_tar_path: Path | None = None
    temp_manifest_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".tar", dir=tiles_dir, delete=False
        ) as tmp:
            temp_tar_path = Path(tmp.name)
        with tempfile.NamedTemporaryFile(
            suffix=".manifest.csv", dir=tiles_dir, delete=False, mode="w", newline=""
        ) as tmp_manifest:
            temp_manifest_path = Path(tmp_manifest.name)

        with (
            tarfile.open(temp_tar_path, "w") as tf,
            temp_manifest_path.open("w", newline="") as manifest_handle,
        ):
            manifest_writer = csv.DictWriter(
                manifest_handle,
                fieldnames=["tile_index", "x", "y"],
            )
            manifest_writer.writeheader()
            tile_records = iter_tile_records_from_result(
                result=result,
                num_workers=num_workers,
                gpu_decode=gpu_decode,
                supertile_sizes=supertile_sizes,
            )
            while True:
                read_start = time.perf_counter()
                try:
                    record = next(tile_records)
                except StopIteration:
                    break
                read_duration = time.perf_counter() - read_start
                if phase_recorder is not None:
                    phase_recorder.record("read", read_duration, tile_count=1)
                tile_arr = record.tile_arr
                if tile_arr.shape[2] > 3:
                    tile_arr = tile_arr[:, :, :3]

                encode_start = time.perf_counter()
                if result.read_tile_size_px != result.requested_tile_size_px:
                    img = Image.fromarray(tile_arr).convert("RGB")
                    img = img.resize(
                        (
                            result.requested_tile_size_px,
                            result.requested_tile_size_px,
                        ),
                        resample=Image.Resampling.BILINEAR,
                    )
                    tile_arr = np.asarray(img)

                if jpeg_backend == "turbojpeg":
                    assert _jpeg_encoder is not None
                    jpeg_bytes = _jpeg_encoder.encode(
                        tile_arr,
                        quality=jpeg_quality,
                        pixel_format=turbojpeg.TJPF_RGB,
                        jpeg_subsample=turbojpeg.TJSAMP_420,
                    )
                elif jpeg_backend == "pil":
                    img = Image.fromarray(tile_arr).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=jpeg_quality)
                    jpeg_bytes = buf.getvalue()
                else:
                    raise ValueError(f"Unsupported jpeg_backend: {jpeg_backend}")
                buf = io.BytesIO(jpeg_bytes)
                encode_duration = time.perf_counter() - encode_start
                if phase_recorder is not None:
                    phase_recorder.record(
                        "encode",
                        encode_duration,
                        tile_count=1,
                        jpeg_bytes=len(jpeg_bytes),
                    )

                tile_index = int(result.tile_index[record.tile_index])
                write_start = time.perf_counter()
                info = tarfile.TarInfo(name=_format_tar_member_name(tile_index))
                info.size = len(jpeg_bytes)
                tf.addfile(info, buf)
                write_duration = time.perf_counter() - write_start
                if phase_recorder is not None:
                    phase_recorder.record("write", write_duration, tile_count=1)

                manifest_writer.writerow(
                    {
                        "tile_index": tile_index,
                        "x": int(record.x),
                        "y": int(record.y),
                    }
                )
                kept_indices.append(record.tile_index)

        temp_tar_path.replace(tar_path)
        temp_tar_path = None
        temp_manifest_path.replace(manifest_path)
        temp_manifest_path = None
    finally:
        if temp_tar_path is not None:
            temp_tar_path.unlink(missing_ok=True)
        if temp_manifest_path is not None:
            temp_manifest_path.unlink(missing_ok=True)

    if len(kept_indices) == len(result.x):
        return tar_path, result

    kept = np.asarray(sorted(kept_indices), dtype=np.int64)
    filtered_result = replace(
        result,
        tiles=replace(
            result.tiles,
            x=result.x[kept],
            y=result.y[kept],
            tissue_fractions=result.tissue_fractions[kept],
            tile_index=np.arange(len(kept), dtype=np.int32),
        ),
    )
    return tar_path, filtered_result

def _format_tar_member_name(tile_index: int) -> str:
    return f"{int(tile_index):06d}.jpg"


def _needs_pixel_filtering(filtering: FilterConfig) -> bool:
    return needs_pixel_qc(filtering)


def _apply_qc_filtering_to_result(
    *,
    result: TilingResult,
    filter_params: FilterConfig,
    num_workers: int,
    gpu_decode: bool = False,
) -> TilingResult:
    with open_reader_for_result(
        result,
        gpu_decode=gpu_decode,
    ) as slide:
        batch_read_windows = None
        if isinstance(slide, BatchRegionReader):
            batch_read_windows = (
                lambda locations, size, level, workers: slide.read_regions(
                    locations,
                    level,
                    size,
                    num_workers=workers,
                )
            )
        keep_flags = filter_coordinate_tiles(
            coord_candidates=np.column_stack((result.x, result.y)),
            keep_flags=np.ones(len(result.x), dtype=np.uint8),
            level_dimensions=slide.level_dimensions,
            level_downsamples=slide.level_downsamples,
            requested_tile_size_px=result.requested_tile_size_px,
            requested_spacing_um=result.requested_spacing_um,
            base_spacing_um=result.base_spacing_um,
            tolerance=result.tolerance,
            filter_params=filter_params,
            read_window=lambda x, y, width, height, level: slide.read_region(
                (x, y),
                level,
                (width, height),
            ),
            batch_read_windows=batch_read_windows,
            num_workers=num_workers,
            source_label=str(result.image_path),
        )
    keep = np.asarray(keep_flags, dtype=bool)
    if int(keep.sum()) == len(result.x):
        return replace(
            result,
            filter_white=filter_params.filter_white,
            filter_black=filter_params.filter_black,
            white_threshold=filter_params.white_threshold,
            black_threshold=filter_params.black_threshold,
            fraction_threshold=filter_params.fraction_threshold,
            filter_grayspace=filter_params.filter_grayspace,
            grayspace_saturation_threshold=filter_params.grayspace_saturation_threshold,
            grayspace_fraction_threshold=filter_params.grayspace_fraction_threshold,
            filter_blur=filter_params.filter_blur,
            blur_threshold=filter_params.blur_threshold,
            qc_spacing_um=filter_params.qc_spacing_um,
        )

    return replace(
        result,
        tiles=replace(
            result.tiles,
            x=result.x[keep],
            y=result.y[keep],
            tissue_fractions=result.tissue_fractions[keep],
            tile_index=np.arange(int(keep.sum()), dtype=np.int32),
        ),
        filter_white=filter_params.filter_white,
        filter_black=filter_params.filter_black,
        white_threshold=filter_params.white_threshold,
        black_threshold=filter_params.black_threshold,
        fraction_threshold=filter_params.fraction_threshold,
        filter_grayspace=filter_params.filter_grayspace,
        grayspace_saturation_threshold=filter_params.grayspace_saturation_threshold,
        grayspace_fraction_threshold=filter_params.grayspace_fraction_threshold,
        filter_blur=filter_params.filter_blur,
        blur_threshold=filter_params.blur_threshold,
        qc_spacing_um=filter_params.qc_spacing_um,
    )


def _validate_whole_slides(whole_slides: Sequence[SlideSpec]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for whole_slide in whole_slides:
        if whole_slide.sample_id in seen:
            duplicates.append(whole_slide.sample_id)
        seen.add(whole_slide.sample_id)
    if duplicates:
        duplicate_text = ", ".join(sorted(set(duplicates)))
        raise ValueError(
            f"Duplicate sample_id values are not allowed: {duplicate_text}"
        )


def write_tiling_preview(
    *,
    result: TilingResult,
    output_dir: Path,
    downsample: int,
) -> Path | None:
    """Render a tiling preview image for a previously computed result.

    Args:
        result: Tiling coordinates and read metadata for one slide.
        output_dir: Root output directory where ``preview/tiling`` is created.
        downsample: Preview downsample passed to ``write_coordinate_preview``.

    Returns:
        Path to the rendered preview image, or ``None`` when there are no tiles.
    """
    if len(result.x) == 0:
        return None
    save_dir = output_dir / "preview" / "tiling"
    save_dir.mkdir(parents=True, exist_ok=True)
    coordinates = list(zip(result.x.tolist(), result.y.tolist()))
    write_coordinate_preview(
        wsi_path=result.image_path,
        coordinates=coordinates,
        tile_size_lv0=result.tile_size_lv0,
        save_dir=save_dir,
        downsample=downsample,
        backend=result.backend,
        sample_id=result.sample_id,
    )
    return save_dir / f"{result.sample_id}.jpg"


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
):
    """Render a mask overlay preview for a slide.

    This is the public API counterpart to the batch QC preview written by
    ``tile_slides(..., preview=PreviewConfig(save_mask_preview=True, ...))``. It can
    overlay either a mask file from disk or an in-memory mask array.
    """

    return _overlay_mask_on_slide(
        wsi_path=wsi_path,
        annotation_mask_path=annotation_mask_path,
        downsample=downsample,
        backend=backend,
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        alpha=alpha,
        mask_arr=mask_arr,
    )


@dataclass
class _PendingPreview:
    whole_slide: SlideSpec
    base_artifact: TilingArtifacts
    mask_preview_path: Path | None
    future: Any


@dataclass(frozen=True)
class _SlideWork:
    whole_slide: SlideSpec
    artifact: TilingArtifacts | None = None
    compute_request: Any | None = None
    error: str | None = None
    traceback_text: str | None = None


@dataclass(frozen=True)
class _ComputeRequest:
    input_index: int
    whole_slide: SlideSpec
    tiling: TilingConfig
    segmentation: SegmentationConfig | None
    resolved_mask: ResolvedTissueMask | None
    filtering: FilterConfig
    mask_preview_path: Path | None
    output_dir: Path
    num_workers: int
    preview_downsample: int = 32
    mask_overlay_color: tuple[int, int, int] = (157, 219, 129)
    mask_overlay_alpha: float = 0.5
    jpeg_backend: str = "turbojpeg"
    gpu_decode: bool = False
    include_result: bool = False
    save_tiles: bool = False


@dataclass(frozen=True)
class _ComputeResponse:
    input_index: int
    whole_slide: SlideSpec
    ok: bool
    artifact: TilingArtifacts | None = None
    result: TilingResult | None = None
    requested_backend: str | None = None
    backend: str | None = None
    mask_preview_path: Path | None = None
    error: str | None = None
    traceback_text: str | None = None


@dataclass(frozen=True)
class _MaskResolutionRequest:
    input_index: int
    whole_slide: SlideSpec
    tiling: TilingConfig
    segmentation: SegmentationConfig | None


@dataclass(frozen=True)
class _MaskResolutionResponse:
    input_index: int
    whole_slide: SlideSpec
    ok: bool
    resolved_mask: ResolvedTissueMask | None = None
    requested_backend: str | None = None
    backend: str | None = None
    error: str | None = None
    traceback_text: str | None = None


def _build_success_artifact(
    *,
    base_artifact: TilingArtifacts,
    mask_preview_path: Path | None,
    tiling_preview_path: Path | None,
) -> TilingArtifacts:
    return TilingArtifacts(
        sample_id=base_artifact.sample_id,
        coordinates_npz_path=base_artifact.coordinates_npz_path,
        coordinates_meta_path=base_artifact.coordinates_meta_path,
        num_tiles=base_artifact.num_tiles,
        tiles_tar_path=base_artifact.tiles_tar_path,
        mask_preview_path=mask_preview_path,
        tiling_preview_path=tiling_preview_path,
        backend=base_artifact.backend,
        requested_backend=base_artifact.requested_backend,
    )


def _build_success_process_row(
    *,
    whole_slide: SlideSpec,
    artifact: TilingArtifacts,
) -> dict[str, Any]:
    return {
        "sample_id": whole_slide.sample_id,
        "image_path": str(whole_slide.image_path),
        "mask_path": (
            str(whole_slide.mask_path) if whole_slide.mask_path is not None else None
        ),
        "requested_backend": artifact.requested_backend,
        "backend": artifact.backend,
        "tiling_status": "success",
        "num_tiles": artifact.num_tiles,
        "coordinates_npz_path": str(artifact.coordinates_npz_path) if artifact.coordinates_npz_path is not None else np.nan,
        "coordinates_meta_path": str(artifact.coordinates_meta_path),
        "tiles_tar_path": str(artifact.tiles_tar_path) if artifact.tiles_tar_path is not None else np.nan,
        "error": np.nan,
        "traceback": np.nan,
    }


def _build_failure_process_row(
    *,
    whole_slide: SlideSpec,
    error: str,
    traceback_text: str,
    requested_backend: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": whole_slide.sample_id,
        "image_path": str(whole_slide.image_path),
        "mask_path": (
            str(whole_slide.mask_path) if whole_slide.mask_path is not None else None
        ),
        "requested_backend": requested_backend,
        "backend": backend,
        "tiling_status": "failed",
        "num_tiles": 0,
        "coordinates_npz_path": np.nan,
        "coordinates_meta_path": np.nan,
        "tiles_tar_path": np.nan,
        "error": error,
        "traceback": traceback_text,
    }


def _finalize_pending_tiling_preview(
    *,
    pending: _PendingPreview,
) -> tuple[TilingArtifacts | None, dict[str, Any]]:
    tiling_preview_path = pending.future.result()
    tiling_preview_path = (
        tiling_preview_path
        if tiling_preview_path is not None and tiling_preview_path.is_file()
        else None
    )
    artifact = _build_success_artifact(
        base_artifact=pending.base_artifact,
        mask_preview_path=pending.mask_preview_path,
        tiling_preview_path=tiling_preview_path,
    )
    row = _build_success_process_row(
        whole_slide=pending.whole_slide,
        artifact=artifact,
    )
    return artifact, row


def _compute_and_save_tiling_artifacts_from_request(
    request: _ComputeRequest,
) -> _ComputeResponse:
    try:
        defer_pixel_filtering = request.save_tiles and _needs_pixel_filtering(request.filtering)
        effective_filtering = (
            replace(
                request.filtering,
                filter_white=False,
                filter_black=False,
                filter_grayspace=False,
                filter_blur=False,
            )
            if defer_pixel_filtering
            else request.filtering
        )
        effective_tiling = request.tiling
        result = _compute_tiling_result(
            request.whole_slide,
            tiling=effective_tiling,
            segmentation=request.segmentation,
            filtering=effective_filtering,
            mask_preview_path=request.mask_preview_path,
            preview_downsample=request.preview_downsample,
            mask_overlay_color=request.mask_overlay_color,
            mask_overlay_alpha=request.mask_overlay_alpha,
            num_workers=request.num_workers,
            resolved_mask=request.resolved_mask,
        )
        tiles_tar_path: Path | None = None
        if request.save_tiles:
            tiles_tar_path, result = extract_tiles_to_tar(
                result,
                output_dir=request.output_dir,
                jpeg_backend=request.jpeg_backend,
                filter_params=request.filtering if _needs_pixel_filtering(request.filtering) else None,
                num_workers=request.num_workers,
                gpu_decode=request.gpu_decode,
            )
        artifact = save_tiling_result(result, output_dir=request.output_dir)
        artifact = TilingArtifacts(
            sample_id=artifact.sample_id,
            coordinates_npz_path=artifact.coordinates_npz_path,
            coordinates_meta_path=artifact.coordinates_meta_path,
            num_tiles=artifact.num_tiles,
            tiles_tar_path=tiles_tar_path,
            mask_preview_path=artifact.mask_preview_path,
            tiling_preview_path=artifact.tiling_preview_path,
            backend=artifact.backend,
            requested_backend=artifact.requested_backend,
        )
        mask_preview_path = (
            request.mask_preview_path
            if request.mask_preview_path is not None
            and request.mask_preview_path.is_file()
            else None
        )
        return _ComputeResponse(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=True,
            artifact=artifact,
            result=result if request.include_result else None,
            requested_backend=request.tiling.requested_backend,
            backend=effective_tiling.backend,
            mask_preview_path=mask_preview_path,
        )
    except Exception as exc:
        return _ComputeResponse(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=False,
            requested_backend=request.tiling.requested_backend,
            backend=request.tiling.backend,
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )


def _resolve_mask_for_request(
    request: _MaskResolutionRequest,
) -> _MaskResolutionResponse:
    try:
        effective_tiling = request.tiling
        backend_selection = resolve_backend(
            effective_tiling.backend,
            wsi_path=request.whole_slide.image_path,
            mask_path=request.whole_slide.mask_path,
        )
        effective_tiling = (
            effective_tiling
            if backend_selection.backend == effective_tiling.requested_backend
            else replace(effective_tiling, backend=backend_selection.backend)
        )
        slide = open_preprocessing_slide(
            request.whole_slide.image_path,
            backend=effective_tiling.backend,
            spacing_override=request.whole_slide.spacing_at_level_0,
        )
        try:
            effective_segmentation = _resolve_effective_segmentation(
                whole_slide=request.whole_slide,
                segmentation=request.segmentation,
            )
            resolved_mask = resolve_tissue_mask(
                slide=slide,
                tissue_method=effective_segmentation.method,
                tissue_mask_path=request.whole_slide.mask_path,
                tissue_mask_tissue_value=1,
                sthresh=effective_segmentation.sthresh,
                sthresh_up=effective_segmentation.sthresh_up,
                mthresh=effective_segmentation.mthresh,
                close=effective_segmentation.close,
                seg_downsample=effective_segmentation.downsample,
                sam2_checkpoint_path=effective_segmentation.sam2_checkpoint_path,
                sam2_config_path=effective_segmentation.sam2_config_path,
                sam2_device=effective_segmentation.sam2_device,
            )
        finally:
            slide.close()
        return _MaskResolutionResponse(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=True,
            resolved_mask=resolved_mask,
            requested_backend=request.tiling.requested_backend,
            backend=effective_tiling.backend,
        )
    except Exception as exc:
        return _MaskResolutionResponse(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=False,
            requested_backend=request.tiling.requested_backend,
            backend=request.tiling.backend,
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )


def _resolve_tiling_worker_allocation(
    *, num_workers: int, compute_count: int
) -> tuple[bool, int, int]:
    total_workers = max(1, int(num_workers))
    pending_compute = max(0, int(compute_count))
    use_slide_pool = total_workers > 1 and pending_compute > 1
    if not use_slide_pool:
        return False, 1, total_workers
    outer_workers = min(total_workers, pending_compute)
    inner_workers = max(1, total_workers // outer_workers)
    return True, outer_workers, inner_workers


def _write_tiling_preview_from_artifacts(
    *,
    artifact: TilingArtifacts,
    output_dir: Path,
    downsample: int,
) -> Path | None:
    result = load_tiling_result(artifact.coordinates_npz_path, artifact.coordinates_meta_path)
    return write_tiling_preview(
        result=result,
        output_dir=output_dir,
        downsample=downsample,
    )


def tile_slides(
    whole_slides: Sequence[SlideSpec],
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig | None = None,
    filtering: FilterConfig = FilterConfig(),
    preview: PreviewConfig | None = None,
    output_dir: Path,
    num_workers: int = 1,
    resume: bool = False,
    read_coordinates_from: Path | None = None,
    save_tiles: bool = False,
    jpeg_backend: str = "turbojpeg",
    gpu_decode: bool = False,
) -> list[TilingArtifacts]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_whole_slides(whole_slides)
    _validate_segmentation_requirement(
        whole_slides=whole_slides,
        segmentation=segmentation,
    )
    artifacts: list[TilingArtifacts] = []
    process_rows: list[dict[str, Any]] = []
    process_list_path = output_dir / "process_list.csv"
    existing_successes: dict[str, dict[str, Any]] = {}
    if resume and process_list_path.is_file():
        existing_df = pd.read_csv(process_list_path)
        legacy_mask_columns = [
            column
            for column in ("tissue_mask_path", "annotation_mask_path")
            if column in existing_df.columns
        ]
        if legacy_mask_columns:
            raise ValueError(
                "Unsupported tiling process_list.csv schema in "
                f"{process_list_path}; deprecated mask columns present: "
                + ", ".join(sorted(legacy_mask_columns))
            )
        validate_required_columns(
            existing_df,
            required_columns={
                "sample_id",
                "image_path",
                "mask_path",
                "requested_backend",
                "backend",
                "tiling_status",
                "num_tiles",
                "coordinates_npz_path",
                "coordinates_meta_path",
                "error",
                "traceback",
            },
            file_path=process_list_path,
            file_label="tiling process_list.csv",
        )
        for row in existing_df.to_dict(orient="records"):
            if row.get("tiling_status") == "success":
                existing_successes[str(row["sample_id"])] = row
    compatibility_specs: dict[tuple[bool, str], CompatibilitySpec] = {}

    def _resolve_effective_tiling(whole_slide: SlideSpec) -> TilingConfig:
        backend_selection = resolve_backend(
            tiling.backend,
            wsi_path=whole_slide.image_path,
            mask_path=whole_slide.mask_path,
        )
        if backend_selection.reason is not None:
            emit_progress(
                "backend.selected",
                sample_id=whole_slide.sample_id,
                backend=backend_selection.backend,
                reason=backend_selection.reason,
            )
        return (
            tiling
            if backend_selection.backend == tiling.requested_backend
            else replace(tiling, backend=backend_selection.backend)
        )

    planned_work: list[_SlideWork] = []
    compute_requests: list[_ComputeRequest] = []
    for whole_slide in whole_slides:
        try:
            effective_tiling = _resolve_effective_tiling(whole_slide)
            key = (whole_slide.mask_path is not None, effective_tiling.backend)
            if key not in compatibility_specs:
                has_mask = whole_slide.mask_path is not None
                compatibility_specs[key] = CompatibilitySpec(
                    tiling=effective_tiling,
                    segmentation=_resolve_effective_segmentation(
                        whole_slide=whole_slide,
                        segmentation=segmentation,
                    ),
                    filtering=filtering,
                    selection_strategy=(
                        CoordinateSelectionStrategy.MERGED_DEFAULT_TILING if has_mask else None
                    ),
                    output_mode=(
                        CoordinateOutputMode.SINGLE_OUTPUT if has_mask else None
                    ),
                )
            compatibility = compatibility_specs[key]
            artifact: TilingArtifacts | None = None
            if whole_slide.sample_id in existing_successes:
                row = existing_successes[whole_slide.sample_id]
                npz_path = optional_path(row["coordinates_npz_path"])
                meta_path = Path(str(row["coordinates_meta_path"]))
                artifact = validate_tiling_artifacts(
                    whole_slide=whole_slide,
                    coordinates_npz_path=npz_path,
                    coordinates_meta_path=meta_path,
                    compatibility=compatibility,
                )
            if read_coordinates_from is not None and artifact is None:
                artifact = maybe_load_existing_artifacts(
                    whole_slide=whole_slide,
                    read_coordinates_from=Path(read_coordinates_from),
                    compatibility=compatibility,
                )
            if artifact is not None:
                planned_work.append(
                    _SlideWork(
                        whole_slide=whole_slide,
                        artifact=artifact,
                    )
                )
                continue
            mask_preview_path = None
            if preview is not None and preview.save_mask_preview:
                mask_dir = output_dir / "preview" / "mask"
                mask_preview_path = mask_dir / f"{whole_slide.sample_id}.jpg"
            compute_request = _ComputeRequest(
                input_index=len(planned_work),
                whole_slide=whole_slide,
                tiling=effective_tiling,
                segmentation=segmentation,
                resolved_mask=None,
                filtering=filtering,
                mask_preview_path=mask_preview_path,
                preview_downsample=(
                    preview.downsample if preview is not None else 32
                ),
                mask_overlay_color=(
                    preview.mask_overlay_color
                    if preview is not None
                    else (157, 219, 129)
                ),
                mask_overlay_alpha=(
                    preview.mask_overlay_alpha if preview is not None else 0.5
                ),
                output_dir=output_dir,
                num_workers=1,
                jpeg_backend=str(jpeg_backend),
                gpu_decode=gpu_decode,
                save_tiles=save_tiles,
            )
            planned_work.append(
                _SlideWork(
                    whole_slide=whole_slide,
                    compute_request=compute_request,
                )
            )
            compute_requests.append(compute_request)
        except Exception as exc:
            planned_work.append(
                _SlideWork(
                    whole_slide=whole_slide,
                    error=str(exc),
                    traceback_text=traceback.format_exc(),
                )
            )
    mask_resolution_requests = [
        _MaskResolutionRequest(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            tiling=request.tiling,
            segmentation=request.segmentation,
        )
        for request in compute_requests
    ]
    resolved_masks: dict[int, ResolvedTissueMask] = {}
    mask_failures: dict[int, _MaskResolutionResponse] = {}
    if mask_resolution_requests:
        emit_progress("tissue.started", total=len(mask_resolution_requests))
        use_mask_pool, mask_outer_workers, _ = _resolve_tiling_worker_allocation(
            num_workers=num_workers,
            compute_count=len(mask_resolution_requests),
        )
        def _resolve_mask_response(request: _MaskResolutionRequest) -> _MaskResolutionResponse:
            try:
                return _resolve_mask_for_request(request)
            except Exception as exc:
                return _MaskResolutionResponse(
                    input_index=request.input_index,
                    whole_slide=request.whole_slide,
                    ok=False,
                    requested_backend=request.tiling.requested_backend,
                    backend=request.tiling.backend,
                    error=str(exc),
                    traceback_text=traceback.format_exc(),
                )

        def _record_mask_response(response: _MaskResolutionResponse) -> None:
            if response.ok and response.resolved_mask is not None:
                resolved_masks[response.input_index] = response.resolved_mask
            else:
                mask_failures[response.input_index] = response
            snapshot_completed = len(resolved_masks)
            snapshot_failed = len(mask_failures)
            emit_progress(
                "tissue.progress",
                total=len(mask_resolution_requests),
                completed=snapshot_completed,
                failed=snapshot_failed,
                pending=max(
                    0, len(mask_resolution_requests) - snapshot_completed - snapshot_failed
                ),
            )

        if use_mask_pool:
            with ThreadPoolExecutor(max_workers=mask_outer_workers) as pool:
                futures = [
                    pool.submit(_resolve_mask_response, request)
                    for request in mask_resolution_requests
                ]
                for future in as_completed(futures):
                    _record_mask_response(future.result())
        else:
            for request in mask_resolution_requests:
                _record_mask_response(_resolve_mask_response(request))
        snapshot_completed = len(resolved_masks)
        snapshot_failed = len(mask_failures)
        emit_progress(
            "tissue.finished",
            total=len(mask_resolution_requests),
            completed=snapshot_completed,
            failed=snapshot_failed,
            pending=max(
                0, len(mask_resolution_requests) - snapshot_completed - snapshot_failed
            ),
        )
    compute_requests = [
        replace(request, resolved_mask=resolved_masks.get(request.input_index))
        for request in compute_requests
        if request.input_index not in mask_failures
    ]
    use_slide_pool, pool_processes, worker_inner_workers = _resolve_tiling_worker_allocation(
        num_workers=num_workers,
        compute_count=len(compute_requests),
    )
    preview_executor = (
        ThreadPoolExecutor(max_workers=max(1, pool_processes))
        if preview is not None and preview.save_tiling_preview
        else None
    )
    pending_previews: list[_PendingPreview] = []
    total_slides = len(planned_work)

    def _progress_snapshot() -> dict[str, int]:
        completed = 0
        failed = 0
        discovered_tiles = 0
        zero_tile_successes = 0
        for row in process_rows:
            status = str(row.get("tiling_status", ""))
            num_tiles = int(row.get("num_tiles", 0) or 0)
            if status == "success":
                completed += 1
                discovered_tiles += num_tiles
                if num_tiles == 0:
                    zero_tile_successes += 1
            elif status in {"failed", "error"}:
                failed += 1
        return {
            "total": total_slides,
            "completed": completed,
            "failed": failed,
            "pending": max(0, total_slides - completed - failed),
            "discovered_tiles": discovered_tiles,
            "zero_tile_successes": zero_tile_successes,
        }

    def _record_process_row(row: dict[str, Any]) -> None:
        process_rows.append(row)
        snapshot = _progress_snapshot()
        emit_progress(
            "tiling.progress",
            total=snapshot["total"],
            completed=snapshot["completed"],
            failed=snapshot["failed"],
            pending=snapshot["pending"],
            discovered_tiles=snapshot["discovered_tiles"],
        )

    emit_progress("tiling.started", total=total_slides)

    def _finalize_all_pending_previews() -> None:
        for previous_pending in pending_previews:
            try:
                finalized_artifact, finalized_row = _finalize_pending_tiling_preview(
                    pending=previous_pending
                )
                if finalized_artifact is not None:
                    artifacts.append(finalized_artifact)
                _record_process_row(finalized_row)
            except Exception as exc:
                emit_progress_log(
                    f"[tile_slides] FAILED {previous_pending.whole_slide.sample_id}: {exc}",
                )
                _record_process_row(
                    _build_failure_process_row(
                        whole_slide=previous_pending.whole_slide,
                        error=str(exc),
                        traceback_text=traceback.format_exc(),
                        requested_backend=None,
                        backend=None,
                    )
                )
        pending_previews.clear()

    def _process_compute_response(response: _ComputeResponse) -> None:
        if not response.ok:
            emit_progress_log(
                f"[tile_slides] FAILED {response.whole_slide.sample_id}: {response.error}",
            )
            _record_process_row(
                _build_failure_process_row(
                    whole_slide=response.whole_slide,
                    error=response.error or "unknown error",
                    traceback_text=response.traceback_text or "",
                    requested_backend=response.requested_backend,
                    backend=response.backend,
                )
            )
            return

        assert response.artifact is not None
        base_artifact = response.artifact
        if (
            preview_executor is not None
            and preview is not None
            and preview.save_tiling_preview
            and base_artifact.num_tiles > 0
        ):
            preview_result = getattr(response, "result", None)
            if preview_result is not None:
                future = preview_executor.submit(
                    write_tiling_preview,
                    result=preview_result,
                    output_dir=output_dir,
                    downsample=preview.downsample,
                )
            else:
                future = preview_executor.submit(
                    _write_tiling_preview_from_artifacts,
                    artifact=base_artifact,
                    output_dir=output_dir,
                    downsample=preview.downsample,
                )
            pending_previews.append(_PendingPreview(
                whole_slide=response.whole_slide,
                base_artifact=base_artifact,
                mask_preview_path=response.mask_preview_path,
                future=future,
            ))
            return

        artifact = _build_success_artifact(
            base_artifact=base_artifact,
            mask_preview_path=response.mask_preview_path,
            tiling_preview_path=None,
        )
        artifacts.append(artifact)
        _record_process_row(
            _build_success_process_row(
                whole_slide=response.whole_slide,
                artifact=artifact,
            )
        )

    def _drain_planned_work(compute_response_iter) -> None:
        buffered_responses: dict[int, _ComputeResponse] = {}

        def _await_response(input_index: int) -> _ComputeResponse:
            while input_index not in buffered_responses:
                response = next(compute_response_iter)
                buffered_responses[response.input_index] = response
            return buffered_responses.pop(input_index)

        for planned in planned_work:
            if planned.artifact is not None:
                artifacts.append(planned.artifact)
                _record_process_row(
                    _build_success_process_row(
                        whole_slide=planned.whole_slide,
                        artifact=planned.artifact,
                    )
                )
                continue
            if planned.error is not None:
                emit_progress_log(
                    f"[tile_slides] FAILED {planned.whole_slide.sample_id}: {planned.error}",
                )
                _record_process_row(
                    _build_failure_process_row(
                        whole_slide=planned.whole_slide,
                        error=planned.error,
                        traceback_text=planned.traceback_text or "",
                        requested_backend=None,
                        backend=None,
                    )
                )
                continue
            assert planned.compute_request is not None
            if planned.compute_request.input_index in mask_failures:
                failure = mask_failures[planned.compute_request.input_index]
                emit_progress_log(
                    f"[tile_slides] FAILED {planned.whole_slide.sample_id}: {failure.error}",
                )
                _record_process_row(
                    _build_failure_process_row(
                        whole_slide=planned.whole_slide,
                        error=failure.error or "unknown error",
                        traceback_text=failure.traceback_text or "",
                        requested_backend=failure.requested_backend,
                        backend=failure.backend,
                    )
                )
                continue
            response = _await_response(planned.compute_request.input_index)
            _process_compute_response(response)

    try:
        if use_slide_pool:
            pool_requests = [
                _ComputeRequest(
                    input_index=request.input_index,
                    whole_slide=request.whole_slide,
                    tiling=request.tiling,
                    segmentation=request.segmentation,
                    resolved_mask=request.resolved_mask,
                    filtering=request.filtering,
                    mask_preview_path=request.mask_preview_path,
                    preview_downsample=request.preview_downsample,
                    mask_overlay_color=request.mask_overlay_color,
                    mask_overlay_alpha=request.mask_overlay_alpha,
                    output_dir=request.output_dir,
                    num_workers=worker_inner_workers,
                    jpeg_backend=request.jpeg_backend,
                    gpu_decode=request.gpu_decode,
                    include_result=False,
                    save_tiles=request.save_tiles,
                )
                for request in compute_requests
            ]
            with mp.Pool(processes=pool_processes) as pool:
                _drain_planned_work(
                    iter(
                        pool.imap_unordered(
                            _compute_and_save_tiling_artifacts_from_request,
                            pool_requests,
                        )
                    )
                )
        else:
            serial_requests = [
                _ComputeRequest(
                    input_index=request.input_index,
                    whole_slide=request.whole_slide,
                    tiling=request.tiling,
                    segmentation=request.segmentation,
                    resolved_mask=request.resolved_mask,
                    filtering=request.filtering,
                    mask_preview_path=request.mask_preview_path,
                    preview_downsample=request.preview_downsample,
                    mask_overlay_color=request.mask_overlay_color,
                    mask_overlay_alpha=request.mask_overlay_alpha,
                    output_dir=request.output_dir,
                    num_workers=worker_inner_workers,
                    jpeg_backend=request.jpeg_backend,
                    gpu_decode=request.gpu_decode,
                    include_result=True,
                    save_tiles=request.save_tiles,
                )
                for request in compute_requests
            ]
            _drain_planned_work(
                iter(
                    _compute_and_save_tiling_artifacts_from_request(request)
                    for request in serial_requests
                )
            )
        _finalize_all_pending_previews()
    finally:
        if preview_executor is not None:
            preview_executor.shutdown(wait=True)
    write_process_list(process_rows, process_list_path)
    snapshot = _progress_snapshot()
    emit_progress(
        "tiling.finished",
        total=snapshot["total"],
        completed=snapshot["completed"],
        failed=snapshot["failed"],
        pending=snapshot["pending"],
        discovered_tiles=snapshot["discovered_tiles"],
        output_dir=str(output_dir),
        process_list_path=str(process_list_path),
        zero_tile_successes=snapshot["zero_tile_successes"],
    )
    return artifacts
