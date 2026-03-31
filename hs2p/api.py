import csv
import importlib
import io
import itertools
import multiprocessing as mp
import tarfile
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from hs2p.configs import (
    FilterConfig,
    PreviewConfig,
    SegmentationConfig,
    TilingConfig,
)
from hs2p.configs.resolvers import build_default_sampling_spec
from hs2p.progress import emit_progress, emit_progress_log
from hs2p.wsi import (
    CoordinateOutputMode,
    CoordinateSelectionStrategy,
    extract_coordinates,
    overlay_mask_on_slide as _overlay_mask_on_slide,
    write_coordinate_preview,
)
from hs2p.wsi.backend import coerce_wsd_path, resolve_backend
from hs2p.wsi.read_plans import (
    GroupedReadPlan,
    iter_grouped_read_plans,
    resolve_read_step_px,
    resolve_step_px_lv0,
)
from hs2p.wsi.region_tiles import iter_plan_region_tile_views
from hs2p.preprocessing import (
    TilingResult,
    preprocess_slide,
    save_tiling_result as save_preprocessing_tiling_result,
)
from hs2p.tiling_artifacts import (
    ArtifactCompatibilitySpec,
    SlideSpec,
    TilingArtifacts,
    load_tiling_result,
    load_whole_slides_from_rows,
    maybe_load_existing_artifacts,
    validate_required_columns,
    validate_result_consistency,
    validate_tiling_artifacts,
    write_process_list,
)


def _write_mask_preview(
    *,
    mask_preview_path: Path | None,
    tissue_mask: np.ndarray | None,
) -> None:
    if mask_preview_path is None or tissue_mask is None:
        return
    mask_preview_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(tissue_mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8, copy=False)
    if mask.max(initial=0) <= 1:
        mask = mask * 255
    from PIL import Image

    Image.fromarray(mask).save(mask_preview_path)


def _compute_tiling_result(
    whole_slide: SlideSpec,
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
    mask_preview_path: Path | None,
    num_workers: int,
) -> TilingResult:
    sampling_spec = None
    if whole_slide.mask_path is not None:
        sampling_spec = build_default_sampling_spec(tiling)
    preprocessing_result = preprocess_slide(
        image_path=whole_slide.image_path,
        sample_id=whole_slide.sample_id,
        tissue_mask_path=whole_slide.mask_path,
        backend=tiling.backend,
        spacing_override=whole_slide.spacing_at_level_0,
        requested_tile_size_px=tiling.target_tile_size_px,
        requested_spacing_um=tiling.target_spacing_um,
        use_hsv=segmentation.use_hsv,
        use_otsu=segmentation.use_otsu,
        sthresh=segmentation.sthresh,
        sthresh_up=segmentation.sthresh_up,
        mthresh=segmentation.mthresh,
        close=segmentation.close,
        min_tissue_fraction=tiling.tissue_threshold,
        overlap=tiling.overlap,
        use_padding=tiling.use_padding,
        seg_downsample=segmentation.downsample,
        tolerance=tiling.tolerance,
        ref_tile_size_px=filtering.ref_tile_size,
        a_t=filtering.a_t,
        a_h=filtering.a_h,
filter_white=filtering.filter_white,
        filter_black=filtering.filter_black,
        white_threshold=filtering.white_threshold,
        black_threshold=filtering.black_threshold,
        fraction_threshold=filtering.fraction_threshold,
        num_workers=num_workers,
        selection_strategy=(
            CoordinateSelectionStrategy.MERGED_DEFAULT_TILING
            if sampling_spec is not None
            else None
        ),
        output_mode=(
            CoordinateOutputMode.SINGLE_OUTPUT if sampling_spec is not None else None
        ),
    )
    _write_mask_preview(
        mask_preview_path=mask_preview_path,
        tissue_mask=preprocessing_result.tissue_mask,
    )
    return preprocessing_result


def tile_slide(
    whole_slide: SlideSpec,
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
    preview: PreviewConfig | None = None,
    num_workers: int = 1,
) -> TilingResult:
    if preview is not None and (
        preview.save_mask_preview or preview.save_tiling_preview
    ):
        warnings.warn(
            "tile_slide() is compute-only and does not write preview artifacts; "
            "use write_tiling_preview() for tiling overlays and "
            "overlay_mask_on_slide() for mask overlays.",
            stacklevel=2,
        )
    sampling_spec = (
        build_default_sampling_spec(tiling)
        if whole_slide.mask_path is not None
        else None
    )
    backend_selection = resolve_backend(
        tiling.requested_backend,
        wsi_path=whole_slide.image_path,
        mask_path=whole_slide.mask_path,
    )
    if backend_selection.reason is not None:
        emit_progress_log(
            f"[backend] {whole_slide.sample_id}: {backend_selection.reason}"
        )
    effective_tiling = (
        tiling
        if backend_selection.backend == tiling.requested_backend
        else replace(tiling, backend=backend_selection.backend)
    )
    return _compute_tiling_result(
        whole_slide,
        tiling=effective_tiling,
        segmentation=segmentation,
        filtering=filtering,
        mask_preview_path=None,
        num_workers=num_workers,
    )


def save_tiling_result(
    result: TilingResult,
    output_dir: Path,
    *,
    tiles_dir: Path | None = None,
) -> TilingArtifacts:
    validate_result_consistency(result)
    tiles_dir = (
        Path(tiles_dir)
        if tiles_dir is not None
        else Path(output_dir) / "tiles"
    )
    tiles_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = save_preprocessing_tiling_result(
        result,
        output_dir=tiles_dir,
        sample_id=result.sample_id,
    )
    return TilingArtifacts(
        sample_id=result.sample_id,
        coordinates_npz_path=artifact_paths["npz"],
        coordinates_meta_path=artifact_paths["meta"],
        num_tiles=len(result.coordinates),
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

    When *filter_params* requests white/black filtering the tiles are checked
    during extraction so that pixel data is read only once.  The returned
    ``TilingResult`` has its coordinate arrays trimmed to the surviving tiles.
    """
    from PIL import Image

    jpeg_backend = str(jpeg_backend)
    _jpeg_encoder = None
    if jpeg_backend == "turbojpeg":
        import turbojpeg

        _jpeg_encoder = turbojpeg.TurboJPEG()

    tiles_dir = Path(tiles_dir) if tiles_dir is not None else Path(output_dir) / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tar_path = tiles_dir / f"{result.sample_id}.tiles.tar"
    manifest_path = tiles_dir / f"{result.sample_id}.tiles.manifest.csv"

    do_filter_white = filter_params is not None and filter_params.filter_white
    do_filter_black = filter_params is not None and filter_params.filter_black
    white_thresh = getattr(filter_params, "white_threshold", 220) if filter_params else 220
    black_thresh = getattr(filter_params, "black_threshold", 25) if filter_params else 25
    frac_thresh = getattr(filter_params, "fraction_threshold", 0.9) if filter_params else 0.9

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
            tile_records = _iter_tile_records_for_tar_extraction(
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

                if do_filter_white or do_filter_black:
                    total_pixels = tile_arr.shape[0] * tile_arr.shape[1]
                    if do_filter_white:
                        white_frac = np.all(tile_arr > white_thresh, axis=-1).sum() / total_pixels
                        if white_frac > frac_thresh:
                            continue
                    if do_filter_black:
                        black_frac = np.all(tile_arr < black_thresh, axis=-1).sum() / total_pixels
                        if black_frac > frac_thresh:
                            continue

                encode_start = time.perf_counter()
                if result.effective_tile_size_px != result.requested_tile_size_px:
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

    if len(kept_indices) == len(result.coordinates):
        return tar_path, result

    kept = np.asarray(sorted(kept_indices), dtype=np.int64)
    filtered_result = replace(
        result,
        tiles=replace(
            result.tiles,
            coordinates=result.coordinates[kept],
            tissue_fractions=result.tissue_fractions[kept],
            tile_index=np.arange(len(kept), dtype=np.int32),
        ),
    )
    return tar_path, filtered_result


def _iter_tile_arrays_for_tar_extraction(
    *,
    result: TilingResult,
    num_workers: int,
    gpu_decode: bool = False,
    supertile_sizes: Sequence[int] | None = None,
):
    tile_records = _iter_tile_records_for_tar_extraction(
        result=result,
        num_workers=num_workers,
        gpu_decode=gpu_decode,
        supertile_sizes=supertile_sizes,
    )
    for record in tile_records:
        yield record.tile_arr


def _iter_tile_records_for_tar_extraction(
    *,
    result: TilingResult,
    num_workers: int,
    gpu_decode: bool = False,
    supertile_sizes: Sequence[int] | None = None,
):
    tile_records = _iter_cucim_tile_records_for_tar_extraction(
        result=result,
        num_workers=num_workers,
        gpu_decode=gpu_decode,
        supertile_sizes=supertile_sizes,
    )
    if tile_records is not None:
        yield from tile_records
        return
    yield from _iter_wsd_tile_records_for_tar_extraction(
        result=result,
        supertile_sizes=supertile_sizes,
    )


def _iter_cucim_tile_records_for_tar_extraction(
    *,
    result: TilingResult,
    num_workers: int,
    gpu_decode: bool = False,
    supertile_sizes: Sequence[int] | None = None,
):
    if result.backend != "cucim":
        return None
    from hs2p.wsi.batched_reads import (
        BatchedReadRequest,
        iter_cucim_batched_read_regions,
    )
    try:
        requests = []
        read_step_px = resolve_read_step_px(result)
        step_px_lv0 = resolve_step_px_lv0(result)
        read_plans = list(
            iter_grouped_read_plans(
                result=result,
                read_step_px=read_step_px,
                step_px_lv0=step_px_lv0,
                supertile_sizes=supertile_sizes,
            )
        )
        for read_plan in read_plans:
            requests.append(
                BatchedReadRequest(
                    location=(int(read_plan.x), int(read_plan.y)),
                    size=(int(read_plan.read_size_px), int(read_plan.read_size_px)),
                    payload=read_plan,
                )
            )
        batched_regions = iter_cucim_batched_read_regions(
            image_path=result.image_path,
            requests=requests,
            level=int(result.read_level),
            num_workers=int(num_workers),
            gpu_decode=gpu_decode,
        )
    except ModuleNotFoundError:
        warnings.warn(
            "CuCIM is unavailable for backend='cucim'; falling back to sequential wholeslidedata tile extraction.",
            UserWarning,
            stacklevel=2,
        )
        return None

    def _iter_tiles():
        for request, region in batched_regions:
            read_plan = request.payload
            yield from _iter_planned_tile_views_from_read_plan_region(
                np.asarray(region),
                read_plan=read_plan,
                tile_size_px=int(result.effective_tile_size_px),
                read_step_px=read_step_px,
            )

    return _iter_tiles()
def _iter_cucim_tile_arrays_for_tar_extraction(
    *,
    result: TilingResult,
    num_workers: int,
    gpu_decode: bool = False,
):
    tile_records = _iter_cucim_tile_records_for_tar_extraction(
        result=result,
        num_workers=num_workers,
        gpu_decode=gpu_decode,
    )
    if tile_records is None:
        return None

    def _iter_tiles():
        for record in tile_records:
            yield record.tile_arr

    return _iter_tiles()


def _iter_wsd_tile_records_for_tar_extraction(
    *,
    result: TilingResult,
    supertile_sizes: Sequence[int] | None = None,
):
    import wholeslidedata as wsd

    wsi = wsd.WholeSlideImage(
        coerce_wsd_path(result.image_path, backend=result.backend),
        backend=result.backend,
    )
    read_step_px = resolve_read_step_px(result)
    step_px_lv0 = resolve_step_px_lv0(result)
    for read_plan in iter_grouped_read_plans(
        result=result,
        read_step_px=read_step_px,
        step_px_lv0=step_px_lv0,
        supertile_sizes=supertile_sizes,
    ):
        region = wsi.get_patch(
            int(read_plan.x),
            int(read_plan.y),
            int(read_plan.read_size_px),
            int(read_plan.read_size_px),
            spacing=float(result.effective_spacing_um),
            center=False,
        )
        region = np.asarray(region)
        yield from _iter_planned_tile_views_from_read_plan_region(
            region,
            read_plan=read_plan,
            tile_size_px=int(result.effective_tile_size_px),
            read_step_px=read_step_px,
        )


def _iter_wsd_tile_arrays_for_tar_extraction(
    *,
    result: TilingResult,
):
    tile_records = _iter_wsd_tile_records_for_tar_extraction(result=result)

    def _iter_tiles():
        for record in tile_records:
            yield record.tile_arr

    return _iter_tiles()

def _iter_planned_tile_views_from_read_plan_region(
    region: np.ndarray,
    *,
    read_plan: GroupedReadPlan,
    tile_size_px: int,
    read_step_px: int,
):
    for tile_view in iter_plan_region_tile_views(
        region,
        read_plan=read_plan,
        tile_size_px=int(tile_size_px),
        read_step_px=int(read_step_px),
    ):
        yield tile_view


def _format_tar_member_name(tile_index: int) -> str:
    return f"{int(tile_index):06d}.jpg"


def _needs_pixel_filtering(filtering: FilterConfig) -> bool:
    return bool(filtering.filter_white or filtering.filter_black)


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
    if len(result.coordinates) == 0:
        return None
    save_dir = output_dir / "preview" / "tiling"
    save_dir.mkdir(parents=True, exist_ok=True)
    coordinates = [tuple(coord) for coord in result.coordinates.tolist()]
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
class _PendingTilingPreview:
    whole_slide: SlideSpec
    base_artifact: TilingArtifacts
    mask_preview_path: Path | None
    future: Any


@dataclass(frozen=True)
class _PlannedSlideWork:
    whole_slide: SlideSpec
    artifact: TilingArtifacts | None = None
    compute_request: Any | None = None
    error: str | None = None
    traceback_text: str | None = None


@dataclass(frozen=True)
class _SlideComputeRequest:
    input_index: int
    whole_slide: SlideSpec
    tiling: TilingConfig
    segmentation: SegmentationConfig
    filtering: FilterConfig
    mask_preview_path: Path | None
    output_dir: Path
    num_workers: int
    jpeg_backend: str = "turbojpeg"
    gpu_decode: bool = False
    include_result: bool = False
    save_tiles: bool = False


@dataclass(frozen=True)
class _SlideComputeResponse:
    input_index: int
    whole_slide: SlideSpec
    ok: bool
    artifact: TilingArtifacts | None = None
    result: TilingResult | None = None
    mask_preview_path: Path | None = None
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
    )


def _build_success_process_row(
    *,
    whole_slide: SlideSpec,
    artifact: TilingArtifacts,
) -> dict[str, Any]:
    return {
        "sample_id": whole_slide.sample_id,
        "image_path": str(whole_slide.image_path),
        "tissue_mask_path": (
            str(whole_slide.mask_path) if whole_slide.mask_path is not None else None
        ),
        "tiling_status": "success",
        "num_tiles": artifact.num_tiles,
        "coordinates_npz_path": str(artifact.coordinates_npz_path),
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
) -> dict[str, Any]:
    return {
        "sample_id": whole_slide.sample_id,
        "image_path": str(whole_slide.image_path),
        "tissue_mask_path": (
            str(whole_slide.mask_path) if whole_slide.mask_path is not None else None
        ),
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
    pending: _PendingTilingPreview,
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
    request: _SlideComputeRequest,
) -> _SlideComputeResponse:
    try:
        defer_pixel_filtering = request.save_tiles and _needs_pixel_filtering(request.filtering)
        effective_filtering = (
            replace(request.filtering, filter_white=False, filter_black=False)
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
            num_workers=request.num_workers,
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
        )
        mask_preview_path = (
            request.mask_preview_path
            if request.mask_preview_path is not None
            and request.mask_preview_path.is_file()
            else None
        )
        return _SlideComputeResponse(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=True,
            artifact=artifact,
            result=result if request.include_result else None,
            mask_preview_path=mask_preview_path,
        )
    except Exception as exc:
        return _SlideComputeResponse(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=False,
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
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
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
    artifacts: list[TilingArtifacts] = []
    process_rows: list[dict[str, Any]] = []
    process_list_path = output_dir / "process_list.csv"
    existing_successes: dict[str, dict[str, Any]] = {}
    if resume and process_list_path.is_file():
        existing_df = pd.read_csv(process_list_path)
        validate_required_columns(
            existing_df,
            required_columns={
                "sample_id",
                "image_path",
                "tissue_mask_path",
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
    compatibility_specs: dict[tuple[bool, str], ArtifactCompatibilitySpec] = {}

    def _resolve_effective_tiling(whole_slide: SlideSpec) -> TilingConfig:
        backend_selection = resolve_backend(
            tiling.backend,
            wsi_path=whole_slide.image_path,
            mask_path=whole_slide.mask_path,
        )
        if backend_selection.reason is not None:
            emit_progress_log(
                f"[backend] {whole_slide.sample_id}: {backend_selection.reason}"
            )
        return (
            tiling
            if backend_selection.backend == tiling.requested_backend
            else replace(tiling, backend=backend_selection.backend)
        )

    planned_work: list[_PlannedSlideWork] = []
    compute_requests: list[_SlideComputeRequest] = []
    for whole_slide in whole_slides:
        try:
            effective_tiling = _resolve_effective_tiling(whole_slide)
            key = (whole_slide.mask_path is not None, effective_tiling.backend)
            if key not in compatibility_specs:
                sampling_spec = (
                    build_default_sampling_spec(effective_tiling)
                    if whole_slide.mask_path is not None
                    else None
                )
                compatibility_specs[key] = ArtifactCompatibilitySpec(
                    tiling=effective_tiling,
                    segmentation=segmentation,
                    filtering=filtering,
                    selection_strategy=(
                        CoordinateSelectionStrategy.MERGED_DEFAULT_TILING
                        if sampling_spec is not None
                        else None
                    ),
                    output_mode=(
                        CoordinateOutputMode.SINGLE_OUTPUT
                        if sampling_spec is not None
                        else None
                    ),
                )
            compatibility = compatibility_specs[key]
            artifact: TilingArtifacts | None = None
            if whole_slide.sample_id in existing_successes:
                row = existing_successes[whole_slide.sample_id]
                npz_path = Path(str(row["coordinates_npz_path"]))
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
                    _PlannedSlideWork(
                        whole_slide=whole_slide,
                        artifact=artifact,
                    )
                )
                continue
            mask_preview_path = None
            if preview is not None and preview.save_mask_preview:
                mask_dir = output_dir / "preview" / "mask"
                mask_preview_path = mask_dir / f"{whole_slide.sample_id}.jpg"
            compute_request = _SlideComputeRequest(
                input_index=len(planned_work),
                whole_slide=whole_slide,
                tiling=effective_tiling,
                segmentation=segmentation,
                filtering=filtering,
                mask_preview_path=mask_preview_path,
                output_dir=output_dir,
                num_workers=1,
                jpeg_backend=str(jpeg_backend),
                gpu_decode=gpu_decode,
                save_tiles=save_tiles,
            )
            planned_work.append(
                _PlannedSlideWork(
                    whole_slide=whole_slide,
                    compute_request=compute_request,
                )
            )
            compute_requests.append(compute_request)
        except Exception as exc:
            planned_work.append(
                _PlannedSlideWork(
                    whole_slide=whole_slide,
                    error=str(exc),
                    traceback_text=traceback.format_exc(),
                )
            )
    use_slide_pool, pool_processes, worker_inner_workers = _resolve_tiling_worker_allocation(
        num_workers=num_workers,
        compute_count=len(compute_requests),
    )
    preview_executor = (
        ThreadPoolExecutor(max_workers=1)
        if preview is not None and preview.save_tiling_preview
        else None
    )
    pending_preview: _PendingTilingPreview | None = None
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

    def _finalize_pending_preview_if_any() -> None:
        nonlocal pending_preview
        if pending_preview is None:
            return
        previous_pending = pending_preview
        pending_preview = None
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
                )
            )

    def _process_compute_response(response: _SlideComputeResponse) -> None:
        nonlocal pending_preview
        if not response.ok:
            _finalize_pending_preview_if_any()
            emit_progress_log(
                f"[tile_slides] FAILED {response.whole_slide.sample_id}: {response.error}",
            )
            _record_process_row(
                _build_failure_process_row(
                    whole_slide=response.whole_slide,
                    error=response.error or "unknown error",
                    traceback_text=response.traceback_text or "",
                )
            )
            return

        assert response.artifact is not None
        base_artifact = response.artifact
        _finalize_pending_preview_if_any()
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
            pending_preview = _PendingTilingPreview(
                whole_slide=response.whole_slide,
                base_artifact=base_artifact,
                mask_preview_path=response.mask_preview_path,
                future=future,
            )
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
        buffered_responses: dict[int, _SlideComputeResponse] = {}

        def _await_response(input_index: int) -> _SlideComputeResponse:
            while input_index not in buffered_responses:
                response = next(compute_response_iter)
                buffered_responses[response.input_index] = response
            return buffered_responses.pop(input_index)

        for planned in planned_work:
            if planned.artifact is not None:
                _finalize_pending_preview_if_any()
                artifacts.append(planned.artifact)
                _record_process_row(
                    _build_success_process_row(
                        whole_slide=planned.whole_slide,
                        artifact=planned.artifact,
                    )
                )
                continue
            if planned.error is not None:
                _finalize_pending_preview_if_any()
                emit_progress_log(
                    f"[tile_slides] FAILED {planned.whole_slide.sample_id}: {planned.error}",
                )
                _record_process_row(
                    _build_failure_process_row(
                        whole_slide=planned.whole_slide,
                        error=planned.error,
                        traceback_text=planned.traceback_text or "",
                    )
                )
                continue
            assert planned.compute_request is not None
            response = _await_response(planned.compute_request.input_index)
            _process_compute_response(response)

    try:
        if use_slide_pool:
            pool_requests = [
                _SlideComputeRequest(
                    input_index=request.input_index,
                    whole_slide=request.whole_slide,
                    tiling=request.tiling,
                    segmentation=request.segmentation,
                    filtering=request.filtering,
                    mask_preview_path=request.mask_preview_path,
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
                _SlideComputeRequest(
                    input_index=request.input_index,
                    whole_slide=request.whole_slide,
                    tiling=request.tiling,
                    segmentation=request.segmentation,
                    filtering=request.filtering,
                    mask_preview_path=request.mask_preview_path,
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
        _finalize_pending_preview_if_any()
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
