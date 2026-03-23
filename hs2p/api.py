import csv
import hashlib
import importlib
import io
import itertools
import json
import multiprocessing as mp
import tarfile
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass, replace
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
    ResolvedSamplingSpec,
    extract_coordinates,
    overlay_mask_on_slide as _overlay_mask_on_slide,
    write_coordinate_preview,
)
from hs2p.wsi.backend import coerce_wsd_path, resolve_backend


@dataclass(frozen=True)
class SlideSpec:
    """Identify one slide and its optional mask.

    Attributes:
        sample_id: Stable sample identifier used to name outputs.
        image_path: Path to the whole-slide image.
        mask_path: Optional path to a tissue or annotation mask.
        spacing_at_level_0: Optional override for the slide's native spacing at
            pyramid level 0 (µm/px).  Use this when the embedded metadata is
            missing or incorrect.  All other pyramid-level spacings are rescaled
            proportionally from this value.
    """

    sample_id: str
    image_path: Path
    mask_path: Path | None = None
    spacing_at_level_0: float | None = None


@dataclass
class TilingResult:
    """In-memory tiling output for one slide.

    Attributes:
        sample_id: Sample identifier associated with the tiling run.
        image_path: Slide path used to generate the coordinates.
        mask_path: Mask path used during generation, if any.
        backend: Slide-reading backend used during extraction.
        x: Tile origin x-coordinates in level-0 pixels.
        y: Tile origin y-coordinates in level-0 pixels.
        tile_index: Stable per-tile ids aligned with the coordinate arrays.
        target_spacing_um: Requested output spacing in microns per pixel.
        target_tile_size_px: Requested tile width and height at the target spacing.
        read_level: Pyramid level actually read from the slide.
        read_spacing_um: Native spacing of the pyramid level that was read.
        read_tile_size_px: Tile width and height at the read level.
        tile_size_lv0: Tile width and height expressed in level-0 pixels.
        overlap: Requested overlap fraction between neighboring tiles.
        tissue_threshold: Minimum tissue fraction used to keep tiles.
        num_tiles: Number of retained tiles.
        config_hash: Hash of the effective tiling, segmentation, and filtering config.
        tissue_fraction: Optional per-tile tissue coverage values.
        annotation: Optional annotation label for per-annotation sampling artifacts.
        selection_strategy: Optional internal coordinate-selection strategy marker.
        output_mode: Optional internal output-mode marker.
    """

    sample_id: str
    image_path: Path
    mask_path: Path | None
    backend: str
    x: np.ndarray
    y: np.ndarray
    tile_index: np.ndarray
    target_spacing_um: float
    target_tile_size_px: int
    read_level: int
    read_spacing_um: float
    read_tile_size_px: int
    tile_size_lv0: int
    overlap: float
    tissue_threshold: float
    num_tiles: int
    config_hash: str
    read_step_px: int | None = None
    step_px_lv0: int | None = None
    tissue_fraction: np.ndarray | None = None
    annotation: str | None = None
    selection_strategy: str | None = None
    output_mode: str | None = None


@dataclass(frozen=True)
class TilingArtifacts:
    """Named on-disk artifacts produced by a tiling run.

    Attributes:
        sample_id: Sample identifier that names the artifact files.
        coordinates_npz_path: Path to the saved ``.coordinates.npz`` coordinate artifact.
        coordinates_meta_path: Path to the saved ``.coordinates.meta.json`` metadata artifact.
        num_tiles: Number of tiles stored in the artifact pair.
        mask_preview_path: Optional path to the saved mask preview image.
        tiling_preview_path: Optional path to the saved tiling preview image.
    """

    sample_id: str
    coordinates_npz_path: Path
    coordinates_meta_path: Path
    num_tiles: int
    tiles_tar_path: Path | None = None
    mask_preview_path: Path | None = None
    tiling_preview_path: Path | None = None


def _validate_vector(name: str, value: np.ndarray | None) -> int | None:
    if value is None:
        return None
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {value.shape}")
    return int(value.shape[0])


def _validate_result_consistency(result: TilingResult) -> None:
    lengths = {
        "x": _validate_vector("x", result.x),
        "y": _validate_vector("y", result.y),
        "tile_index": _validate_vector("tile_index", result.tile_index),
    }
    if result.tissue_fraction is not None:
        lengths["tissue_fraction"] = _validate_vector(
            "tissue_fraction", result.tissue_fraction
        )
    expected = int(result.num_tiles)
    mismatched = [name for name, length in lengths.items() if length != expected]
    if mismatched:
        raise ValueError(
            "TilingResult arrays do not match num_tiles for fields: "
            + ", ".join(mismatched)
        )
    expected_index = np.arange(expected, dtype=np.int32)
    actual_index = result.tile_index.astype(np.int32, copy=False)
    if actual_index.shape != expected_index.shape or not np.array_equal(
        actual_index, expected_index
    ):
        raise ValueError("tile_index must be a contiguous range from 0 to num_tiles-1")


def compute_effective_config_hash(
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
    sampling_spec: ResolvedSamplingSpec | None = None,
    selection_strategy: str | None = None,
    output_mode: str | None = None,
    annotation: str | None = None,
) -> str:
    if (
        sampling_spec is None
        or selection_strategy is None
        or output_mode is None
    ):
        return compute_config_hash(
            tiling=tiling,
            segmentation=segmentation,
            filtering=filtering,
        )
    payload: dict[str, Any] = {
        "sampling": {
            "selection_strategy": selection_strategy,
            "output_mode": output_mode,
            "resolved_sampling_spec": sampling_spec,
        }
    }
    if annotation is not None:
        payload["sampling"]["annotation"] = annotation
    return compute_config_hash(
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        extra=payload,
    )


def _optional_path(value: Any) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan"}:
        return None
    return Path(text)


def _compute_tiling_result(
    whole_slide: SlideSpec,
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
    mask_preview_path: Path | None,
    num_workers: int,
    config_hash: str | None = None,
) -> TilingResult:
    sampling_spec = None
    if whole_slide.mask_path is not None:
        sampling_spec = build_default_sampling_spec(tiling)
    extraction = extract_coordinates(
        wsi_path=whole_slide.image_path,
        mask_path=whole_slide.mask_path,
        backend=tiling.backend,
        segment_params=segmentation,
        tiling_params=tiling,
        filter_params=filtering,
        sampling_spec=sampling_spec,
        mask_preview_path=mask_preview_path,
        spacing_at_level_0=whole_slide.spacing_at_level_0,
        disable_tqdm=True,
        num_workers=num_workers,
    )
    x = extraction.x.astype(np.int64, copy=False)
    y = extraction.y.astype(np.int64, copy=False)
    num_tiles = int(x.shape[0])
    tissue_fraction = None
    if extraction.tissue_percentages is not None:
        tissue_fraction = np.asarray(extraction.tissue_percentages, dtype=np.float32)
        if tissue_fraction.ndim != 1 or tissue_fraction.shape[0] != num_tiles:
            raise ValueError(
                "tissue_percentages length mismatch for "
                f"{whole_slide.sample_id}: expected {num_tiles}, "
                f"got shape {tissue_fraction.shape}"
            )
    return TilingResult(
        sample_id=whole_slide.sample_id,
        image_path=whole_slide.image_path,
        mask_path=whole_slide.mask_path,
        backend=tiling.backend,
        x=x,
        y=y,
        tile_index=np.arange(num_tiles, dtype=np.int32),
        tissue_fraction=tissue_fraction,
        target_spacing_um=tiling.target_spacing_um,
        target_tile_size_px=tiling.target_tile_size_px,
        read_level=extraction.read_level,
        read_spacing_um=extraction.read_spacing_um,
        read_tile_size_px=extraction.read_tile_size_px,
        tile_size_lv0=extraction.tile_size_lv0,
        overlap=tiling.overlap,
        tissue_threshold=tiling.tissue_threshold,
        num_tiles=num_tiles,
        config_hash=(
            config_hash
            if config_hash is not None
            else compute_effective_config_hash(
                tiling=tiling,
                segmentation=segmentation,
                filtering=filtering,
                sampling_spec=sampling_spec,
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
        ),
        read_step_px=extraction.read_step_px,
        step_px_lv0=extraction.step_px_lv0,
        selection_strategy=(
            CoordinateSelectionStrategy.MERGED_DEFAULT_TILING
            if sampling_spec is not None
            else None
        ),
        output_mode=(
            CoordinateOutputMode.SINGLE_OUTPUT if sampling_spec is not None else None
        ),
    )


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            key: _normalize_for_hash(val)
            for key, val in sorted(asdict(value).items())
        }
    if isinstance(value, dict):
        return {k: _normalize_for_hash(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    return value


def compute_config_hash(
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
    extra: dict[str, Any] | None = None,
) -> str:
    payload = {
        "tiling": _normalize_for_hash(asdict(tiling)),
        "segmentation": _normalize_for_hash(asdict(segmentation)),
        "filtering": _normalize_for_hash(asdict(filtering)),
    }
    if extra:
        payload["extra"] = _normalize_for_hash(extra)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_required_columns(
    df: pd.DataFrame,
    *,
    required_columns: set[str],
    file_path: Path,
    file_label: str,
) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            f"Unsupported {file_label} schema in {file_path}; missing required columns: "
            + ", ".join(missing)
        )


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
        config_hash=compute_effective_config_hash(
            tiling=effective_tiling,
            segmentation=segmentation,
            filtering=filtering,
            sampling_spec=sampling_spec,
            selection_strategy=(
                CoordinateSelectionStrategy.MERGED_DEFAULT_TILING
                if sampling_spec is not None
                else None
            ),
            output_mode=(
                CoordinateOutputMode.SINGLE_OUTPUT if sampling_spec is not None else None
            ),
        ),
    )


def save_tiling_result(
    result: TilingResult,
    output_dir: Path,
    *,
    tiles_dir: Path | None = None,
) -> TilingArtifacts:
    _validate_result_consistency(result)
    tiles_dir = (
        Path(tiles_dir)
        if tiles_dir is not None
        else Path(output_dir) / "tiles"
    )
    tiles_dir.mkdir(parents=True, exist_ok=True)
    npz_path = tiles_dir / f"{result.sample_id}.coordinates.npz"
    meta_path = tiles_dir / f"{result.sample_id}.coordinates.meta.json"

    payload = {
        "tile_index": result.tile_index.astype(np.int32, copy=False),
        "x": result.x.astype(np.int64, copy=False),
        "y": result.y.astype(np.int64, copy=False),
    }
    if result.tissue_fraction is not None:
        payload["tissue_fraction"] = result.tissue_fraction.astype(
            np.float32, copy=False
        )
    meta = {
        "sample_id": result.sample_id,
        "image_path": str(result.image_path),
        "mask_path": str(result.mask_path) if result.mask_path is not None else None,
        "backend": result.backend,
        "target_spacing_um": result.target_spacing_um,
        "target_tile_size_px": result.target_tile_size_px,
        "read_level": result.read_level,
        "read_spacing_um": result.read_spacing_um,
        "read_tile_size_px": result.read_tile_size_px,
        "read_step_px": result.read_step_px,
        "tile_size_lv0": result.tile_size_lv0,
        "step_px_lv0": result.step_px_lv0,
        "overlap": result.overlap,
        "tissue_threshold": result.tissue_threshold,
        "num_tiles": result.num_tiles,
        "config_hash": result.config_hash,
    }
    if result.annotation is not None:
        meta["annotation"] = result.annotation
    if result.selection_strategy is not None:
        meta["selection_strategy"] = result.selection_strategy
    if result.output_mode is not None:
        meta["output_mode"] = result.output_mode
    temp_npz_path: Path | None = None
    temp_meta_path: Path | None = None
    write_complete = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            dir=tiles_dir,
            delete=False,
        ) as handle:
            temp_npz_path = Path(handle.name)
            np.savez(handle, **payload)
            handle.flush()

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            dir=tiles_dir,
            delete=False,
        ) as handle:
            temp_meta_path = Path(handle.name)
            handle.write(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            handle.flush()

        temp_npz_path.replace(npz_path)
        temp_npz_path = None
        temp_meta_path.replace(meta_path)
        temp_meta_path = None
        write_complete = True
    finally:
        if not write_complete:
            npz_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        if temp_npz_path is not None:
            temp_npz_path.unlink(missing_ok=True)
        if temp_meta_path is not None:
            temp_meta_path.unlink(missing_ok=True)
    return TilingArtifacts(
        sample_id=result.sample_id,
        coordinates_npz_path=npz_path,
        coordinates_meta_path=meta_path,
        num_tiles=result.num_tiles,
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
    gpu_decode: bool = True,
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
                if result.read_tile_size_px != result.target_tile_size_px:
                    img = Image.fromarray(tile_arr).convert("RGB")
                    img = img.resize(
                        (result.target_tile_size_px, result.target_tile_size_px),
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

    if len(kept_indices) == result.num_tiles:
        return tar_path, result

    kept = np.asarray(sorted(kept_indices), dtype=np.int64)
    filtered_result = TilingResult(
        sample_id=result.sample_id,
        image_path=result.image_path,
        mask_path=result.mask_path,
        backend=result.backend,
        x=result.x[kept],
        y=result.y[kept],
        tile_index=np.arange(len(kept), dtype=np.int32),
        target_spacing_um=result.target_spacing_um,
        target_tile_size_px=result.target_tile_size_px,
        read_level=result.read_level,
        read_spacing_um=result.read_spacing_um,
        read_tile_size_px=result.read_tile_size_px,
        tile_size_lv0=result.tile_size_lv0,
        overlap=result.overlap,
        tissue_threshold=result.tissue_threshold,
        num_tiles=len(kept),
        config_hash=result.config_hash,
        read_step_px=result.read_step_px,
        step_px_lv0=result.step_px_lv0,
        tissue_fraction=(
            result.tissue_fraction[kept]
            if result.tissue_fraction is not None
            else None
        ),
        annotation=result.annotation,
        selection_strategy=result.selection_strategy,
        output_mode=result.output_mode,
    )
    return tar_path, filtered_result


def _iter_tile_arrays_for_tar_extraction(
    *,
    result: TilingResult,
    num_workers: int,
    gpu_decode: bool = True,
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
    gpu_decode: bool = True,
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
    gpu_decode: bool = True,
    supertile_sizes: Sequence[int] | None = None,
):
    if result.backend != "cucim":
        return None
    from hs2p.wsi.cucim_reader import CuImageReader
    try:
        reader = CuImageReader(result.image_path, gpu_decode=gpu_decode)
        reader._ensure_open()
    except ModuleNotFoundError:
        warnings.warn(
            "CuCIM is unavailable for backend='cucim'; falling back to sequential wholeslidedata tile extraction.",
            UserWarning,
            stacklevel=2,
        )
        return None

    read_step_px = _resolve_read_step_px(result)
    step_px_lv0 = _resolve_step_px_lv0(result)
    read_plans = list(
        _iter_grouped_read_plans_for_tar_extraction(
            result=result,
            read_step_px=read_step_px,
            step_px_lv0=step_px_lv0,
            supertile_sizes=supertile_sizes,
        )
    )

    def _iter_tiles():
        for read_size_px, size_plans in _group_read_plans_by_read_size(read_plans).items():
            locations = [(int(plan.x), int(plan.y)) for plan in size_plans]
            regions = reader.read_region(
                locations,
                (int(read_size_px), int(read_size_px)),
                level=int(result.read_level),
                num_workers=int(num_workers),
            )
            for read_plan, region in zip(size_plans, regions):
                yield from _iter_tile_records_from_read_plan_region(
                    np.asarray(region),
                    read_plan=read_plan,
                    tile_size_px=int(result.read_tile_size_px),
                    read_step_px=read_step_px,
                )

    return _iter_tiles()


def _group_read_plans_by_read_size(
    read_plans: Sequence["_WSDTarReadPlan"],
) -> dict[int, list["_WSDTarReadPlan"]]:
    grouped: dict[int, list["_WSDTarReadPlan"]] = {}
    for plan in read_plans:
        grouped.setdefault(int(plan.read_size_px), []).append(plan)
    return grouped


def _iter_cucim_tile_arrays_for_tar_extraction(
    *,
    result: TilingResult,
    num_workers: int,
    gpu_decode: bool = True,
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
    read_step_px = _resolve_read_step_px(result)
    step_px_lv0 = _resolve_step_px_lv0(result)
    for read_plan in _iter_grouped_read_plans_for_tar_extraction(
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
            spacing=float(result.read_spacing_um),
            center=False,
        )
        region = np.asarray(region)
        yield from _iter_tile_records_from_read_plan_region(
            region,
            read_plan=read_plan,
            tile_size_px=int(result.read_tile_size_px),
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


@dataclass(frozen=True)
class _WSDTarReadPlan:
    x: int
    y: int
    read_size_px: int
    block_size: int
    tile_indices: tuple[int, ...]


@dataclass(frozen=True)
class _TarTileRecord:
    tile_index: int
    x: int
    y: int
    tile_arr: np.ndarray


def _iter_tile_records_from_read_plan_region(
    region: np.ndarray,
    *,
    read_plan: _WSDTarReadPlan,
    tile_size_px: int,
    read_step_px: int,
):
    region = np.asarray(region)
    if read_plan.block_size == 1:
        yield _TarTileRecord(
            tile_index=int(read_plan.tile_indices[0]),
            x=int(read_plan.x),
            y=int(read_plan.y),
            tile_arr=region,
        )
        return
    tile_index_iter = iter(int(idx) for idx in read_plan.tile_indices)
    for x_idx in range(read_plan.block_size):
        x0 = x_idx * read_step_px
        for y_idx in range(read_plan.block_size):
            y0 = y_idx * read_step_px
            yield _TarTileRecord(
                tile_index=next(tile_index_iter),
                x=int(read_plan.x + x0),
                y=int(read_plan.y + y0),
                tile_arr=region[
                    y0 : y0 + tile_size_px,
                    x0 : x0 + tile_size_px,
                ],
            )


def _format_tar_member_name(tile_index: int) -> str:
    return f"{int(tile_index):06d}.jpg"


def _resolve_read_step_px(result: TilingResult) -> int:
    if result.read_step_px is not None:
        return int(result.read_step_px)
    return max(
        1,
        int(round(int(result.read_tile_size_px) * (1.0 - float(result.overlap)), 0)),
    )


def _resolve_step_px_lv0(result: TilingResult) -> int:
    if result.step_px_lv0 is not None:
        return int(result.step_px_lv0)
    if result.x.size > 1:
        unique_x = np.unique(np.sort(result.x.astype(np.int64, copy=False)))
        diffs = np.diff(unique_x)
        diffs = diffs[diffs > 0]
        if diffs.size > 0:
            return int(diffs.min())
    if result.y.size > 1:
        unique_y = np.unique(np.sort(result.y.astype(np.int64, copy=False)))
        diffs = np.diff(unique_y)
        diffs = diffs[diffs > 0]
        if diffs.size > 0:
            return int(diffs.min())
    return max(
        1,
        int(round(int(result.tile_size_lv0) * (1.0 - float(result.overlap)), 0)),
    )


def _iter_grouped_read_plans_for_tar_extraction(
    *,
    result: TilingResult,
    read_step_px: int,
    step_px_lv0: int,
    supertile_sizes: Sequence[int] | None = None,
):
    if supertile_sizes is None:
        supertile_sizes = (8, 4, 2)
    grouped_sizes = tuple(
        sorted(
            {
                int(size)
                for size in supertile_sizes
                if int(size) > 1
            },
            reverse=True,
        )
    )
    if step_px_lv0 <= 0:
        step_px_lv0 = int(result.tile_size_lv0)
    coord_to_index = {
        (int(x), int(y)): idx
        for idx, (x, y) in enumerate(
            zip(
                result.x.astype(np.int64, copy=False).tolist(),
                result.y.astype(np.int64, copy=False).tolist(),
            )
        )
    }
    consumed = np.zeros(result.num_tiles, dtype=bool)
    tile_size_px = int(result.read_tile_size_px)
    grouped_plans: dict[int, list[_WSDTarReadPlan]] = {
        size: [] for size in grouped_sizes
    }
    grouped_plans[1] = []

    def _build_grouped_plan(idx: int, block_size: int) -> _WSDTarReadPlan | None:
        if consumed[idx]:
            return None
        x0 = int(result.x[idx])
        y0 = int(result.y[idx])
        indices: list[int] = []
        for x_idx in range(block_size):
            for y_idx in range(block_size):
                coord = (
                    x0 + x_idx * step_px_lv0,
                    y0 + y_idx * step_px_lv0,
                )
                match_idx = coord_to_index.get(coord)
                if match_idx is None or consumed[match_idx]:
                    return None
                indices.append(match_idx)
            if len(indices) < (x_idx + 1) * block_size:
                return None
        return _WSDTarReadPlan(
            x=x0,
            y=y0,
            read_size_px=tile_size_px + (block_size - 1) * read_step_px,
            block_size=block_size,
            tile_indices=tuple(indices),
        )

    for block_size in grouped_sizes:
        if result.num_tiles < block_size * block_size:
            continue
        for idx in range(result.num_tiles):
            plan = _build_grouped_plan(idx, block_size)
            if plan is None:
                continue
            for match_idx in plan.tile_indices:
                consumed[match_idx] = True
            grouped_plans[block_size].append(plan)

    for idx in range(result.num_tiles):
        if consumed[idx]:
            continue
        consumed[idx] = True
        grouped_plans[1].append(
            _WSDTarReadPlan(
                x=int(result.x[idx]),
                y=int(result.y[idx]),
                read_size_px=tile_size_px,
                block_size=1,
                tile_indices=(idx,),
            )
        )

    for block_size in (*grouped_sizes, 1):
        yield from grouped_plans[block_size]


def _needs_pixel_filtering(filtering: FilterConfig) -> bool:
    return bool(filtering.filter_white or filtering.filter_black)


def load_tiling_result(
    coordinates_npz_path: Path,
    coordinates_meta_path: Path,
) -> TilingResult:
    try:
        tiles = np.load(coordinates_npz_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"Unable to load tiling npz artifact {coordinates_npz_path}: {exc}"
        ) from exc
    try:
        meta = json.loads(Path(coordinates_meta_path).read_text())
    except Exception as exc:
        raise ValueError(
            f"Unable to load tiling metadata artifact {coordinates_meta_path}: {exc}"
        ) from exc
    required_npz_keys = {"tile_index", "x", "y"}
    missing_npz_keys = sorted(required_npz_keys - set(tiles.files))
    if missing_npz_keys:
        raise ValueError(
            f"Invalid tiling npz artifact {coordinates_npz_path}; missing keys: "
            + ", ".join(missing_npz_keys)
        )
    required_meta_keys = {
        "sample_id",
        "image_path",
        "mask_path",
        "backend",
        "target_spacing_um",
        "target_tile_size_px",
        "read_level",
        "read_spacing_um",
        "read_tile_size_px",
        "tile_size_lv0",
        "overlap",
        "tissue_threshold",
        "num_tiles",
        "config_hash",
    }
    missing_meta_keys = sorted(required_meta_keys - set(meta))
    if missing_meta_keys:
        raise ValueError(
            f"Invalid tiling metadata artifact {coordinates_meta_path}; missing keys: "
            + ", ".join(missing_meta_keys)
        )
    x = tiles["x"].astype(np.int64, copy=False)
    y = tiles["y"].astype(np.int64, copy=False)
    tile_index = tiles["tile_index"].astype(np.int32, copy=False)
    tissue_fraction = None
    if "tissue_fraction" in tiles:
        tissue_fraction = tiles["tissue_fraction"].astype(np.float32, copy=False)
    result = TilingResult(
        sample_id=meta["sample_id"],
        image_path=Path(meta["image_path"]),
        mask_path=Path(meta["mask_path"]) if meta.get("mask_path") else None,
        backend=meta["backend"],
        x=x,
        y=y,
        tile_index=tile_index,
        tissue_fraction=tissue_fraction,
        target_spacing_um=float(meta["target_spacing_um"]),
        target_tile_size_px=int(meta["target_tile_size_px"]),
        read_level=int(meta["read_level"]),
        read_spacing_um=float(meta["read_spacing_um"]),
        read_tile_size_px=int(meta["read_tile_size_px"]),
        tile_size_lv0=int(meta["tile_size_lv0"]),
        overlap=float(meta["overlap"]),
        tissue_threshold=float(meta["tissue_threshold"]),
        num_tiles=int(meta["num_tiles"]),
        config_hash=str(meta["config_hash"]),
        read_step_px=(
            int(meta["read_step_px"])
            if meta.get("read_step_px") is not None
            else None
        ),
        step_px_lv0=(
            int(meta["step_px_lv0"])
            if meta.get("step_px_lv0") is not None
            else None
        ),
        annotation=(
            str(meta["annotation"]) if meta.get("annotation") is not None else None
        ),
        selection_strategy=(
            str(meta["selection_strategy"])
            if meta.get("selection_strategy") is not None
            else None
        ),
        output_mode=(
            str(meta["output_mode"]) if meta.get("output_mode") is not None else None
        ),
    )
    _validate_result_consistency(result)
    return result


def validate_tiling_artifacts(
    *,
    whole_slide: SlideSpec,
    coordinates_npz_path: Path,
    coordinates_meta_path: Path,
    expected_config_hash: str,
) -> TilingArtifacts:
    result = load_tiling_result(
        coordinates_npz_path=coordinates_npz_path, coordinates_meta_path=coordinates_meta_path
    )
    if result.sample_id != whole_slide.sample_id:
        raise ValueError(
            f"Precomputed tiles sample_id mismatch for {whole_slide.sample_id}: "
            f"found {result.sample_id}"
        )
    if result.config_hash != expected_config_hash:
        raise ValueError(
            f"Precomputed tiles config_hash mismatch for {whole_slide.sample_id}: "
            f"stored={result.config_hash!r}, expected={expected_config_hash!r}"
        )
    if result.image_path != whole_slide.image_path:
        raise ValueError(
            f"Precomputed tiles image_path mismatch for {whole_slide.sample_id}: "
            f"expected {whole_slide.image_path}, found {result.image_path}"
        )
    if result.mask_path != whole_slide.mask_path:
        raise ValueError(
            f"Precomputed tiles mask_path mismatch for {whole_slide.sample_id}: "
            f"expected {whole_slide.mask_path}, found {result.mask_path}"
        )
    return TilingArtifacts(
        sample_id=result.sample_id,
        coordinates_npz_path=coordinates_npz_path,
        coordinates_meta_path=coordinates_meta_path,
        num_tiles=result.num_tiles,
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


def _maybe_load_existing_artifacts(
    *,
    whole_slide: SlideSpec,
    read_coordinates_from: Path,
    expected_config_hash: str,
) -> TilingArtifacts | None:
    npz_path = read_coordinates_from / f"{whole_slide.sample_id}.coordinates.npz"
    meta_path = read_coordinates_from / f"{whole_slide.sample_id}.coordinates.meta.json"
    if not npz_path.is_file() and not meta_path.is_file():
        return None
    if not npz_path.is_file() or not meta_path.is_file():
        raise ValueError(
            f"Missing tiling sidecar for sample_id={whole_slide.sample_id} in {read_coordinates_from}"
        )
    return validate_tiling_artifacts(
        whole_slide=whole_slide,
        coordinates_npz_path=npz_path,
        coordinates_meta_path=meta_path,
        expected_config_hash=expected_config_hash,
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
    if result.num_tiles == 0:
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


def _write_process_list(
    process_rows: list[dict[str, Any]], process_list_path: Path
) -> None:
    process_list_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            dir=process_list_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            pd.DataFrame(process_rows).to_csv(handle, index=False)
        temp_path.replace(process_list_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


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
    config_hash: str
    mask_preview_path: Path | None
    output_dir: Path
    num_workers: int
    jpeg_backend: str = "turbojpeg"
    gpu_decode: bool = True
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
        "mask_path": (
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
        "mask_path": (
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
            config_hash=request.config_hash,
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
    gpu_decode: bool = True,
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
        _validate_required_columns(
            existing_df,
            required_columns={
                "sample_id",
                "image_path",
                "mask_path",
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
    expected_hashes: dict[tuple[bool, str], str] = {}

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
            if key not in expected_hashes:
                sampling_spec = (
                    build_default_sampling_spec(effective_tiling)
                    if whole_slide.mask_path is not None
                    else None
                )
                expected_hashes[key] = compute_effective_config_hash(
                    tiling=effective_tiling,
                    segmentation=segmentation,
                    filtering=filtering,
                    sampling_spec=sampling_spec,
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
            expected_hash = expected_hashes[key]
            artifact: TilingArtifacts | None = None
            if whole_slide.sample_id in existing_successes:
                row = existing_successes[whole_slide.sample_id]
                npz_path = Path(str(row["coordinates_npz_path"]))
                meta_path = Path(str(row["coordinates_meta_path"]))
                artifact = validate_tiling_artifacts(
                    whole_slide=whole_slide,
                    coordinates_npz_path=npz_path,
                    coordinates_meta_path=meta_path,
                    expected_config_hash=expected_hash,
                )
            if read_coordinates_from is not None and artifact is None:
                artifact = _maybe_load_existing_artifacts(
                    whole_slide=whole_slide,
                    read_coordinates_from=Path(read_coordinates_from),
                    expected_config_hash=expected_hash,
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
                config_hash=expected_hash,
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
                    config_hash=request.config_hash,
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
                    config_hash=request.config_hash,
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
    _write_process_list(process_rows, process_list_path)
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


def load_whole_slides_from_rows(rows: Sequence[dict[str, Any]]) -> list[SlideSpec]:
    whole_slides: list[SlideSpec] = []
    for row in rows:
        whole_slides.append(
            SlideSpec(
                sample_id=str(row["sample_id"]),
                image_path=Path(row["image_path"]),
                mask_path=_optional_path(row.get("mask_path")),
            )
        )
    return whole_slides
