from __future__ import annotations

import csv
import io
import tarfile
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from hs2p.tiling.result import TilingResult
from hs2p.tile_qc import filter_coordinate_tiles, needs_pixel_qc
from hs2p.wsi import iter_tile_records_from_result, open_reader_for_result
from hs2p.wsi.reader import BatchRegionReader


def _annotation_tar_stem(sample_id: str, annotation: str | None) -> str:
    if annotation is None or annotation == "tissue":
        return f"{sample_id}.tiles"
    return f"{sample_id}.{annotation}.tiles"


def _format_tar_member_name(tile_index: int) -> str:
    return f"{int(tile_index):06d}.jpg"


def _needs_pixel_filtering(filtering) -> bool:
    return needs_pixel_qc(filtering)


def _apply_qc_filtering_to_result(
    *,
    result: TilingResult,
    filter_params,
    num_workers: int,
    gpu_decode: bool = False,
) -> TilingResult:
    with open_reader_for_result(result, gpu_decode=gpu_decode) as slide:
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


def extract_tiles_to_tar(
    result: TilingResult,
    output_dir: Path,
    *,
    annotation: str | None = None,
    jpeg_quality: int = 90,
    jpeg_backend: str = "turbojpeg",
    supertile_sizes: Sequence[int] | None = None,
    tiles_dir: Path | None = None,
    filter_params: Any | None = None,
    num_workers: int = 4,
    gpu_decode: bool = False,
    phase_recorder: Any | None = None,
) -> tuple[Path, TilingResult]:
    jpeg_backend = str(jpeg_backend)
    _jpeg_encoder = None
    if jpeg_backend == "turbojpeg":
        import turbojpeg

        _jpeg_encoder = turbojpeg.TurboJPEG()

    from hs2p.artifacts import _annotation_tiles_dir
    tiles_dir = Path(tiles_dir) if tiles_dir is not None else _annotation_tiles_dir(output_dir, annotation)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    stem = _annotation_tar_stem(result.sample_id, annotation)
    tar_path = tiles_dir / f"{stem}.tar"
    manifest_path = tiles_dir / f"{stem}.manifest.csv"

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
                        (result.requested_tile_size_px, result.requested_tile_size_px),
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
                    {"tile_index": tile_index, "x": int(record.x), "y": int(record.y)}
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


__all__ = [
    "_annotation_tar_stem",
    "_apply_qc_filtering_to_result",
    "extract_tiles_to_tar",
]
