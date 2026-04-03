
import pandas as pd
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hs2p.configs import FilterConfig, SegmentationConfig, TilingConfig
from hs2p.preprocessing import (
    TilingResult,
    _save_tiling_result,
    _load_tiling_result_from_paths,
)


@dataclass(frozen=True)
class SlideSpec:
    """Identify one slide and its optional mask."""

    sample_id: str
    image_path: Path
    mask_path: Path | None = None
    spacing_at_level_0: float | None = None


@dataclass(frozen=True)
class TilingArtifacts:
    """Named on-disk artifacts produced by a tiling run."""

    sample_id: str
    coordinates_npz_path: Path | None
    coordinates_meta_path: Path
    num_tiles: int
    tiles_tar_path: Path | None = None
    mask_preview_path: Path | None = None
    tiling_preview_path: Path | None = None


@dataclass(frozen=True)
class CompatibilitySpec:
    tiling: TilingConfig
    segmentation: SegmentationConfig
    filtering: FilterConfig
    selection_strategy: str | None = None
    output_mode: str | None = None
    annotation: str | None = None


def _validate_vector(name: str, value: np.ndarray | None) -> int | None:
    if value is None:
        return None
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {value.shape}")
    return int(value.shape[0])


def validate_result_consistency(result: TilingResult) -> None:
    x = np.asarray(result.x)
    y = np.asarray(result.y)
    lengths = {
        "x": int(x.shape[0]),
        "y": int(y.shape[0]),
        "tile_index": _validate_vector("tile_index", result.tile_index),
    }
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"x and y must have shape (N,), got {x.shape} and {y.shape}")
    lengths["tissue_fractions"] = _validate_vector(
        "tissue_fractions", result.tissue_fractions
    )
    expected = int(x.shape[0])
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


def save_tiling_result(
    result: TilingResult,
    output_dir: Path,
    *,
    tiles_dir: Path | None = None,
) -> "TilingArtifacts":
    validate_result_consistency(result)
    tiles_dir = (
        Path(tiles_dir)
        if tiles_dir is not None
        else Path(output_dir) / "tiles"
    )
    tiles_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _save_tiling_result(
        result,
        output_dir=tiles_dir,
        sample_id=result.sample_id,
    )
    return TilingArtifacts(
        sample_id=result.sample_id,
        coordinates_npz_path=artifact_paths["npz"],
        coordinates_meta_path=artifact_paths["meta"],
        num_tiles=len(result.x),
    )


def load_tiling_result(
    coordinates_npz_path: Path | None,
    coordinates_meta_path: Path,
) -> TilingResult:
    try:
        result = _load_tiling_result_from_paths(
            coordinates_npz_path,
            coordinates_meta_path,
        )
    except Exception as exc:
        raise ValueError(
            "Unable to load tiling artifacts "
            f"{coordinates_npz_path} and {coordinates_meta_path}: {exc}"
        ) from exc
    validate_result_consistency(result)
    return result


def optional_path(value: Any) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan"}:
        return None
    return Path(text)


def validate_required_columns(
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


def validate_tiling_artifacts(
    *,
    whole_slide: SlideSpec,
    coordinates_npz_path: Path | None,
    coordinates_meta_path: Path,
    compatibility: CompatibilitySpec,
) -> TilingArtifacts:
    result = load_tiling_result(
        coordinates_npz_path=coordinates_npz_path, coordinates_meta_path=coordinates_meta_path
    )
    if result.sample_id != whole_slide.sample_id:
        raise ValueError(
            f"Precomputed tiles sample_id mismatch for {whole_slide.sample_id}: "
            f"found {result.sample_id}"
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
    if result.backend != compatibility.tiling.backend:
        raise ValueError("precomputed tiles backend mismatch")
    if result.requested_spacing_um != compatibility.tiling.target_spacing_um:
        raise ValueError("precomputed tiles target_spacing_um mismatch")
    if result.requested_tile_size_px != compatibility.tiling.target_tile_size_px:
        raise ValueError("precomputed tiles target_tile_size_px mismatch")
    if result.overlap != compatibility.tiling.overlap:
        raise ValueError("precomputed tiles overlap mismatch")
    if result.min_tissue_fraction != compatibility.tiling.tissue_threshold:
        raise ValueError("precomputed tiles tissue_threshold mismatch")
    if result.tolerance != compatibility.tiling.tolerance:
        raise ValueError("precomputed tiles tolerance mismatch")
    if result.seg_downsample != compatibility.segmentation.downsample:
        raise ValueError("precomputed tiles seg_downsample mismatch")
    if result.seg_sthresh != compatibility.segmentation.sthresh:
        raise ValueError("precomputed tiles sthresh mismatch")
    if result.seg_sthresh_up != compatibility.segmentation.sthresh_up:
        raise ValueError("precomputed tiles sthresh_up mismatch")
    if result.seg_mthresh != compatibility.segmentation.mthresh:
        raise ValueError("precomputed tiles mthresh mismatch")
    if result.seg_close != compatibility.segmentation.close:
        raise ValueError("precomputed tiles close mismatch")
    if result.seg_use_otsu != compatibility.segmentation.use_otsu:
        raise ValueError("precomputed tiles use_otsu mismatch")
    if result.seg_use_hsv != compatibility.segmentation.use_hsv:
        raise ValueError("precomputed tiles use_hsv mismatch")
    if result.ref_tile_size_px != compatibility.filtering.ref_tile_size:
        raise ValueError("precomputed tiles ref_tile_size mismatch")
    if result.a_t != compatibility.filtering.a_t:
        raise ValueError("precomputed tiles a_t mismatch")
    if result.a_h != compatibility.filtering.a_h:
        raise ValueError("precomputed tiles a_h mismatch")
    if result.filter_white != compatibility.filtering.filter_white:
        raise ValueError("precomputed tiles filter_white mismatch")
    if result.filter_black != compatibility.filtering.filter_black:
        raise ValueError("precomputed tiles filter_black mismatch")
    if (
        compatibility.filtering.filter_white
        and result.white_threshold != compatibility.filtering.white_threshold
    ):
        raise ValueError("precomputed tiles white_threshold mismatch")
    if (
        compatibility.filtering.filter_black
        and result.black_threshold != compatibility.filtering.black_threshold
    ):
        raise ValueError("precomputed tiles black_threshold mismatch")
    if (
        (compatibility.filtering.filter_white or compatibility.filtering.filter_black)
        and result.fraction_threshold != compatibility.filtering.fraction_threshold
    ):
        raise ValueError("precomputed tiles fraction_threshold mismatch")
    if result.filter_grayspace != compatibility.filtering.filter_grayspace:
        raise ValueError("precomputed tiles filter_grayspace mismatch")
    if (
        compatibility.filtering.filter_grayspace
        and (
        result.grayspace_saturation_threshold
        != compatibility.filtering.grayspace_saturation_threshold
        )
    ):
        raise ValueError("precomputed tiles grayspace_saturation_threshold mismatch")
    if (
        compatibility.filtering.filter_grayspace
        and (
        result.grayspace_fraction_threshold
        != compatibility.filtering.grayspace_fraction_threshold
        )
    ):
        raise ValueError("precomputed tiles grayspace_fraction_threshold mismatch")
    if result.filter_blur != compatibility.filtering.filter_blur:
        raise ValueError("precomputed tiles filter_blur mismatch")
    if (
        compatibility.filtering.filter_blur
        and result.blur_threshold != compatibility.filtering.blur_threshold
    ):
        raise ValueError("precomputed tiles blur_threshold mismatch")
    if (
        (
            compatibility.filtering.filter_white
            or compatibility.filtering.filter_black
            or compatibility.filtering.filter_grayspace
            or compatibility.filtering.filter_blur
        )
        and result.qc_spacing_um != compatibility.filtering.qc_spacing_um
    ):
        raise ValueError("precomputed tiles qc_spacing_um mismatch")
    if result.selection_strategy != compatibility.selection_strategy:
        raise ValueError("precomputed tiles selection_strategy mismatch")
    if result.output_mode != compatibility.output_mode:
        raise ValueError("precomputed tiles output_mode mismatch")
    if result.annotation != compatibility.annotation:
        raise ValueError("precomputed tiles annotation mismatch")
    return TilingArtifacts(
        sample_id=result.sample_id,
        coordinates_npz_path=coordinates_npz_path,
        coordinates_meta_path=coordinates_meta_path,
        num_tiles=len(result.x),
    )


def maybe_load_existing_artifacts(
    *,
    whole_slide: SlideSpec,
    read_coordinates_from: Path,
    compatibility: CompatibilitySpec,
) -> TilingArtifacts | None:
    npz_path = read_coordinates_from / f"{whole_slide.sample_id}.coordinates.npz"
    meta_path = read_coordinates_from / f"{whole_slide.sample_id}.coordinates.meta.json"
    if not npz_path.is_file() and not meta_path.is_file():
        return None
    if not meta_path.is_file():
        raise ValueError(
            f"Missing tiling sidecar for sample_id={whole_slide.sample_id} in {read_coordinates_from}"
        )
    resolved_npz: Path | None = npz_path if npz_path.is_file() else None
    if resolved_npz is None:
        import json as _json
        n_tiles = _json.loads(meta_path.read_text()).get("tiling", {}).get("n_tiles", -1)
        if n_tiles != 0:
            raise ValueError(
                f"Missing tiling sidecar for sample_id={whole_slide.sample_id} in {read_coordinates_from}"
            )
    return validate_tiling_artifacts(
        whole_slide=whole_slide,
        coordinates_npz_path=resolved_npz,
        coordinates_meta_path=meta_path,
        compatibility=compatibility,
    )


def write_process_list(process_rows: list[dict[str, Any]], process_list_path: Path) -> None:
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


def load_whole_slides_from_rows(rows: Sequence[dict[str, Any]]) -> list[SlideSpec]:
    whole_slides: list[SlideSpec] = []
    for row in rows:
        legacy_mask_columns = [
            column
            for column in ("tissue_mask_path", "annotation_mask_path")
            if column in row
        ]
        if legacy_mask_columns:
            raise ValueError(
                "Process list rows use deprecated mask columns: "
                + ", ".join(sorted(legacy_mask_columns))
                + "; use 'mask_path' instead"
            )
        whole_slides.append(
            SlideSpec(
                sample_id=str(row["sample_id"]),
                image_path=Path(row["image_path"]),
                mask_path=optional_path(row.get("mask_path")),
            )
        )
    return whole_slides


__all__ = [
    "CompatibilitySpec",
    "SlideSpec",
    "TilingArtifacts",
    "load_tiling_result",
    "load_whole_slides_from_rows",
    "save_tiling_result",
    "maybe_load_existing_artifacts",
    "optional_path",
    "validate_required_columns",
    "validate_result_consistency",
    "validate_tiling_artifacts",
    "write_process_list",
]
