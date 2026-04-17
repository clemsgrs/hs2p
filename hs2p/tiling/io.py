from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from hs2p.tiling.generate import canonicalize_tiling_result
from hs2p.tiling.result import TileGeometry, TilingResult

COORDINATE_SPACE = "level0_px"
TILE_ORDER = "x_then_y"

_TOP_LEVEL_META_KEYS = {
    "provenance",
    "slide",
    "tiling",
    "segmentation",
    "filtering",
    "artifact",
}
_PROVENANCE_KEYS = {
    "sample_id",
    "image_path",
    "mask_path",
    "backend",
    "requested_backend",
}
_SLIDE_KEYS = {
    "dimensions",
    "base_spacing_um",
    "level_downsamples",
}
_TILING_KEYS = {
    "requested_tile_size_px",
    "requested_spacing_um",
    "read_level",
    "read_tile_size_px",
    "read_spacing_um",
    "tile_size_lv0",
    "tolerance",
    "step_px_lv0",
    "overlap",
    "min_tissue_fraction",
    "is_within_tolerance",
    "n_tiles",
}
_SEGMENTATION_KEYS = {
    "tissue_method",
    "seg_downsample",
    "seg_level",
    "seg_spacing_um",
    "sthresh",
    "sthresh_up",
    "mthresh",
    "close",
    "mask_path",
    "ref_tile_size_px",
    "tissue_mask_tissue_value",
    "mask_level",
    "mask_spacing_um",
}
_FILTERING_KEYS = {
    "a_t",
    "a_h",
    "filter_white",
    "filter_black",
    "white_threshold",
    "black_threshold",
    "fraction_threshold",
    "filter_grayspace",
    "grayspace_saturation_threshold",
    "grayspace_fraction_threshold",
    "filter_blur",
    "blur_threshold",
    "qc_spacing_um",
}
_ARTIFACT_KEYS = {
    "coordinate_space",
    "tile_order",
    "annotation",
    "selection_strategy",
    "output_mode",
}


def _build_tiling_metadata(result: TilingResult) -> dict[str, Any]:
    n_tiles = len(result.x)
    provenance = {
        "sample_id": result.sample_id,
        "image_path": str(result.image_path),
        "mask_path": str(result.mask_path) if result.mask_path is not None else None,
        "backend": result.backend,
        "requested_backend": result.requested_backend,
    }
    slide = {
        "dimensions": result.slide_dimensions,
        "base_spacing_um": result.base_spacing_um,
        "level_downsamples": result.level_downsamples,
    }
    tiling = {
        "requested_tile_size_px": result.requested_tile_size_px,
        "requested_spacing_um": result.requested_spacing_um,
        "read_level": result.read_level,
        "read_tile_size_px": result.read_tile_size_px,
        "read_spacing_um": result.read_spacing_um,
        "tile_size_lv0": result.tile_size_lv0,
        "tolerance": result.tolerance,
        "step_px_lv0": result.step_px_lv0,
        "overlap": result.overlap,
        "min_tissue_fraction": result.min_tissue_fraction,
        "is_within_tolerance": result.is_within_tolerance,
        "n_tiles": n_tiles,
    }
    segmentation = {
        "tissue_method": result.tissue_method,
        "seg_downsample": result.seg_downsample,
        "seg_level": result.seg_level,
        "seg_spacing_um": result.seg_spacing_um,
        "sthresh": result.seg_sthresh,
        "sthresh_up": result.seg_sthresh_up,
        "mthresh": result.seg_mthresh,
        "close": result.seg_close,
        "mask_path": str(result.mask_path) if result.mask_path is not None else None,
        "ref_tile_size_px": result.ref_tile_size_px,
        "tissue_mask_tissue_value": result.tissue_mask_tissue_value,
        "mask_level": result.mask_level,
        "mask_spacing_um": result.mask_spacing_um,
    }
    filtering = {
        "a_t": result.a_t,
        "a_h": result.a_h,
        "filter_white": result.filter_white,
        "filter_black": result.filter_black,
        "white_threshold": result.white_threshold,
        "black_threshold": result.black_threshold,
        "fraction_threshold": result.fraction_threshold,
        "filter_grayspace": result.filter_grayspace,
        "grayspace_saturation_threshold": result.grayspace_saturation_threshold,
        "grayspace_fraction_threshold": result.grayspace_fraction_threshold,
        "filter_blur": result.filter_blur,
        "blur_threshold": result.blur_threshold,
        "qc_spacing_um": result.qc_spacing_um,
    }
    artifact = {
        "coordinate_space": COORDINATE_SPACE,
        "tile_order": TILE_ORDER,
        "annotation": result.annotation,
        "selection_strategy": result.selection_strategy,
        "output_mode": result.output_mode,
    }
    return {
        "provenance": provenance,
        "slide": slide,
        "tiling": tiling,
        "segmentation": segmentation,
        "filtering": filtering,
        "artifact": artifact,
    }


def _validate_tile_index(tile_index: np.ndarray, n_tiles: int) -> np.ndarray:
    tile_index = np.asarray(tile_index, dtype=np.int32)
    if tile_index.ndim != 1 or tile_index.shape[0] != n_tiles:
        raise ValueError("tile_index must be a 1D array aligned with x/y")
    expected = np.arange(n_tiles, dtype=np.int32)
    if not np.array_equal(tile_index, expected):
        raise ValueError("tile_index must be a contiguous range from 0 to n_tiles-1")
    return tile_index


def _validate_metadata_schema(meta: dict[str, Any]) -> None:
    def _raise_key_error(section: str, missing: set[str], extra: set[str]) -> None:
        parts: list[str] = []
        if missing:
            parts.append(f"missing keys {sorted(missing)}")
        if extra:
            parts.append(f"unexpected keys {sorted(extra)}")
        raise ValueError(f"Invalid tiling metadata in {section}: " + "; ".join(parts))

    top_keys = set(meta)
    missing_top = _TOP_LEVEL_META_KEYS - top_keys
    extra_top = top_keys - _TOP_LEVEL_META_KEYS
    if missing_top or extra_top:
        _raise_key_error("top-level", missing_top, extra_top)

    sections = {
        "provenance": _PROVENANCE_KEYS,
        "slide": _SLIDE_KEYS,
        "tiling": _TILING_KEYS,
        "segmentation": _SEGMENTATION_KEYS,
        "filtering": _FILTERING_KEYS,
        "artifact": _ARTIFACT_KEYS,
    }
    for section_name, expected_keys in sections.items():
        section = meta[section_name]
        if not isinstance(section, dict):
            raise ValueError(f"Invalid tiling metadata in {section_name}: expected object")
        section_keys = set(section)
        missing = expected_keys - section_keys
        extra = section_keys - expected_keys
        if missing or extra:
            _raise_key_error(section_name, missing, extra)


def normalize_artifact_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve(strict=False))


def validate_tiling_result_provenance(
    result: TilingResult,
    *,
    sample_id: str,
    image_path: str | Path,
    mask_path: str | Path | None,
    tissue_mask_tissue_value: int | None,
) -> None:
    if result.sample_id != sample_id:
        raise ValueError(
            f"Precomputed tiles sample_id mismatch: expected {sample_id!r}, found {result.sample_id!r}"
        )
    expected_image = normalize_artifact_path(image_path)
    actual_image = normalize_artifact_path(result.image_path)
    if actual_image != expected_image:
        raise ValueError(
            "Precomputed tiles image_path mismatch: "
            f"expected {expected_image!r}, found {actual_image!r}"
        )
    expected_mask = normalize_artifact_path(mask_path)
    actual_mask = normalize_artifact_path(result.mask_path)
    if actual_mask != expected_mask:
        raise ValueError(
            "Precomputed tiles mask_path mismatch: "
            f"expected {expected_mask!r}, found {actual_mask!r}"
        )
    if result.tissue_mask_tissue_value != tissue_mask_tissue_value:
        raise ValueError(
            "Precomputed tiles tissue_mask_tissue_value mismatch: "
            f"expected {tissue_mask_tissue_value!r}, found {result.tissue_mask_tissue_value!r}"
        )


def _save_tiling_result(
    result: TilingResult,
    output_dir: Path,
    sample_id: str | None = None,
) -> dict[str, "Path | None"]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = replace(result, tiles=canonicalize_tiling_result(result.tiles))
    artifact_name = sample_id or canonical.sample_id
    if artifact_name is None:
        raise ValueError("sample_id is required when saving a TilingResult")

    meta_path = output_dir / f"{artifact_name}.coordinates.meta.json"

    committed_npz_path: Path | None = None
    temp_npz_path: Path | None = None
    temp_meta_path: Path | None = None
    try:
        if len(canonical.x) > 0:
            npz_path: Path | None = output_dir / f"{artifact_name}.coordinates.npz"
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npz",
                dir=output_dir,
                delete=False,
            ) as handle:
                temp_npz_path = Path(handle.name)
                np.savez_compressed(
                    handle,
                    tile_index=canonical.tile_index.astype(np.int32, copy=False),
                    x=canonical.x.astype(np.int64, copy=False),
                    y=canonical.y.astype(np.int64, copy=False),
                    tissue_fractions=canonical.tissue_fractions.astype(np.float32, copy=False),
                )
                handle.flush()
            temp_npz_path.replace(npz_path)
            temp_npz_path = None
            committed_npz_path = npz_path
        else:
            npz_path = None

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            dir=output_dir,
            delete=False,
        ) as handle:
            temp_meta_path = Path(handle.name)
            handle.write(json.dumps(_build_tiling_metadata(canonical), indent=2, sort_keys=True) + "\n")
            handle.flush()
        temp_meta_path.replace(meta_path)
        temp_meta_path = None
        committed_npz_path = None
    finally:
        if temp_npz_path is not None:
            temp_npz_path.unlink(missing_ok=True)
        if temp_meta_path is not None:
            temp_meta_path.unlink(missing_ok=True)
        if committed_npz_path is not None:
            committed_npz_path.unlink(missing_ok=True)

    return {"npz": npz_path, "meta": meta_path}


def _load_tiling_result_from_paths(npz_path: "Path | None", meta_path: Path) -> TilingResult:
    meta = json.loads(Path(meta_path).read_text())
    return _load_tiling_result(npz_path=npz_path, meta=meta)


def _load_tiling_result(*, npz_path: "Path | None", meta: dict[str, Any]) -> TilingResult:
    _validate_metadata_schema(meta)

    if meta["tiling"]["n_tiles"] == 0:
        x = np.empty(0, dtype=np.int64)
        y = np.empty(0, dtype=np.int64)
        tissue_fractions = np.empty(0, dtype=np.float32)
        tile_index = np.empty(0, dtype=np.int32)
    else:
        if npz_path is None:
            raise ValueError("npz_path is required when n_tiles > 0")
        data = np.load(npz_path, allow_pickle=False)
        if "tile_index" not in data:
            raise ValueError("Invalid tiling artifact: missing tile_index")
        if "x" not in data:
            raise ValueError("Invalid tiling artifact: missing x")
        if "y" not in data:
            raise ValueError("Invalid tiling artifact: missing y")
        if "tissue_fractions" not in data:
            raise ValueError("Invalid tiling artifact: missing tissue_fractions")
        x = np.asarray(data["x"], dtype=np.int64)
        y = np.asarray(data["y"], dtype=np.int64)
        tissue_fractions = np.asarray(data["tissue_fractions"], dtype=np.float32)
        tile_index = _validate_tile_index(np.asarray(data["tile_index"]), len(x))
    provenance = meta["provenance"]
    slide = meta["slide"]
    tiling = meta["tiling"]
    segmentation = meta["segmentation"]
    filtering = meta["filtering"]
    artifact = meta["artifact"]

    tiles = TileGeometry(
        x=x,
        y=y,
        tissue_fractions=tissue_fractions,
        tile_index=tile_index,
        requested_tile_size_px=int(tiling["requested_tile_size_px"]),
        requested_spacing_um=float(tiling["requested_spacing_um"]),
        read_level=int(tiling["read_level"]),
        read_tile_size_px=int(tiling["read_tile_size_px"]),
        read_spacing_um=float(tiling["read_spacing_um"]),
        tile_size_lv0=int(tiling["tile_size_lv0"]),
        is_within_tolerance=bool(tiling["is_within_tolerance"]),
        base_spacing_um=(
            float(slide["base_spacing_um"])
            if slide["base_spacing_um"] is not None
            else 0.0
        ),
        slide_dimensions=slide["dimensions"],
        level_downsamples=slide["level_downsamples"],
        overlap=tiling["overlap"],
        min_tissue_fraction=tiling["min_tissue_fraction"],
    )
    return TilingResult(
        tiles=tiles,
        sample_id=provenance["sample_id"],
        image_path=provenance["image_path"],
        backend=provenance["backend"],
        requested_backend=provenance["requested_backend"],
        tolerance=tiling["tolerance"],
        step_px_lv0=tiling["step_px_lv0"],
        tissue_method=segmentation["tissue_method"],
        seg_downsample=segmentation["seg_downsample"],
        seg_level=segmentation["seg_level"],
        seg_spacing_um=segmentation["seg_spacing_um"],
        seg_sthresh=segmentation["sthresh"],
        seg_sthresh_up=segmentation["sthresh_up"],
        seg_mthresh=segmentation["mthresh"],
        seg_close=segmentation["close"],
        ref_tile_size_px=segmentation["ref_tile_size_px"],
        a_t=filtering["a_t"],
        a_h=filtering["a_h"],
        filter_white=filtering["filter_white"],
        filter_black=filtering["filter_black"],
        white_threshold=filtering["white_threshold"],
        black_threshold=filtering["black_threshold"],
        fraction_threshold=filtering["fraction_threshold"],
        filter_grayspace=filtering["filter_grayspace"],
        grayspace_saturation_threshold=filtering["grayspace_saturation_threshold"],
        grayspace_fraction_threshold=filtering["grayspace_fraction_threshold"],
        filter_blur=filtering["filter_blur"],
        blur_threshold=filtering["blur_threshold"],
        qc_spacing_um=filtering["qc_spacing_um"],
        mask_path=segmentation["mask_path"],
        tissue_mask_tissue_value=segmentation["tissue_mask_tissue_value"],
        mask_level=segmentation["mask_level"],
        mask_spacing_um=segmentation["mask_spacing_um"],
        annotation=artifact["annotation"],
        selection_strategy=artifact["selection_strategy"],
        output_mode=artifact["output_mode"],
    )


__all__ = [
    "COORDINATE_SPACE",
    "TILE_ORDER",
    "_build_tiling_metadata",
    "_load_tiling_result",
    "_load_tiling_result_from_paths",
    "_save_tiling_result",
    "_validate_metadata_schema",
    "_validate_tile_index",
    "normalize_artifact_path",
    "validate_tiling_result_provenance",
]
