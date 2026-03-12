from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from hs2p.wsi import SamplingParameters, extract_coordinates, visualize_coordinates


@dataclass(frozen=True)
class WholeSlide:
    sample_id: str
    image_path: Path
    mask_path: Path | None = None


@dataclass(frozen=True)
class TilingConfig:
    target_spacing_um: float
    target_tile_size_px: int
    tolerance: float
    overlap: float
    tissue_threshold: float
    drop_holes: bool
    use_padding: bool
    backend: str = "asap"

    # Legacy compatibility for wrapped core logic.
    @property
    def spacing(self) -> float:
        return self.target_spacing_um

    @property
    def tile_size(self) -> int:
        return self.target_tile_size_px

    @property
    def min_tissue_percentage(self) -> float:
        return self.tissue_threshold


@dataclass(frozen=True)
class SegmentationConfig:
    downsample: int
    sthresh: int
    sthresh_up: int
    mthresh: int
    close: int
    use_otsu: bool
    use_hsv: bool


@dataclass(frozen=True)
class FilterConfig:
    ref_tile_size: int
    a_t: int
    a_h: int
    max_n_holes: int
    filter_white: bool
    filter_black: bool
    white_threshold: int
    black_threshold: int
    fraction_threshold: float


@dataclass(frozen=True)
class QCConfig:
    save_mask_preview: bool = False
    save_tiling_preview: bool = False
    downsample: int = 32


@dataclass
class TilingResult:
    sample_id: str
    image_path: Path
    mask_path: Path | None
    backend: str
    x_lv0: np.ndarray
    y_lv0: np.ndarray
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
    tissue_fraction: np.ndarray | None = None


@dataclass(frozen=True)
class TilingArtifacts:
    sample_id: str
    tiles_npz_path: Path
    tiles_meta_path: Path
    num_tiles: int
    mask_preview_path: Path | None = None
    tiling_preview_path: Path | None = None


def _validate_result_consistency(result: TilingResult) -> None:
    lengths = {
        "x_lv0": int(result.x_lv0.shape[0]),
        "y_lv0": int(result.y_lv0.shape[0]),
        "tile_index": int(result.tile_index.shape[0]),
    }
    if result.tissue_fraction is not None:
        lengths["tissue_fraction"] = int(result.tissue_fraction.shape[0])
    expected = int(result.num_tiles)
    mismatched = [name for name, length in lengths.items() if length != expected]
    if mismatched:
        raise ValueError(
            "TilingResult arrays do not match num_tiles for fields: " + ", ".join(mismatched)
        )
    expected_index = np.arange(expected, dtype=np.int32)
    actual_index = result.tile_index.astype(np.int32, copy=False)
    if actual_index.shape != expected_index.shape or not np.array_equal(actual_index, expected_index):
        raise ValueError("tile_index must be a contiguous range from 0 to num_tiles-1")


def _build_default_sampling_params(tiling: TilingConfig) -> SamplingParameters:
    return SamplingParameters(
        pixel_mapping={"background": 0, "tissue": 1},
        color_mapping={"background": None, "tissue": None},
        tissue_percentage={"background": None, "tissue": tiling.tissue_threshold},
    )


def _compute_tiling_result(
    whole_slide: WholeSlide,
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
    mask_visu_path: Path | None,
    num_workers: int,
) -> TilingResult:
    sampling_params = None
    if whole_slide.mask_path is not None:
        sampling_params = _build_default_sampling_params(tiling)
    extraction = extract_coordinates(
        wsi_path=whole_slide.image_path,
        mask_path=whole_slide.mask_path,
        backend=tiling.backend,
        segment_params=segmentation,
        tiling_params=tiling,
        filter_params=filtering,
        sampling_params=sampling_params,
        mask_visu_path=mask_visu_path,
        disable_tqdm=True,
        num_workers=num_workers,
    )
    x_lv0 = extraction.x_lv0.astype(np.int64, copy=False)
    y_lv0 = extraction.y_lv0.astype(np.int64, copy=False)
    num_tiles = int(x_lv0.shape[0])
    tissue_fraction = np.asarray(extraction.tissue_percentages, dtype=np.float32)
    if tissue_fraction.shape[0] != num_tiles:
        tissue_fraction = None
    return TilingResult(
        sample_id=whole_slide.sample_id,
        image_path=whole_slide.image_path,
        mask_path=whole_slide.mask_path,
        backend=tiling.backend,
        x_lv0=x_lv0,
        y_lv0=y_lv0,
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
        config_hash=compute_config_hash(
            tiling=tiling,
            segmentation=segmentation,
            filtering=filtering,
        ),
    )


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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


def _build_cli_configs(cfg: Any) -> tuple[TilingConfig, SegmentationConfig, FilterConfig]:
    return (
        TilingConfig(
            target_spacing_um=cfg.tiling.params.target_spacing_um,
            target_tile_size_px=cfg.tiling.params.target_tile_size_px,
            tolerance=cfg.tiling.params.tolerance,
            overlap=cfg.tiling.params.overlap,
            tissue_threshold=cfg.tiling.params.tissue_threshold,
            drop_holes=cfg.tiling.params.drop_holes,
            use_padding=cfg.tiling.params.use_padding,
            backend=cfg.tiling.backend,
        ),
        SegmentationConfig(**dict(cfg.tiling.seg_params)),
        FilterConfig(**dict(cfg.tiling.filter_params)),
    )


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
    whole_slide: WholeSlide,
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
    qc: QCConfig | None = None,
    num_workers: int = 1,
) -> TilingResult:
    del qc
    return _compute_tiling_result(
        whole_slide,
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        mask_visu_path=None,
        num_workers=num_workers,
    )


def save_tiling_result(
    result: TilingResult,
    output_dir: Path,
    *,
    coordinates_dir: Path | None = None,
) -> TilingArtifacts:
    _validate_result_consistency(result)
    coordinates_dir = Path(coordinates_dir) if coordinates_dir is not None else Path(output_dir) / "coordinates"
    coordinates_dir.mkdir(parents=True, exist_ok=True)
    npz_path = coordinates_dir / f"{result.sample_id}.tiles.npz"
    meta_path = coordinates_dir / f"{result.sample_id}.tiles.meta.json"

    payload = {
        "tile_index": result.tile_index.astype(np.int32, copy=False),
        "x_lv0": result.x_lv0.astype(np.int64, copy=False),
        "y_lv0": result.y_lv0.astype(np.int64, copy=False),
    }
    if result.tissue_fraction is not None:
        payload["tissue_fraction"] = result.tissue_fraction.astype(np.float32, copy=False)
    np.savez(npz_path, **payload)

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
        "tile_size_lv0": result.tile_size_lv0,
        "overlap": result.overlap,
        "tissue_threshold": result.tissue_threshold,
        "num_tiles": result.num_tiles,
        "config_hash": result.config_hash,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return TilingArtifacts(
        sample_id=result.sample_id,
        tiles_npz_path=npz_path,
        tiles_meta_path=meta_path,
        num_tiles=result.num_tiles,
    )


def load_tiling_result(
    tiles_npz_path: Path,
    tiles_meta_path: Path,
) -> TilingResult:
    tiles = np.load(tiles_npz_path, allow_pickle=False)
    meta = json.loads(Path(tiles_meta_path).read_text())
    x_lv0 = tiles["x_lv0"].astype(np.int64, copy=False)
    y_lv0 = tiles["y_lv0"].astype(np.int64, copy=False)
    tile_index = tiles["tile_index"].astype(np.int32, copy=False)
    tissue_fraction = None
    if "tissue_fraction" in tiles:
        tissue_fraction = tiles["tissue_fraction"].astype(np.float32, copy=False)
    result = TilingResult(
        sample_id=meta["sample_id"],
        image_path=Path(meta["image_path"]),
        mask_path=Path(meta["mask_path"]) if meta.get("mask_path") else None,
        backend=meta["backend"],
        x_lv0=x_lv0,
        y_lv0=y_lv0,
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
    )
    _validate_result_consistency(result)
    return result


def validate_tiling_artifacts(
    *,
    whole_slide: WholeSlide,
    tiles_npz_path: Path,
    tiles_meta_path: Path,
    expected_config_hash: str,
) -> TilingArtifacts:
    result = load_tiling_result(tiles_npz_path=tiles_npz_path, tiles_meta_path=tiles_meta_path)
    if result.sample_id != whole_slide.sample_id:
        raise ValueError(
            f"Precomputed tiles sample_id mismatch for {whole_slide.sample_id}: "
            f"found {result.sample_id}"
        )
    if result.config_hash != expected_config_hash:
        raise ValueError(
            f"Precomputed tiles config_hash mismatch for {whole_slide.sample_id}"
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
        tiles_npz_path=tiles_npz_path,
        tiles_meta_path=tiles_meta_path,
        num_tiles=result.num_tiles,
    )


def _validate_whole_slides(whole_slides: Sequence[WholeSlide]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for whole_slide in whole_slides:
        if whole_slide.sample_id in seen:
            duplicates.append(whole_slide.sample_id)
        seen.add(whole_slide.sample_id)
    if duplicates:
        duplicate_text = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate sample_id values are not allowed: {duplicate_text}")


def _maybe_load_existing_artifacts(
    *,
    whole_slide: WholeSlide,
    read_tiles_from: Path,
    expected_config_hash: str,
) -> TilingArtifacts | None:
    npz_path = read_tiles_from / f"{whole_slide.sample_id}.tiles.npz"
    meta_path = read_tiles_from / f"{whole_slide.sample_id}.tiles.meta.json"
    if not npz_path.is_file() and not meta_path.is_file():
        return None
    if not npz_path.is_file() or not meta_path.is_file():
        raise ValueError(
            f"Missing tiling sidecar for sample_id={whole_slide.sample_id} in {read_tiles_from}"
        )
    return validate_tiling_artifacts(
        whole_slide=whole_slide,
        tiles_npz_path=npz_path,
        tiles_meta_path=meta_path,
        expected_config_hash=expected_config_hash,
    )


def _write_tiling_preview(
    *,
    result: TilingResult,
    output_dir: Path,
    downsample: int,
) -> Path:
    save_dir = output_dir / "visualization" / "tiling"
    save_dir.mkdir(parents=True, exist_ok=True)
    coordinates = list(zip(result.x_lv0.tolist(), result.y_lv0.tolist()))
    visualize_coordinates(
        wsi_path=result.image_path,
        coordinates=coordinates,
        tile_size_lv0=result.tile_size_lv0,
        save_dir=save_dir,
        downsample=downsample,
        backend=result.backend,
        sample_id=result.sample_id,
    )
    return save_dir / f"{result.sample_id}.jpg"


def tile_slides(
    whole_slides: Sequence[WholeSlide],
    *,
    tiling: TilingConfig,
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
    qc: QCConfig | None = None,
    output_dir: Path,
    num_workers: int = 1,
    resume: bool = False,
    read_tiles_from: Path | None = None,
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
                "tiles_npz_path",
                "tiles_meta_path",
                "error",
                "traceback",
            },
            file_path=process_list_path,
            file_label="tiling process_list.csv",
        )
        for row in existing_df.to_dict(orient="records"):
            if row.get("tiling_status") == "success":
                existing_successes[str(row["sample_id"])] = row
    for whole_slide in whole_slides:
        expected_hash = compute_config_hash(
            tiling=tiling,
            segmentation=segmentation,
            filtering=filtering,
        )
        try:
            artifact: TilingArtifacts | None = None
            if whole_slide.sample_id in existing_successes:
                row = existing_successes[whole_slide.sample_id]
                npz_path = Path(str(row["tiles_npz_path"]))
                meta_path = Path(str(row["tiles_meta_path"]))
                artifact = validate_tiling_artifacts(
                    whole_slide=whole_slide,
                    tiles_npz_path=npz_path,
                    tiles_meta_path=meta_path,
                    expected_config_hash=expected_hash,
                )
            if read_tiles_from is not None:
                if artifact is None:
                    artifact = _maybe_load_existing_artifacts(
                        whole_slide=whole_slide,
                        read_tiles_from=Path(read_tiles_from),
                        expected_config_hash=expected_hash,
                    )
            if artifact is None:
                mask_visu_path = None
                if qc is not None and qc.save_mask_preview:
                    mask_dir = output_dir / "visualization" / "mask"
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    mask_visu_path = mask_dir / f"{whole_slide.sample_id}.jpg"
                result = _compute_tiling_result(
                    whole_slide,
                    tiling=tiling,
                    segmentation=segmentation,
                    filtering=filtering,
                    mask_visu_path=mask_visu_path,
                    num_workers=num_workers,
                )
                artifact = save_tiling_result(result, output_dir=output_dir)
                mask_preview_path = mask_visu_path if mask_visu_path is not None and mask_visu_path.is_file() else None
                tiling_preview_path = None
                if qc is not None and qc.save_tiling_preview:
                    tiling_preview_path = _write_tiling_preview(
                        result=result,
                        output_dir=output_dir,
                        downsample=qc.downsample,
                    )
                artifact = TilingArtifacts(
                    sample_id=artifact.sample_id,
                    tiles_npz_path=artifact.tiles_npz_path,
                    tiles_meta_path=artifact.tiles_meta_path,
                    num_tiles=artifact.num_tiles,
                    mask_preview_path=mask_preview_path,
                    tiling_preview_path=tiling_preview_path,
                )
            artifacts.append(artifact)
            process_rows.append(
                {
                    "sample_id": whole_slide.sample_id,
                    "image_path": str(whole_slide.image_path),
                    "mask_path": str(whole_slide.mask_path) if whole_slide.mask_path is not None else None,
                    "tiling_status": "success",
                    "num_tiles": artifact.num_tiles,
                    "tiles_npz_path": str(artifact.tiles_npz_path),
                    "tiles_meta_path": str(artifact.tiles_meta_path),
                    "error": np.nan,
                    "traceback": np.nan,
                }
            )
        except Exception as exc:
            process_rows.append(
                {
                    "sample_id": whole_slide.sample_id,
                    "image_path": str(whole_slide.image_path),
                    "mask_path": str(whole_slide.mask_path) if whole_slide.mask_path is not None else None,
                    "tiling_status": "failed",
                    "num_tiles": 0,
                    "tiles_npz_path": np.nan,
                    "tiles_meta_path": np.nan,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    pd.DataFrame(process_rows).to_csv(process_list_path, index=False)
    return artifacts


def load_whole_slides_from_rows(rows: Sequence[dict[str, Any]]) -> list[WholeSlide]:
    whole_slides: list[WholeSlide] = []
    for row in rows:
        whole_slides.append(
            WholeSlide(
                sample_id=str(row["sample_id"]),
                image_path=Path(row["image_path"]),
                mask_path=Path(row["mask_path"]) if row.get("mask_path") else None,
            )
        )
    return whole_slides
