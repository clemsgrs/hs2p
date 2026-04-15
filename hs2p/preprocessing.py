"""Reusable low-level preprocessing primitives shared with downstream projects."""


import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image

from hs2p.configs import SegmentationConfig
from hs2p.segmentation import segment_tissue_image
from hs2p.wsi.geometry import project_discrete_grid_origins
from hs2p.wsi.reader import open_slide, select_level, select_level_for_downsample
from hs2p.tile_qc import filter_coordinate_tiles, needs_pixel_qc


def _normalize_level_downsamples(
    level_downsamples: list[float] | list[tuple[float, float]],
) -> list[float]:
    normalized: list[float] = []
    for downsample in level_downsamples:
        if isinstance(downsample, tuple):
            normalized.append(float(downsample[0]))
        else:
            normalized.append(float(downsample))
    return normalized


@dataclass(frozen=True)
class ContourResult:
    contours: list[np.ndarray]
    holes: list[list[np.ndarray]]
    mask: np.ndarray


@dataclass(frozen=True)
class ResolvedTissueMask:
    tissue_mask: np.ndarray
    tissue_method: str
    seg_downsample: int
    seg_level: int
    seg_spacing_um: float
    mask_path: str | Path | None = None
    tissue_mask_tissue_value: int | None = None
    mask_level: int | None = None
    mask_spacing_um: float | None = None

    def __post_init__(self) -> None:
        if self.mask_path is not None:
            object.__setattr__(self, "mask_path", Path(self.mask_path))


def detect_contours(
    tissue_mask: np.ndarray,
    *,
    slide_dimensions: tuple[int, int],
    ref_tile_size_px: int = 16,
    requested_spacing_um: float = 0.5,
    a_t: int = 4,
    base_spacing_um: float | None = None,
    level_downsamples: list[float] | list[tuple[float, float]] | None = None,
    tolerance: float = 0.05,
) -> ContourResult:
    """Detect tissue contours and holes, scaled into level-0 pixel space."""
    if tissue_mask.max() == 0:
        return ContourResult(contours=[], holes=[], mask=tissue_mask)

    mask_h, mask_w = tissue_mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = slide_w / mask_w
    scale_y = slide_h / mask_h

    min_fg_area = 0
    if a_t > 0:
        if base_spacing_um is None or level_downsamples is None:
            raise ValueError(
                "base_spacing_um and level_downsamples are required when a_t > 0 "
                "so contour filtering can use actual slide geometry."
            )
        level_sel = select_level(
            requested_spacing_um=requested_spacing_um,
            level0_spacing_um=base_spacing_um,
            level_downsamples=[
                (float(downsample), float(downsample))
                for downsample in _normalize_level_downsamples(level_downsamples)
            ],
            tolerance=tolerance,
        )
        current_scale = level_sel.read_spacing_um / base_spacing_um
        ref_tile_mask_w = ref_tile_size_px * current_scale / scale_x
        ref_tile_mask_h = ref_tile_size_px * current_scale / scale_y
        scaled_ref_tile_area = int(ref_tile_mask_w * ref_tile_mask_h)
        min_fg_area = a_t * scaled_ref_tile_area

    if tissue_mask.ndim == 3:
        tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2GRAY)
    raw_contours, hierarchy = cv2.findContours(
        tissue_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )

    if hierarchy is None or len(raw_contours) == 0:
        return ContourResult(contours=[], holes=[], mask=tissue_mask)

    hierarchy = hierarchy[0]
    filtered_contours = []
    filtered_holes: list[list[np.ndarray]] = []
    for fg_idx, h in enumerate(hierarchy):
        if h[3] != -1:
            continue
        child_hole_indices = np.flatnonzero(hierarchy[:, 3] == fg_idx)
        area = cv2.contourArea(raw_contours[fg_idx])
        if child_hole_indices.size > 0:
            hole_areas = [cv2.contourArea(raw_contours[idx]) for idx in child_hole_indices]
            area -= float(np.sum(hole_areas))
        if area == 0 or area <= min_fg_area:
            continue

        contour_lv0 = raw_contours[fg_idx].copy().astype(np.float64)
        contour_lv0[:, 0, 0] *= scale_x
        contour_lv0[:, 0, 1] *= scale_y
        filtered_contours.append(contour_lv0.astype(np.int32))

        hole_contours_lv0 = []
        for hole_idx in child_hole_indices.tolist():
            hole_lv0 = raw_contours[hole_idx].copy().astype(np.float64)
            hole_lv0[:, 0, 0] *= scale_x
            hole_lv0[:, 0, 1] *= scale_y
            hole_contours_lv0.append(hole_lv0.astype(np.int32))
        filtered_holes.append(hole_contours_lv0)

    return ContourResult(
        contours=filtered_contours,
        holes=filtered_holes,
        mask=tissue_mask,
    )


def _validate_geometry_arrays(
    x: np.ndarray,
    y: np.ndarray,
    tissue_fractions: np.ndarray,
    tile_index: np.ndarray | None,
) -> np.ndarray:
    """Validate geometry arrays and return a canonical tile_index."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    n_tiles = int(x.shape[0])
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"x and y must be 1D arrays of equal length, got {x.shape} and {y.shape}")
    if tissue_fractions.ndim != 1 or tissue_fractions.shape[0] != n_tiles:
        raise ValueError("tissue_fractions must be a 1D array aligned with x/y")
    if tile_index is None:
        return np.arange(n_tiles, dtype=np.int32)
    tile_index = np.asarray(tile_index, dtype=np.int32)
    if tile_index.ndim != 1 or tile_index.shape[0] != n_tiles:
        raise ValueError("tile_index must be a 1D array aligned with x/y")
    return tile_index


@dataclass
class TileGeometry:
    """Core tile geometry returned by :func:`generate_tiles`.

    All fields are required (no ``None`` defaults) so the caller always
    knows the exact tile layout after generation.
    """

    x: np.ndarray
    y: np.ndarray
    tissue_fractions: np.ndarray
    requested_tile_size_px: int
    requested_spacing_um: float
    read_level: int
    read_tile_size_px: int
    read_spacing_um: float
    tile_size_lv0: int
    is_within_tolerance: bool
    base_spacing_um: float
    slide_dimensions: list[int]
    level_downsamples: list[float]
    overlap: float
    min_tissue_fraction: float
    tile_index: np.ndarray | None = None
    tissue_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.tile_index = _validate_geometry_arrays(
            self.x, self.y, self.tissue_fractions, self.tile_index,
        )


@dataclass
class TilingResult:
    """Full provenance-enriched tiling result returned by :func:`preprocess_slide`.

    Composes a :class:`TileGeometry` (accessible via ``tiles``) with
    provenance, segmentation, and filtering metadata.  Geometry fields
    are also accessible directly on this object via ``__getattr__``
    delegation, so ``result.x`` and ``result.tiles.x`` are equivalent.
    """

    tiles: TileGeometry
    # -- provenance --
    sample_id: str
    image_path: str | Path
    backend: str
    requested_backend: str
    # -- tiling config --
    tolerance: float
    step_px_lv0: int
    tissue_method: str
    # -- segmentation --
    seg_downsample: int
    seg_level: int
    seg_spacing_um: float
    seg_sthresh: int
    seg_sthresh_up: int
    seg_mthresh: int
    seg_close: int
    # -- filtering --
    ref_tile_size_px: int
    a_t: float
    a_h: float
    filter_white: bool
    filter_black: bool
    white_threshold: int
    black_threshold: int
    fraction_threshold: float
    filter_grayspace: bool = False
    grayspace_saturation_threshold: float = 0.05
    grayspace_fraction_threshold: float = 0.6
    filter_blur: bool = False
    blur_threshold: float = 50.0
    qc_spacing_um: float = 2.0
    # -- optional (legitimately None in some paths) --
    mask_path: str | Path | None = None
    tissue_mask_tissue_value: int | None = None
    mask_level: int | None = None
    mask_spacing_um: float | None = None
    annotation: str | None = None
    selection_strategy: str | None = None
    output_mode: str | None = None

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)
        if self.mask_path is not None:
            self.mask_path = Path(self.mask_path)

    @property
    def num_tiles(self) -> int:
        return len(self.tiles.x)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying TileGeometry."""
        try:
            return getattr(object.__getattribute__(self, "tiles"), name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute {name!r}"
            ) from None

def canonicalize_tiling_result(tiles: TileGeometry) -> TileGeometry:
    """Deduplicate and sort tile coordinates in column-major order."""
    coords = np.column_stack((tiles.x, tiles.y))
    fracs = tiles.tissue_fractions
    if len(coords) > 1:
        _, unique_idx = np.unique(coords, axis=0, return_index=True)
        unique_idx.sort()
        coords = coords[unique_idx]
        fracs = fracs[unique_idx]
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        coords = coords[order]
        fracs = fracs[order]

    return replace(
        tiles,
        x=coords[:, 0],
        y=coords[:, 1],
        tissue_fractions=fracs,
        tile_index=np.arange(len(coords), dtype=np.int32),
    )


def resolve_base_spacing_um(result: TilingResult | TileGeometry) -> float:
    if result.base_spacing_um is None:
        raise ValueError("result is missing base_spacing_um metadata")
    return float(result.base_spacing_um)


def generate_tiles(
    slide_dimensions: tuple[int, int],
    contours: ContourResult,
    *,
    requested_tile_size_px: int = 256,
    requested_spacing_um: float = 0.5,
    base_spacing_um: float,
    level_downsamples: list[float] | list[tuple[float, float]],
    overlap: float = 0.0,
    min_tissue_fraction: float = 0.5,
    tolerance: float = 0.05,
    num_workers: int = 1,
) -> TileGeometry:
    normalized_downsamples = _normalize_level_downsamples(level_downsamples)
    if requested_spacing_um < base_spacing_um:
        relative_diff = abs(base_spacing_um - requested_spacing_um) / requested_spacing_um
        if relative_diff > tolerance:
            raise ValueError(
                f"Desired spacing ({requested_spacing_um}) is smaller than the "
                f"whole-slide image starting spacing ({base_spacing_um}) and does not "
                f"fall within tolerance ({tolerance:.0%})"
            )

    level_sel = select_level(
        requested_spacing_um=requested_spacing_um,
        level0_spacing_um=base_spacing_um,
        level_downsamples=[
            (float(downsample), float(downsample))
            for downsample in normalized_downsamples
        ],
        tolerance=tolerance,
    )
    if level_sel.is_within_tolerance:
        read_tile_size_px = requested_tile_size_px
    else:
        read_tile_size_px = round(
            requested_tile_size_px * requested_spacing_um / level_sel.read_spacing_um
        )
    # The level-0 footprint should reflect the actual read geometry that
    # produced the tile crop, not the nominal requested spacing contract.
    tile_size_lv0 = round(
        read_tile_size_px * level_sel.read_spacing_um / base_spacing_um
    )
    step_lv0 = max(1, round(tile_size_lv0 * (1.0 - overlap)))
    slide_w, slide_h = slide_dimensions

    def _empty_result() -> TileGeometry:
        return TileGeometry(
            x=np.empty(0, dtype=np.int64),
            y=np.empty(0, dtype=np.int64),
            tissue_fractions=np.empty(0, dtype=np.float32),
            requested_tile_size_px=requested_tile_size_px,
            requested_spacing_um=requested_spacing_um,
            read_level=level_sel.level,
            read_tile_size_px=read_tile_size_px,
            read_spacing_um=level_sel.read_spacing_um,
            tile_size_lv0=tile_size_lv0,
            is_within_tolerance=level_sel.is_within_tolerance,
            tissue_mask=contours.mask,
            base_spacing_um=base_spacing_um,
            slide_dimensions=list(slide_dimensions),
            level_downsamples=list(normalized_downsamples),
            overlap=overlap,
            min_tissue_fraction=min_tissue_fraction,
        )

    if len(contours.contours) == 0:
        return _empty_result()

    def _process_contour(idx: int) -> tuple[np.ndarray, np.ndarray]:
        contour = contours.contours[idx]
        return _tiles_for_contour(
            contour=contour,
            contour_holes=contours.holes[idx],
            tissue_mask=contours.mask,
            slide_dimensions=slide_dimensions,
            tile_size_lv0=tile_size_lv0,
            step_lv0=step_lv0,
            min_tissue_fraction=min_tissue_fraction,
        )

    if num_workers > 1 and len(contours.contours) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_contour, range(len(contours.contours))))
    else:
        results = [_process_contour(i) for i in range(len(contours.contours))]

    all_coords = []
    all_fracs = []
    for coords, fracs in results:
        if len(coords) > 0:
            all_coords.append(coords)
            all_fracs.append(fracs)

    if not all_coords:
        return _empty_result()

    merged_coords = np.concatenate(all_coords, axis=0)
    merged_fracs = np.concatenate(all_fracs, axis=0)
    return canonicalize_tiling_result(
        TileGeometry(
            x=merged_coords[:, 0],
            y=merged_coords[:, 1],
            tissue_fractions=merged_fracs,
            requested_tile_size_px=requested_tile_size_px,
            requested_spacing_um=requested_spacing_um,
            read_level=level_sel.level,
            read_tile_size_px=read_tile_size_px,
            read_spacing_um=level_sel.read_spacing_um,
            tile_size_lv0=tile_size_lv0,
            is_within_tolerance=level_sel.is_within_tolerance,
            tissue_mask=contours.mask,
            base_spacing_um=base_spacing_um,
            slide_dimensions=list(slide_dimensions),
            level_downsamples=list(normalized_downsamples),
            overlap=overlap,
            min_tissue_fraction=min_tissue_fraction,
        )
    )


def _tiles_for_contour(
    contour: np.ndarray,
    contour_holes: list[np.ndarray],
    tissue_mask: np.ndarray,
    slide_dimensions: tuple[int, int],
    tile_size_lv0: int,
    step_lv0: int,
    min_tissue_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    slide_w, slide_h = slide_dimensions
    x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)
    x_start = max(x_cont, 0)
    y_start = max(y_cont, 0)
    x_end = min(x_cont + w_cont, slide_w)
    y_end = min(y_cont + h_cont, slide_h)

    if x_end <= x_start or y_end <= y_start:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)

    xs = np.arange(x_start, x_end, step_lv0, dtype=np.int64)
    ys = np.arange(y_start, y_end, step_lv0, dtype=np.int64)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    candidates = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    contour_mask = _build_contour_tissue_mask(
        contour=contour,
        contour_holes=contour_holes,
        tissue_mask=tissue_mask,
        slide_dimensions=slide_dimensions,
    )
    fractions = _compute_tissue_fractions(
        candidates=candidates,
        tissue_mask=contour_mask,
        tile_size_lv0=tile_size_lv0,
        slide_dimensions=slide_dimensions,
    )
    keep = fractions >= min_tissue_fraction
    return candidates[keep], fractions[keep]


def _build_contour_tissue_mask(
    contour: np.ndarray,
    contour_holes: list[np.ndarray],
    tissue_mask: np.ndarray,
    slide_dimensions: tuple[int, int],
) -> np.ndarray:
    mask_h, mask_w = tissue_mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = mask_w / slide_w
    scale_y = mask_h / slide_h

    contour_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    contour_mask_scaled = contour.copy().astype(np.float64)
    contour_mask_scaled[:, 0, 0] *= scale_x
    contour_mask_scaled[:, 0, 1] *= scale_y
    contour_mask_scaled = np.round(contour_mask_scaled).astype(np.int32)
    cv2.drawContours(contour_mask, [contour_mask_scaled], -1, 1, thickness=-1)

    if contour_holes:
        holes_scaled = []
        for hole in contour_holes:
            hole_scaled = hole.copy().astype(np.float64)
            hole_scaled[:, 0, 0] *= scale_x
            hole_scaled[:, 0, 1] *= scale_y
            holes_scaled.append(np.round(hole_scaled).astype(np.int32))
        cv2.drawContours(contour_mask, holes_scaled, -1, 0, thickness=-1)

    return contour_mask * (tissue_mask > 0).astype(np.uint8)


def _compute_tissue_fractions(
    candidates: np.ndarray,
    tissue_mask: np.ndarray,
    tile_size_lv0: int,
    slide_dimensions: tuple[int, int],
) -> np.ndarray:
    mask_h, mask_w = tissue_mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = mask_w / slide_w
    scale_y = mask_h / slide_h
    binary = (tissue_mask > 0).astype(np.float64)
    integral = cv2.integral(binary)

    tile_w_mask = max(1, round(tile_size_lv0 * scale_x))
    tile_h_mask = max(1, round(tile_size_lv0 * scale_y))
    tile_area = tile_w_mask * tile_h_mask

    mask_origins = project_discrete_grid_origins(
        candidates,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    x1 = np.clip(mask_origins[:, 0], 0, mask_w)
    y1 = np.clip(mask_origins[:, 1], 0, mask_h)
    x2 = np.clip(mask_origins[:, 0] + tile_w_mask, 0, mask_w)
    y2 = np.clip(mask_origins[:, 1] + tile_h_mask, 0, mask_h)

    tissue_sum = (
        integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    )
    fractions = (tissue_sum / tile_area).astype(np.float32)
    return np.clip(fractions, 0.0, 1.0)


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


def _reduce_mask_channels(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        return np.asarray(mask[..., 0])
    return np.asarray(mask)


def _is_discrete_binary_mask(mask: np.ndarray, *, tissue_value: int) -> bool:
    values = np.unique(mask.astype(np.int64, copy=False))
    non_tissue = [int(value) for value in values.tolist() if int(value) != tissue_value]
    return len(values) <= 2 and len(non_tissue) <= 1


def _read_exact_tiff_mask_level(mask_path: str | Path, *, mask_level: int) -> np.ndarray:
    path = Path(mask_path)
    with Image.open(path) as image:
        n_frames = int(getattr(image, "n_frames", 1))
        if mask_level >= n_frames:
            raise ValueError(
                f"Mask level {mask_level} is unavailable in TIFF mask {path} (n_frames={n_frames})"
            )
        image.seek(mask_level)
        return np.asarray(image)


def _read_mask_level(
    *,
    mask_path: str | Path,
    mask_slide,
    mask_level: int,
    tissue_value: int,
) -> np.ndarray:
    mask_size = mask_slide.level_dimensions[mask_level]
    backend_mask = mask_slide.read_region((0, 0), mask_level, mask_size)
    backend_mask = _reduce_mask_channels(backend_mask)
    if _is_discrete_binary_mask(backend_mask, tissue_value=tissue_value):
        return backend_mask.astype(np.uint8, copy=False)

    suffix = Path(mask_path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        exact_mask = _reduce_mask_channels(
            _read_exact_tiff_mask_level(mask_path, mask_level=mask_level)
        )
        if _is_discrete_binary_mask(exact_mask, tissue_value=tissue_value):
            return exact_mask.astype(np.uint8, copy=False)
    values = np.unique(backend_mask.astype(np.int64, copy=False))
    raise ValueError(
        "Precomputed tissue mask read produced non-discrete labels "
        f"at level {mask_level}: {values.tolist()[:16]}"
    )


def _select_mask_level(
    *,
    mask_slide,
    requested_spacing_um: float,
) -> tuple[int, float]:
    read_spacings = [
        float(mask_slide.spacing) * float(downsample[0] if isinstance(downsample, tuple) else downsample)
        for downsample in mask_slide.level_downsamples
    ]
    level = int(np.argmin([abs(spacing_um - requested_spacing_um) for spacing_um in read_spacings]))
    spacing_um = read_spacings[level]
    while level > 0 and spacing_um > requested_spacing_um:
        level -= 1
        spacing_um = read_spacings[level]
    return level, spacing_um


def load_precomputed_tissue_mask(
    *,
    mask_path: str | Path,
    slide,
    seg_level: int,
    tissue_value: int,
) -> tuple[np.ndarray, int, float]:
    mask_slide = open_slide(mask_path, backend=slide.backend_name)
    try:
        seg_size = slide.level_dimensions[seg_level]
        seg_spacing_um = float(slide.spacing) * float(
            _normalize_level_downsamples(slide.level_downsamples)[seg_level]
        )
        mask_level, mask_spacing_um = _select_mask_level(
            mask_slide=mask_slide,
            requested_spacing_um=seg_spacing_um,
        )
        raw_mask = _read_mask_level(
            mask_path=mask_path,
            mask_slide=mask_slide,
            mask_level=mask_level,
            tissue_value=tissue_value,
        )
    finally:
        mask_slide.close()

    if raw_mask.shape[:2] != (int(seg_size[1]), int(seg_size[0])):
        raw_mask = cv2.resize(
            raw_mask.astype(np.uint8, copy=False),
            (int(seg_size[0]), int(seg_size[1])),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = np.where(raw_mask == tissue_value, 255, 0).astype(np.uint8)
    return mask, mask_level, mask_spacing_um


def resolve_tissue_mask(
    *,
    slide,
    tissue_method: str | None = None,
    tissue_mask_path: str | Path | None,
    tissue_mask_tissue_value: int = 1,
    sthresh: int = 8,
    sthresh_up: int = 255,
    mthresh: int = 7,
    close: int = 4,
    seg_downsample: int = 64,
    sam2_checkpoint_path: str | Path | None = None,
    sam2_config_path: str | Path | None = None,
    sam2_device: str = "cpu",
) -> ResolvedTissueMask:
    normalized_downsamples = _normalize_level_downsamples(slide.level_downsamples)
    seg_level = select_level_for_downsample(
        float(seg_downsample),
        [(float(ds), float(ds)) for ds in normalized_downsamples],
    )
    seg_spacing_um = float(slide.spacing) * float(normalized_downsamples[seg_level])

    if tissue_mask_path is not None:
        mask, mask_level, mask_spacing_um = load_precomputed_tissue_mask(
            mask_path=tissue_mask_path,
            slide=slide,
            seg_level=seg_level,
            tissue_value=int(tissue_mask_tissue_value),
        )
        return ResolvedTissueMask(
            tissue_mask=mask,
            tissue_method="precomputed_mask",
            seg_downsample=int(seg_downsample),
            seg_level=seg_level,
            seg_spacing_um=seg_spacing_um,
            mask_path=tissue_mask_path,
            tissue_mask_tissue_value=int(tissue_mask_tissue_value),
            mask_level=mask_level,
            mask_spacing_um=mask_spacing_um,
        )

    if not tissue_method:
        raise ValueError(
            "tissue_method is required when no precomputed tissue mask is provided"
        )

    seg_size = slide.level_dimensions[seg_level]
    seg_image = slide.read_region((0, 0), seg_level, seg_size)
    segmentation_config = SegmentationConfig(
        method=tissue_method,
        downsample=seg_downsample,
        sthresh=sthresh,
        sthresh_up=sthresh_up,
        mthresh=mthresh,
        close=close,
        sam2_checkpoint_path=(
            Path(sam2_checkpoint_path) if sam2_checkpoint_path is not None else None
        ),
        sam2_config_path=(
            Path(sam2_config_path) if sam2_config_path is not None else None
        ),
        sam2_device=sam2_device,
    )
    mask = segment_tissue_image(
        seg_image,
        config=segmentation_config,
    )
    return ResolvedTissueMask(
        tissue_mask=mask,
        tissue_method=segmentation_config.method,
        seg_downsample=seg_downsample,
        seg_level=seg_level,
        seg_spacing_um=seg_spacing_um,
    )


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
        annotation=annotation,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
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
    "ContourResult",
    "TileGeometry",
    "TilingResult",
    "ResolvedTissueMask",
    "canonicalize_tiling_result",
    "detect_contours",
    "generate_tiles",
    "load_precomputed_tissue_mask",
    "resolve_tissue_mask",
    "build_tiling_result_from_mask",
    "normalize_artifact_path",
    "open_slide",
    "preprocess_slide",
    "resolve_base_spacing_um",
    "select_level",
    "select_level_for_downsample",
    "validate_tiling_result_provenance",
]
