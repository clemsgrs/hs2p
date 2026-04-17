from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _validate_geometry_arrays(
    x: np.ndarray,
    y: np.ndarray,
    tissue_fractions: np.ndarray,
    tile_index: np.ndarray | None,
) -> np.ndarray:
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


@dataclass(frozen=True)
class ResolvedAnnotationMasks:
    """Multi-label annotation mask resolved for tiling, one binary mask per annotation."""

    masks: dict[str, np.ndarray]  # annotation name → binary uint8 mask (255=fg, 0=bg)
    tissue_method: str
    seg_downsample: int
    seg_level: int
    seg_spacing_um: float
    pixel_mapping: dict[str, int]
    mask_path: str | Path | None = None
    mask_level: int | None = None
    mask_spacing_um: float | None = None

    def __post_init__(self) -> None:
        if self.mask_path is not None:
            object.__setattr__(self, "mask_path", Path(self.mask_path))


@dataclass(frozen=True)
class Sam2Thumbnail:
    image: np.ndarray
    seg_level: int
    seg_spacing_um: float
    source_spacing_um: float
    resized: bool


@dataclass
class TileGeometry:
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
    # -- optional --
    mask_path: str | Path | None = None
    tissue_mask_tissue_value: int | None = None
    mask_level: int | None = None
    mask_spacing_um: float | None = None
    contours: ContourResult | None = None
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
        try:
            return getattr(object.__getattribute__(self, "tiles"), name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute {name!r}"
            ) from None


__all__ = [
    "ContourResult",
    "ResolvedAnnotationMasks",
    "ResolvedTissueMask",
    "Sam2Thumbnail",
    "TileGeometry",
    "TilingResult",
]
