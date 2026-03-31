from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LevelSelection:
    level: int
    effective_spacing_um: float
    is_within_tolerance: bool


def project_discrete_grid_origins(
    coordinates: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Project level-0 origin coordinates into a discrete target grid.

    Origin coordinates are truncated toward zero so a top-left anchor stays
    within the same source pixel footprint after projection.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"coordinates must have shape (N, 2), got {coordinates.shape}"
        )
    projected = np.empty_like(coordinates, dtype=np.int64)
    projected[:, 0] = np.floor(
        coordinates[:, 0].astype(np.float64, copy=False) * float(scale_x)
    ).astype(np.int64)
    projected[:, 1] = np.floor(
        coordinates[:, 1].astype(np.float64, copy=False) * float(scale_y)
    ).astype(np.int64)
    return projected


def compute_level_downsamples(
    level_dimensions: list[tuple[int, int]],
) -> list[tuple[float, float]]:
    if not level_dimensions:
        return []
    width_0, height_0 = level_dimensions[0]
    return [
        (width_0 / float(width), height_0 / float(height))
        for width, height in level_dimensions
    ]


def compute_level_spacings(
    *,
    level0_spacing_um: float,
    level_downsamples: list[tuple[float, float]],
) -> list[float]:
    return [float(level0_spacing_um) * float(ds_x) for ds_x, _ in level_downsamples]


def select_level_for_downsample(
    requested_downsample: float,
    level_downsamples: list[tuple[float, float]],
) -> int:
    if len(level_downsamples) == 0:
        raise ValueError("level_downsamples must not be empty")
    return int(
        np.argmin(
            [
                abs(float(downsample[0]) - requested_downsample)
                for downsample in level_downsamples
            ]
        )
    )


def select_level(
    *,
    requested_spacing_um: float,
    level0_spacing_um: float,
    level_downsamples: list[tuple[float, float]],
    tolerance: float = 0.05,
) -> LevelSelection:
    level_spacings = compute_level_spacings(
        level0_spacing_um=level0_spacing_um,
        level_downsamples=level_downsamples,
    )
    level = int(
        np.argmin(
            [
                abs(effective_spacing - requested_spacing_um)
                for effective_spacing in level_spacings
            ]
        )
    )
    best_spacing = level_spacings[level]
    relative_error = abs(best_spacing - requested_spacing_um) / requested_spacing_um
    is_within_tolerance = relative_error <= tolerance

    if not is_within_tolerance:
        while level > 0 and best_spacing > requested_spacing_um:
            level -= 1
            best_spacing = level_spacings[level]
            relative_error = abs(best_spacing - requested_spacing_um) / requested_spacing_um
            is_within_tolerance = relative_error <= tolerance

    return LevelSelection(
        level=level,
        effective_spacing_um=best_spacing,
        is_within_tolerance=is_within_tolerance,
    )
