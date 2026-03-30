from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LevelSelection:
    level: int
    effective_spacing_um: float
    is_within_tolerance: bool


def normalize_level_downsamples(
    level_downsamples: list[float | tuple[float, float]],
) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for value in level_downsamples:
        if isinstance(value, tuple):
            normalized.append((float(value[0]), float(value[1])))
        else:
            normalized.append((float(value), float(value)))
    return normalized


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
