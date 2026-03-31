from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedSamplingSpec:
    pixel_mapping: dict[str, int]
    color_mapping: dict[str, list[int] | None] | None
    tissue_percentage: dict[str, float | None]
    active_annotations: tuple[str, ...]


class CoordinateSelectionStrategy:
    MERGED_DEFAULT_TILING = "merged_default_tiling"
    JOINT_SAMPLING = "joint_sampling"
    INDEPENDENT_SAMPLING = "independent_sampling"


class CoordinateOutputMode:
    SINGLE_OUTPUT = "single_output"
    PER_ANNOTATION = "per_annotation"


__all__ = [
    "CoordinateOutputMode",
    "CoordinateSelectionStrategy",
    "ResolvedSamplingSpec",
]
