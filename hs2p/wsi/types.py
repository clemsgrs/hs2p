
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# One label maps to a raw mask value, or to a list of values merged into a single class.
PixelMapping = dict[str, int | list[int]]


def pixel_values(value: Any) -> tuple[Any, ...]:
    """Raw mask values of one ``pixel_mapping`` entry, scalar or list. The first one is the
    label's representative value wherever a single value is needed (recomposed rasters)."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True, kw_only=True)
class SamplingSpec:
    pixel_mapping: PixelMapping
    color_mapping: dict[str, list[int] | None] | None
    tissue_percentage: dict[str, float | None]
    active_annotations: tuple[str, ...]


class CoordinateSelectionStrategy:
    MERGED_DEFAULT_TILING = "merged_default_tiling"
    JOINT_SAMPLING = "joint_sampling"
    INDEPENDENT_SAMPLING = "independent_sampling"


class CoordinateOutputMode:
    MERGED = "merged"
    PER_ANNOTATION = "per_annotation"


__all__ = [
    "CoordinateOutputMode",
    "CoordinateSelectionStrategy",
    "PixelMapping",
    "SamplingSpec",
    "pixel_values",
]
