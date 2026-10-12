from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hs2p.fileops import validate_annotation_name

from .loader import default_config
from .values import normalize_config_fields

AUTO_BACKEND = "auto"
# Backend names accepted by configuration validation. Kept in lockstep with the runtime
# registry in ``hs2p.wsi.reader._BACKENDS`` (plus ``auto``); duplicated here so the config
# layer validates without importing the (heavier) WSI reader package at model-definition time.
VALID_BACKENDS: frozenset[str] = frozenset(
    {AUTO_BACKEND, "cucim", "asap", "openslide", "pil", "tifffile", "vips"}
)


def _validate_backend_name(value: str, *, field: str) -> str:
    """Reject unknown backend names for a config field.

    Both ``backend`` and ``mask_backend`` accept only ``auto`` plus the concrete
    backends. Unknown strings are configuration errors — including when a
    :class:`TilingConfig` is constructed directly in Python. ``None`` and other non-string
    values are rejected earlier, by the type check on the field's ``str`` annotation.
    """
    if value not in VALID_BACKENDS:
        raise ValueError(
            f"tiling.{field} must be one of {sorted(VALID_BACKENDS)}, got {value!r}"
        )
    return value


_DEFAULT_TILING = default_config.tiling
_DEFAULT_SEGMENTATION = _DEFAULT_TILING.seg_params
_DEFAULT_FILTERING = _DEFAULT_TILING.filter_params
_DEFAULT_PREVIEW = _DEFAULT_TILING.preview
_DEFAULT_MASKS = _DEFAULT_TILING.masks


@dataclass(frozen=True, kw_only=True)
class TilingConfig:
    """Control tile extraction at a target physical resolution."""

    requested_spacing_um: float
    requested_tile_size_px: int
    tolerance: float
    overlap: float
    backend: str = AUTO_BACKEND
    mask_backend: str = AUTO_BACKEND
    # Provenance: the backends originally requested in config, preserved verbatim across the
    # runtime ``replace(tiling, backend=<resolved>)`` auto-resolution step. Default ``None`` is
    # a sentinel meaning "not explicitly supplied" — ``__post_init__`` fills it from the
    # as-constructed ``backend``/``mask_backend`` so a freshly built config reports what was
    # requested, while a resolved config keeps the original request rather than echoing the
    # resolved value back as the request.
    requested_backend: str | None = None
    requested_mask_backend: str | None = None

    def __post_init__(self) -> None:
        normalize_config_fields(self)
        _validate_backend_name(self.backend, field="backend")
        _validate_backend_name(self.mask_backend, field="mask_backend")
        if self.requested_backend is None:
            object.__setattr__(self, "requested_backend", self.backend)
        else:
            _validate_backend_name(self.requested_backend, field="requested_backend")
        if self.requested_mask_backend is None:
            object.__setattr__(self, "requested_mask_backend", self.mask_backend)
        else:
            _validate_backend_name(
                self.requested_mask_backend, field="requested_mask_backend"
            )


@dataclass(frozen=True, kw_only=True)
class SegmentationConfig:
    """Control tissue segmentation before coordinate extraction."""

    method: str
    downsample: int = int(_DEFAULT_SEGMENTATION.downsample)
    sthresh: int = int(_DEFAULT_SEGMENTATION.sthresh)
    sthresh_up: int = int(_DEFAULT_SEGMENTATION.sthresh_up)
    mthresh: int = int(_DEFAULT_SEGMENTATION.mthresh)
    close: int = int(_DEFAULT_SEGMENTATION.close)
    sam2_checkpoint_path: Path | None = (
        Path(_DEFAULT_SEGMENTATION.sam2_checkpoint_path)
        if getattr(_DEFAULT_SEGMENTATION, "sam2_checkpoint_path", None)
        else None
    )
    sam2_config_path: Path | None = (
        Path(_DEFAULT_SEGMENTATION.sam2_config_path)
        if getattr(_DEFAULT_SEGMENTATION, "sam2_config_path", None)
        else None
    )
    sam2_device: str = str(getattr(_DEFAULT_SEGMENTATION, "sam2_device", "cpu"))
    sam2_num_workers: int | None = (
        int(_DEFAULT_SEGMENTATION.sam2_num_workers)
        if getattr(_DEFAULT_SEGMENTATION, "sam2_num_workers", None) is not None
        else None
    )

    def __post_init__(self) -> None:
        normalize_config_fields(self)


@dataclass(frozen=True, kw_only=True)
class FilterConfig:
    """Control contour and tile-level filtering after segmentation."""

    ref_tile_size: int = int(_DEFAULT_FILTERING.ref_tile_size)
    a_t: int = int(_DEFAULT_FILTERING.a_t)
    a_h: int = int(_DEFAULT_FILTERING.a_h)
    filter_white: bool = bool(_DEFAULT_FILTERING.filter_white)
    filter_black: bool = bool(_DEFAULT_FILTERING.filter_black)
    white_threshold: int = int(_DEFAULT_FILTERING.white_threshold)
    black_threshold: int = int(_DEFAULT_FILTERING.black_threshold)
    fraction_threshold: float = float(_DEFAULT_FILTERING.fraction_threshold)
    filter_grayspace: bool = bool(_DEFAULT_FILTERING.filter_grayspace)
    grayspace_saturation_threshold: float = float(
        _DEFAULT_FILTERING.grayspace_saturation_threshold
    )
    grayspace_fraction_threshold: float = float(
        _DEFAULT_FILTERING.grayspace_fraction_threshold
    )
    filter_blur: bool = bool(_DEFAULT_FILTERING.filter_blur)
    blur_threshold: float = float(_DEFAULT_FILTERING.blur_threshold)
    qc_spacing_um: float = float(_DEFAULT_FILTERING.qc_spacing_um)

    def __post_init__(self) -> None:
        normalize_config_fields(self)


@dataclass(frozen=True, kw_only=True)
class PreviewConfig:
    """Control preview generation in batch tiling."""

    save_mask_preview: bool = False
    save_tiling_preview: bool = False
    downsample: int = int(_DEFAULT_PREVIEW.downsample)
    tissue_contour_color: tuple[int, int, int] = tuple(
        _DEFAULT_PREVIEW.tissue_contour_color
    )
    mask_overlay_alpha: float = float(_DEFAULT_PREVIEW.mask_overlay_alpha)

    def __post_init__(self) -> None:
        normalize_config_fields(self)
        if any(channel < 0 or channel > 255 for channel in self.tissue_contour_color):
            raise ValueError(
                "tissue_contour_color must be a length-3 RGB tuple with values in [0, 255]"
            )
        if not 0.0 <= self.mask_overlay_alpha <= 1.0:
            raise ValueError("mask_overlay_alpha must be between 0.0 and 1.0")


def _label_values(entry: Any) -> tuple[Any, ...]:
    """Raw mask values of one ``pixel_mapping`` entry: a scalar, or a sequence of values
    merged into one class."""
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        return tuple(entry)
    return (entry,)


def validate_pixel_mapping(pixel_mapping: Mapping[str, Any]) -> None:
    """Validate the annotation ``pixel_mapping`` up front, before any slide is opened.

    ``pixel_mapping`` is the user's own label vocabulary except for ``"merged"``, which names
    structural merged coordinate output. Each label must map to a distinct integer in the
    preview-safe range ``[0, 255]`` so all consumers preserve it without silent wrapping, or
    to a non-empty list of such integers merged into one class. No raw value may be claimed
    twice, within a label or across labels.

    Label *names* become on-disk path components (``output_dir/tiles/<label>/...`` for
    per-annotation artifacts), so they must be safe single path components — no separators,
    no ``.``/``..`` — to keep artifacts inside the run's output directory.
    """
    seen: dict[int, str] = {}
    for annotation, entry in pixel_mapping.items():
        validate_annotation_name(annotation)
        values = _label_values(entry)
        if not values:
            raise ValueError(
                f"pixel_mapping['{annotation}'] must list at least one integer label value"
            )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"pixel_mapping['{annotation}'] must be an integer label value, "
                    f"got {value!r}"
                )
            ivalue = int(value)
            if ivalue < 0 or ivalue > 255:
                raise ValueError(
                    f"pixel_mapping['{annotation}']={ivalue} is outside the supported "
                    "label range [0, 255]"
                )
            if seen.get(ivalue) == annotation:
                raise ValueError(
                    f"pixel_mapping['{annotation}'] lists {ivalue} more than once"
                )
            if ivalue in seen:
                raise ValueError(
                    "pixel_mapping values must be unique: "
                    f"'{annotation}' and '{seen[ivalue]}' both map to {ivalue}"
                )
            seen[ivalue] = annotation


def validate_color_mapping(
    *,
    pixel_mapping: Mapping[str, Any],
    color_mapping: Mapping[str, Sequence[int] | None],
) -> None:
    missing_annotations = sorted(set(pixel_mapping.keys()) - set(color_mapping.keys()))
    if missing_annotations:
        raise ValueError(
            "color_mapping is missing annotation keys required by pixel_mapping: "
            + ", ".join(missing_annotations)
        )
    unexpected_annotations = sorted(set(color_mapping.keys()) - set(pixel_mapping.keys()))
    if unexpected_annotations:
        raise ValueError(
            "color_mapping has unexpected annotation keys: "
            + ", ".join(unexpected_annotations)
        )

    for annotation, color in color_mapping.items():
        if color is None:
            continue
        if isinstance(color, (str, bytes)):
            raise ValueError(
                f"color_mapping['{annotation}'] must be None or a length-3 RGB sequence"
            )
        if not isinstance(color, Sequence) or len(color) != 3:
            raise ValueError(
                f"color_mapping['{annotation}'] must be None or a length-3 RGB sequence"
            )
        if any(
            (not isinstance(c, (int, np.integer)) or c < 0 or c > 255) for c in color
        ):
            raise ValueError(
                f"color_mapping['{annotation}'] must contain integers in [0, 255]"
            )


def _checked_min_coverage(
    min_coverage: Mapping[str, Any], *, labels: Mapping[str, Any]
) -> dict[str, float]:
    """Each threshold names a declared label and is a fraction in ``[0, 1]``."""
    unknown = sorted(set(min_coverage) - set(labels))
    if unknown:
        raise ValueError("min_coverage references unknown labels: " + ", ".join(unknown))
    checked: dict[str, float] = {}
    for label, coverage in min_coverage.items():
        if isinstance(coverage, (bool, np.bool_)) or not isinstance(
            coverage, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"min_coverage['{label}'] must be a float in [0, 1], got {coverage!r}"
            )
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"min_coverage['{label}']={coverage} is outside [0, 1]")
        checked[label] = float(coverage)
    return checked


@dataclass(frozen=True, kw_only=True)
class MaskSamplingConfig:
    """The annotation-mask vocabulary: which mask values form which label, how each label
    is drawn in previews, which labels are sampled and at what coverage, and how sampled
    coordinates are selected and written."""

    # Label name -> the raw mask values merged into that label. A scalar is accepted and
    # stored as a one-value tuple.
    pixel_mapping: Mapping[str, tuple[int, ...]] = field(hash=False)
    # Label name -> RGB overlay color in previews (``None`` leaves the label undrawn).
    # Lists every label of ``pixel_mapping`` when given; ``None`` draws no overlay.
    colors: Mapping[str, tuple[int, int, int] | None] | None = field(
        default=None, hash=False
    )
    # Label name -> minimum fraction of a tile the label must cover for the tile to be
    # sampled. Only listed labels are sampled.
    min_coverage: Mapping[str, float] = field(hash=False)
    output_mode: Literal["merged", "per_annotation"] = "per_annotation"
    strategy: Literal["joint", "independent"] = "joint"

    def __post_init__(self) -> None:
        normalize_config_fields(self)
        validate_pixel_mapping(self.pixel_mapping)
        object.__setattr__(
            self,
            "pixel_mapping",
            {
                str(label): tuple(int(value) for value in _label_values(entry))
                for label, entry in self.pixel_mapping.items()
            },
        )
        if self.colors is not None:
            validate_color_mapping(
                pixel_mapping=self.pixel_mapping, color_mapping=self.colors
            )
            object.__setattr__(
                self,
                "colors",
                {
                    label: None if color is None else tuple(int(c) for c in color)
                    for label, color in self.colors.items()
                },
            )
        object.__setattr__(
            self,
            "min_coverage",
            _checked_min_coverage(self.min_coverage, labels=self.pixel_mapping),
        )


#: The shipped ``tiling.masks`` block: the binary ``{background, tissue}`` vocabulary with
#: only ``tissue`` sampled. A mask config with this vocabulary and sampled set is binary
#: tissue tiling, and ``min_coverage["tissue"]`` is its tissue-coverage threshold.
DEFAULT_MASK_SAMPLING = MaskSamplingConfig(
    pixel_mapping=dict(_DEFAULT_MASKS.pixel_mapping),
    colors=dict(_DEFAULT_MASKS.colors),
    min_coverage={
        label: coverage
        for label, coverage in _DEFAULT_MASKS.min_coverage.items()
        if coverage is not None
    },
    output_mode=_DEFAULT_MASKS.output_mode,
    strategy=_DEFAULT_MASKS.strategy,
)


@dataclass(frozen=True, kw_only=True)
class RunSettings:
    """Declared types of the top-level and ``speed`` scalars the CLI reads outside the
    typed configs. Config loading checks those values against these annotations; the CLI
    does not build this object."""

    resume: bool
    save_tiles: bool
    seed: int
    num_workers: int
    jpeg_backend: str
