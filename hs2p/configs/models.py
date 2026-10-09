from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass(frozen=True, kw_only=True)
class TilingConfig:
    """Control tile extraction at a target physical resolution."""

    requested_spacing_um: float
    requested_tile_size_px: int
    tolerance: float
    overlap: float
    # Resolved per-class minimum coverage fractions; ``min_coverage["tissue"]`` is the
    # tissue threshold. Excluded from __hash__ so the frozen dataclass stays hashable
    # despite the mapping field.
    min_coverage: Mapping[str, float] = field(hash=False)
    backend: str = AUTO_BACKEND
    mask_backend: str = AUTO_BACKEND
    independent_sampling: bool = False
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
