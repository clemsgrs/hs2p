from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .loader import default_config

AUTO_BACKEND = "auto"
# Backend names accepted by configuration validation. Kept in lockstep with the runtime
# registry in ``hs2p.wsi.reader._BACKENDS`` (plus ``auto``); duplicated here so the config
# layer validates without importing the (heavier) WSI reader package at model-definition time.
VALID_BACKENDS: frozenset[str] = frozenset(
    {AUTO_BACKEND, "cucim", "asap", "openslide", "vips"}
)


def _validate_backend_name(value: object, *, field: str) -> str:
    """Reject null and unknown backend names for a config field.

    Both ``backend`` and ``mask_backend`` accept only ``auto`` plus the four concrete
    backends. ``None`` and unknown strings are configuration errors — including when a
    :class:`TilingConfig` is constructed directly in Python.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"tiling.{field} must be one of {sorted(VALID_BACKENDS)}, got {value!r}"
        )
    if value not in VALID_BACKENDS:
        raise ValueError(
            f"tiling.{field} must be one of {sorted(VALID_BACKENDS)}, got {value!r}"
        )
    return value

_DEFAULT_TILING = default_config.tiling
_DEFAULT_TILING_PARAMS = _DEFAULT_TILING.params
_DEFAULT_SEGMENTATION = _DEFAULT_TILING.seg_params
_DEFAULT_FILTERING = _DEFAULT_TILING.filter_params
_DEFAULT_PREVIEW = _DEFAULT_TILING.preview
_DEFAULT_MASKS = _DEFAULT_TILING.masks


@dataclass(frozen=True)
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
        _validate_backend_name(self.backend, field="backend")
        _validate_backend_name(self.mask_backend, field="mask_backend")
        if self.requested_backend is None:
            object.__setattr__(self, "requested_backend", self.backend)
        if self.requested_mask_backend is None:
            object.__setattr__(self, "requested_mask_backend", self.mask_backend)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class PreviewConfig:
    """Control preview generation in batch tiling."""

    save_mask_preview: bool = False
    save_tiling_preview: bool = False
    downsample: int = int(_DEFAULT_PREVIEW.downsample)
    tissue_contour_color: tuple[int, int, int] = tuple(
        _DEFAULT_PREVIEW.tissue_contour_color
    )
    mask_overlay_alpha: float = float(_DEFAULT_PREVIEW.mask_overlay_alpha)

    def __init__(
        self,
        save_mask_preview: bool = False,
        save_tiling_preview: bool = False,
        downsample: int = int(_DEFAULT_PREVIEW.downsample),
        tissue_contour_color: tuple[int, int, int] = tuple(
            _DEFAULT_PREVIEW.tissue_contour_color
        ),
        mask_overlay_alpha: float = float(_DEFAULT_PREVIEW.mask_overlay_alpha),
    ) -> None:
        color = tuple(int(channel) for channel in tissue_contour_color)
        if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
            raise ValueError(
                "tissue_contour_color must be a length-3 RGB tuple with values in [0, 255]"
            )
        alpha = float(mask_overlay_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("mask_overlay_alpha must be between 0.0 and 1.0")
        object.__setattr__(self, "save_mask_preview", bool(save_mask_preview))
        object.__setattr__(self, "save_tiling_preview", bool(save_tiling_preview))
        object.__setattr__(self, "downsample", int(downsample))
        object.__setattr__(self, "tissue_contour_color", color)
        object.__setattr__(self, "mask_overlay_alpha", alpha)
