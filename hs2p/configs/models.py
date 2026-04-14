from dataclasses import dataclass
from pathlib import Path

from .loader import default_config

AUTO_BACKEND = "auto"

_DEFAULT_TILING = default_config.tiling
_DEFAULT_TILING_PARAMS = _DEFAULT_TILING.params
_DEFAULT_SEGMENTATION = _DEFAULT_TILING.seg_params
_DEFAULT_FILTERING = _DEFAULT_TILING.filter_params
_DEFAULT_PREVIEW = _DEFAULT_TILING.preview


@dataclass(frozen=True)
class TilingConfig:
    """Control tile extraction at a target physical resolution."""

    requested_spacing_um: float
    requested_tile_size_px: int
    tolerance: float
    overlap: float
    tissue_threshold: float
    backend: str = AUTO_BACKEND

    @property
    def requested_backend(self) -> str:
        """Backend requested by config before runtime auto-resolution."""
        return self.backend


@dataclass(frozen=True)
class SegmentationConfig:
    """Control tissue segmentation before coordinate extraction."""

    method: str = str(getattr(_DEFAULT_SEGMENTATION, "method", "hsv"))
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
    sam2_input_size: int = int(getattr(_DEFAULT_SEGMENTATION, "sam2_input_size", 1024))
    sam2_mask_threshold: float = float(
        getattr(_DEFAULT_SEGMENTATION, "sam2_mask_threshold", 0.0)
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
    mask_overlay_color: tuple[int, int, int] = tuple(_DEFAULT_PREVIEW.mask_overlay_color)
    mask_overlay_alpha: float = float(_DEFAULT_PREVIEW.mask_overlay_alpha)

    def __post_init__(self) -> None:
        color = tuple(int(channel) for channel in self.mask_overlay_color)
        if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
            raise ValueError(
                "mask_overlay_color must be a length-3 RGB tuple with values in [0, 255]"
            )
        alpha = float(self.mask_overlay_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("mask_overlay_alpha must be between 0.0 and 1.0")
        object.__setattr__(self, "mask_overlay_color", color)
        object.__setattr__(self, "mask_overlay_alpha", alpha)
