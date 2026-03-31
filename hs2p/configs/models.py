from dataclasses import dataclass

from .loader import default_config

AUTO_BACKEND = "auto"

_DEFAULT_TILING = default_config.tiling
_DEFAULT_TILING_PARAMS = _DEFAULT_TILING.params
_DEFAULT_SEGMENTATION = _DEFAULT_TILING.seg_params
_DEFAULT_FILTERING = _DEFAULT_TILING.filter_params


@dataclass(frozen=True)
class TilingConfig:
    """Control tile extraction at a target physical resolution."""

    target_spacing_um: float
    target_tile_size_px: int
    tolerance: float
    overlap: float
    tissue_threshold: float
    use_padding: bool = bool(_DEFAULT_TILING_PARAMS.use_padding)
    backend: str = AUTO_BACKEND

    @property
    def requested_backend(self) -> str:
        """Backend requested by config before runtime auto-resolution."""
        return self.backend


@dataclass(frozen=True)
class SegmentationConfig:
    """Control tissue segmentation before coordinate extraction."""

    downsample: int = int(_DEFAULT_SEGMENTATION.downsample)
    sthresh: int = int(_DEFAULT_SEGMENTATION.sthresh)
    sthresh_up: int = int(_DEFAULT_SEGMENTATION.sthresh_up)
    mthresh: int = int(_DEFAULT_SEGMENTATION.mthresh)
    close: int = int(_DEFAULT_SEGMENTATION.close)
    use_otsu: bool = bool(_DEFAULT_SEGMENTATION.use_otsu)
    use_hsv: bool = bool(_DEFAULT_SEGMENTATION.use_hsv)


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
    downsample: int = 32
