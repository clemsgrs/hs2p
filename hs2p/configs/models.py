from dataclasses import dataclass

from ._loader import default_config

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
    tissue_mask_tissue_value: int = int(_DEFAULT_TILING_PARAMS.tissue_mask_tissue_value)
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

    @property
    def tissue_method(self) -> str:
        """Resolved tissue segmentation method name."""
        if self.use_hsv:
            return "hsv"
        if self.use_otsu:
            return "otsu"
        return "threshold"


@dataclass(frozen=True)
class FilterConfig:
    """Control contour and tile-level filtering after segmentation."""

    ref_tile_size: int = int(_DEFAULT_FILTERING.ref_tile_size)
    a_t: int = int(_DEFAULT_FILTERING.a_t)
    a_h: int = int(_DEFAULT_FILTERING.a_h)
    max_n_holes: int = int(_DEFAULT_FILTERING.max_n_holes)
    filter_white: bool = bool(_DEFAULT_FILTERING.filter_white)
    filter_black: bool = bool(_DEFAULT_FILTERING.filter_black)
    white_threshold: int = int(_DEFAULT_FILTERING.white_threshold)
    black_threshold: int = int(_DEFAULT_FILTERING.black_threshold)
    fraction_threshold: float = float(_DEFAULT_FILTERING.fraction_threshold)


@dataclass(frozen=True)
class PreviewConfig:
    """Control preview generation in batch tiling."""

    save_mask_preview: bool = False
    save_tiling_preview: bool = False
    downsample: int = 32
