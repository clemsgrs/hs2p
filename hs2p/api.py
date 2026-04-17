"""Public API surface — re-exports from tiling sub-modules."""

from hs2p.artifacts import (
    CompatibilitySpec,
    SlideSpec,
    TilingArtifacts,
    load_tiling_result,
    maybe_load_existing_artifacts,
    save_tiling_result,
    validate_tiling_artifacts,
    write_process_list,
)
from hs2p.configs import FilterConfig, PreviewConfig, SegmentationConfig, TilingConfig
from hs2p.tiling.orchestration import (
    overlay_mask_on_slide,
    tile_slide,
    tile_slides,
    write_tiling_preview,
)
from hs2p.tiling.tar import (
    _annotation_tar_stem,
    _apply_qc_filtering_to_result,
    _needs_pixel_filtering,
    extract_tiles_to_tar,
)
from hs2p.wsi import CoordinateOutputMode, CoordinateSelectionStrategy

__all__ = [
    "CompatibilitySpec",
    "CoordinateOutputMode",
    "CoordinateSelectionStrategy",
    "FilterConfig",
    "PreviewConfig",
    "SegmentationConfig",
    "SlideSpec",
    "TilingArtifacts",
    "TilingConfig",
    "extract_tiles_to_tar",
    "load_tiling_result",
    "maybe_load_existing_artifacts",
    "overlay_mask_on_slide",
    "save_tiling_result",
    "tile_slide",
    "tile_slides",
    "validate_tiling_artifacts",
    "write_process_list",
    "write_tiling_preview",
]
