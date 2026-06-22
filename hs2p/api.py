"""Public API surface — re-exports from tiling sub-modules."""

from hs2p.artifacts import (
    CompatibilitySpec,
    SlideSpec,
    TilingArtifacts,
    load_tiling_result,
    maybe_load_existing_artifacts,
    save_tiling_result,
    validate_tiling_artifacts,
)
from hs2p.configs import FilterConfig, PreviewConfig, SegmentationConfig, TilingConfig
from hs2p.tiling.coverage import summarize_annotation_coverage
from hs2p.tiling.mask import resolve_annotation_masks
from hs2p.tiling.result import ResolvedAnnotationMasks
from hs2p.tiling.tar import (
    _annotation_tar_stem,
    _apply_qc_filtering_to_result,
    _needs_pixel_filtering,
    extract_tiles_to_tar,
)
from hs2p.tiling.orchestration import tile_slide, tile_slides, write_tiling_preview
from hs2p.wsi import CoordinateOutputMode, CoordinateSelectionStrategy, overlay_mask_on_slide

__all__ = [
    "CompatibilitySpec",
    "CoordinateOutputMode",
    "CoordinateSelectionStrategy",
    "FilterConfig",
    "PreviewConfig",
    "ResolvedAnnotationMasks",
    "SegmentationConfig",
    "SlideSpec",
    "TilingArtifacts",
    "TilingConfig",
    "extract_tiles_to_tar",
    "load_tiling_result",
    "maybe_load_existing_artifacts",
    "overlay_mask_on_slide",
    "resolve_annotation_masks",
    "save_tiling_result",
    "summarize_annotation_coverage",
    "tile_slide",
    "tile_slides",
    "validate_tiling_artifacts",
    "write_tiling_preview",
]
