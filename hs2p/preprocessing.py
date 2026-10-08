"""Reusable low-level preprocessing primitives shared with downstream projects."""

from hs2p.tiling.contours import detect_contours
from hs2p.tiling.coverage import compute_tile_coverage, summarize_annotation_coverage
from hs2p.tiling.generate import (
    canonicalize_tiling_result,
    generate_tiles,
    resolve_base_spacing_um,
)
from hs2p.tiling.io import (
    COORDINATE_SPACE,
    TILE_ORDER,
    normalize_artifact_path,
    validate_tiling_result_provenance,
)
from hs2p.tiling.mask import (
    prepare_sam2_thumbnail,
    resolve_annotation_masks,
    resolve_tissue_mask,
)
from hs2p.tiling.result import (
    ContourResult,
    ResolvedAnnotationMasks,
    ResolvedTissueMask,
    Sam2Thumbnail,
    TileGeometry,
    TilingResult,
)
from hs2p.tiling.single import (
    build_per_annotation_tiling_results,
    build_tiling_result_from_mask,
    preprocess_slide,
)
from hs2p.wsi.reader import open_slide, select_level, select_level_for_downsample


__all__ = [
    "ContourResult",
    "TileGeometry",
    "TilingResult",
    "ResolvedTissueMask",
    "ResolvedAnnotationMasks",
    "Sam2Thumbnail",
    "canonicalize_tiling_result",
    "compute_tile_coverage",
    "detect_contours",
    "generate_tiles",
    "prepare_sam2_thumbnail",
    "resolve_annotation_masks",
    "resolve_tissue_mask",
    "summarize_annotation_coverage",
    "build_tiling_result_from_mask",
    "build_per_annotation_tiling_results",
    "normalize_artifact_path",
    "open_slide",
    "preprocess_slide",
    "resolve_base_spacing_um",
    "select_level",
    "select_level_for_downsample",
    "validate_tiling_result_provenance",
]
