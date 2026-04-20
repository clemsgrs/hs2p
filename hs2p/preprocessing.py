"""Reusable low-level preprocessing primitives shared with downstream projects."""

from hs2p.tiling.contours import _normalize_level_downsamples, detect_contours
from hs2p.tiling.coverage import compute_tile_coverage
from hs2p.tiling.generate import (
    _build_contour_tissue_mask,
    _tiles_for_contour,
    canonicalize_tiling_result,
    generate_tiles,
    resolve_base_spacing_um,
)
from hs2p.tiling.io import (
    COORDINATE_SPACE,
    TILE_ORDER,
    _ARTIFACT_KEYS,
    _FILTERING_KEYS,
    _PROVENANCE_KEYS,
    _SEGMENTATION_KEYS,
    _SLIDE_KEYS,
    _TILING_KEYS,
    _TOP_LEVEL_META_KEYS,
    _build_tiling_metadata,
    _load_tiling_result,
    _load_tiling_result_from_paths,
    _save_tiling_result,
    _validate_metadata_schema,
    _validate_tile_index,
    normalize_artifact_path,
    validate_tiling_result_provenance,
)
from hs2p.tiling.mask import (
    load_precomputed_tissue_mask,
    prepare_sam2_thumbnail,
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
    "detect_contours",
    "generate_tiles",
    "load_precomputed_tissue_mask",
    "prepare_sam2_thumbnail",
    "resolve_tissue_mask",
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
