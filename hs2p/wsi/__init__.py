from pathlib import Path
from typing import Any

import tqdm

from hs2p.progress import emit_progress_log
from .api import (
    CoordinateExtractionResult,
    CoordinateOutputMode,
    CoordinateSelectionStrategy,
    UnifiedCoordinateRequest,
    UnifiedCoordinateResponse,
    execute_coordinate_request,
    extract_coordinates,
    filter_coordinates,
    get_mask_coverage,
    sample_coordinates,
    sort_coordinates_with_tissue,
)
from .masks import (
    compose_overlay_mask_from_annotations,
    extract_padded_crop,
    mask_level_downsamples,
    normalize_tissue_mask,
    pad_array_to_shape,
    read_aligned_mask,
)
from .preview import (
    build_overlay_alpha,
    build_palette,
    draw_grid,
    draw_grid_from_coordinates,
    overlay_mask_on_tile,
    pad_to_patch_size,
)
from .visualization import (
    DEFAULT_TISSUE_COLOR_MAPPING,
    DEFAULT_TISSUE_PIXEL_MAPPING,
    overlay_mask_on_slide,
    save_overlay_preview,
    write_coordinate_preview,
)
from .wsi import (
    ResolvedSamplingSpec,
    WholeSlideImage,
)
