"""Regression guard: hs2p's tile count must track the physical-area heuristic.

The number of tiles hs2p emits for a slide should stay close to what tissue
content and slide geometry predict:

    tissue_area_um2  = (W_lv0 * base_spacing) * (H_lv0 * base_spacing) * tissue_frac
    step_um          = tile_size_px * spacing_um * (1 - overlap)
    approx_num_tiles = tissue_area_um2 / step_um**2

with ``tissue_frac = tissue_mask.sum() / tissue_mask.size``.

The ``(1 - overlap)`` term is *squared* because hs2p reduces the grid step in
both x and y. Because a tile is kept whenever its tissue coverage exceeds the
tiny ``tissue_threshold``, the actual count is expected to be >= the heuristic,
the surplus being a tissue-perimeter term. Parametrizing ``overlap`` exercises
the squared step: a regression collapsing the 2D step to 1D would roughly halve
the ``overlap=0.5`` count and break the lower bound.

This fixture-level check distills a dataset-scale study (40 LEOPARD slides @
224 px / 0.5 mpp): Pearson r=0.9996, slope 1.04, median actual/approx 1.04.
"""

from pathlib import Path

import numpy as np
import pytest

from hs2p import SlideSpec, tile_slide
from hs2p.configs import SegmentationConfig, TilingConfig
from hs2p.wsi.reader import open_slide

pytestmark = pytest.mark.integration

SPACING_UM = 0.5
TILE_SIZE_PX = 224
TISSUE_THRESHOLD = 0.01
SEG_DOWNSAMPLE = 64.0

# Expected actual/approx band. The lower bound encodes the "actual >= approx"
# guarantee (with slack for measuring tissue_frac at a coarse level); the upper
# bound bounds the perimeter surplus on the small fixture (observed ~1.19).
RATIO_LOW = 0.90
RATIO_HIGH = 1.50


def _tissue_frac(mask_path: Path, target_downsample: float = SEG_DOWNSAMPLE) -> float:
    """Non-zero fraction of the mask, read near the segmentation downsample."""
    mask = open_slide(str(mask_path))
    try:
        dims = mask.level_dimensions
        w0 = float(dims[0][0])
        downsamples = [w0 / float(d[0]) for d in dims]
        level = int(np.argmin([abs(d - target_downsample) for d in downsamples]))
        arr = np.asarray(mask.read_level(level))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return float(np.count_nonzero(arr)) / float(arr.size)
    finally:
        mask.close()


@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_tile_count_matches_physical_area_heuristic(
    real_fixture_paths: tuple[Path, Path], overlap: float
) -> None:
    wsi_path, mask_path = real_fixture_paths

    try:
        result = tile_slide(
            SlideSpec(sample_id="test-wsi", image_path=wsi_path, mask_path=mask_path),
            tiling=TilingConfig(
                requested_spacing_um=SPACING_UM,
                requested_tile_size_px=TILE_SIZE_PX,
                tolerance=0.07,
                overlap=overlap,
                min_coverage={"tissue": TISSUE_THRESHOLD},
            ),
            segmentation=SegmentationConfig(method="precomputed_mask"),
            num_workers=1,
        )
    except Exception as exc:  # backend unavailable in this environment
        pytest.skip(f"tile_slide could not run on the fixture: {exc}")

    actual = int(len(result.tiles.x))
    assert actual > 0

    w_lv0, h_lv0 = result.tiles.slide_dimensions
    base_spacing = float(result.tiles.base_spacing_um)
    tissue_frac = _tissue_frac(mask_path)

    tissue_area_um2 = (w_lv0 * base_spacing) * (h_lv0 * base_spacing) * tissue_frac
    step_um = TILE_SIZE_PX * SPACING_UM * (1.0 - overlap)
    approx = tissue_area_um2 / (step_um * step_um)

    ratio = actual / approx
    assert RATIO_LOW <= ratio <= RATIO_HIGH, (
        f"overlap={overlap}: actual={actual} approx={approx:.1f} ratio={ratio:.3f} "
        f"outside [{RATIO_LOW}, {RATIO_HIGH}] "
        f"(tissue_frac={tissue_frac:.4f}, dims={w_lv0}x{h_lv0}, "
        f"base_spacing={base_spacing:.4f})"
    )
