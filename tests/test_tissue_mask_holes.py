import numpy as np
import pytest


utils_mod = pytest.importorskip("hs2p.wsi.utils")


def _rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return np.array(
        [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
        dtype=np.int32,
    )


def test_precomputed_mask_subtracts_all_holes():
    contour = _rect(2, 2, 17, 17)
    holes = [_rect(4, 4, 6, 6), _rect(12, 12, 14, 14)]
    tissue_mask = np.full((20, 20), 255, dtype=np.uint8)

    checker = utils_mod.HasEnoughTissue(
        contour=contour,
        contour_holes=holes,
        tissue_mask=tissue_mask,
        target_tile_size=4,
        tile_spacing=1.0,
        resize_factor=1.0,
        seg_spacing=1.0,
        spacing_at_level_0=1.0,
        pct=0.01,
    )

    mask = checker.precomputed_mask
    assert mask[5, 5] == 0
    assert mask[13, 13] == 0
    assert mask[9, 9] == 1
