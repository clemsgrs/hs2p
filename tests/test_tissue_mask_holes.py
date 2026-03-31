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

    checker = utils_mod.TissueFilter(
        contour=contour,
        contour_holes=holes,
        tissue_mask=tissue_mask,
        geometry=utils_mod.ResolvedGeometry(
            target_tile_size_px=4,
            read_spacing_um=1.0,
            resize_factor=1.0,
            seg_spacing_um=1.0,
            level0_spacing_um=1.0,
        ),
        pct=0.01,
    )

    mask = checker.precomputed_mask
    assert mask[5, 5] == 0
    assert mask[13, 13] == 0
    assert mask[9, 9] == 1


def test_check_coordinates_returns_vectorized_outputs_with_expected_coverages():
    contour = _rect(0, 0, 7, 7)
    holes = []
    tissue_mask = np.zeros((8, 8), dtype=np.uint8)
    tissue_mask[:4, :4] = 255
    tissue_mask[4:, :4] = 255

    checker = utils_mod.TissueFilter(
        contour=contour,
        contour_holes=holes,
        tissue_mask=tissue_mask,
        geometry=utils_mod.ResolvedGeometry(
            target_tile_size_px=4,
            read_spacing_um=1.0,
            resize_factor=1.0,
            seg_spacing_um=1.0,
            level0_spacing_um=1.0,
        ),
        pct=0.5,
    )

    keep_flags, tissue_pcts = checker.check_coordinates(
        np.array(
            [
                [0, 0],
                [4, 0],
                [0, 4],
                [4, 4],
            ],
            dtype=np.int64,
        )
    )

    assert isinstance(keep_flags, np.ndarray)
    assert isinstance(tissue_pcts, np.ndarray)
    np.testing.assert_array_equal(keep_flags, np.array([1, 0, 1, 0], dtype=np.uint8))
    np.testing.assert_allclose(
        tissue_pcts, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    )
