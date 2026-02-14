from __future__ import annotations

import numpy as np

from hs2p.wsi import get_mask_coverage, sort_coordinates_with_tissue


def test_sort_coordinates_with_tissue_deduplicates_and_orders_deterministically():
    coordinates = [(10, 0), (2, 2), (10, 0), (1, 9)]
    tissue_percentages = [0.1, 0.2, 0.3, 0.4]
    contour_indices = [5, 6, 7, 8]

    sorted_coords, sorted_tissue, sorted_contours = sort_coordinates_with_tissue(
        coordinates,
        tissue_percentages,
        contour_indices,
    )

    assert sorted_coords == [(10, 0), (1, 9), (2, 2)]
    assert sorted_tissue == [0.1, 0.4, 0.2]
    assert sorted_contours == [5, 8, 6]


def test_get_mask_coverage_returns_exact_fraction_for_label_value():
    mask = np.array(
        [
            [0, 1, 1, 2],
            [2, 2, 1, 1],
        ],
        dtype=np.uint8,
    )

    assert get_mask_coverage(mask, 1) == 0.5
    assert get_mask_coverage(mask, 2) == 0.375
    assert get_mask_coverage(mask, 3) == 0.0
