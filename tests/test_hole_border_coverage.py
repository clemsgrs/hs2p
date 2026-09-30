"""Hole rasterization keeps the foreground ring around each hole (F6)."""

import numpy as np
import pytest

from hs2p.tiling.contours import detect_contours
from hs2p.tiling.generate import _build_contour_tissue_mask, generate_tiles


def _contours(mask: np.ndarray):
    size = mask.shape[0]
    return detect_contours(
        mask,
        slide_dimensions=(size, size),
        a_t=0,
        base_spacing_um=0.5,
        level_downsamples=[1.0],
    )


def test_one_pixel_hole_keeps_its_four_foreground_neighbours():
    # the review's mask: 15 of 16 pixels are foreground
    mask = np.ones((4, 4), dtype=np.uint8)
    mask[1, 1] = 0
    contours = _contours(mask)
    assert len(contours.contours) == 1 and len(contours.holes[0]) == 1

    contour_mask = _build_contour_tissue_mask(
        contours.contours[0], contours.holes[0], mask, (4, 4)
    )

    np.testing.assert_array_equal(contour_mask, mask)
    assert int(contour_mask.sum()) == 15


def test_single_tile_over_the_holed_mask_survives_a_high_threshold():
    mask = np.ones((4, 4), dtype=np.uint8)
    mask[1, 1] = 0

    geometry = generate_tiles(
        (4, 4),
        _contours(mask),
        requested_tile_size_px=4,
        requested_spacing_um=0.5,
        base_spacing_um=0.5,
        level_downsamples=[1.0],
        min_tissue_fraction=0.9,
    )

    assert list(zip(geometry.x.tolist(), geometry.y.tolist())) == [(0, 0)]
    assert geometry.tissue_fractions.tolist() == pytest.approx([15 / 16])


def test_island_inside_a_hole_stays_out_of_the_enclosing_contour():
    # 12x12 tissue with a 6x6 hole holding a 2x2 island that touches nothing
    mask = np.ones((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 0
    mask[5:7, 5:7] = 1
    contours = _contours(mask)
    assert len(contours.contours) == 2

    per_contour = [
        _build_contour_tissue_mask(contour, holes, mask, (12, 12))
        for contour, holes in zip(contours.contours, contours.holes)
    ]
    sums = sorted(int(m.sum()) for m in per_contour)

    # 144 - 36 hole pixels (island included) for the ring, 4 for the island; nothing
    # counted twice and the hole border ring (5x5 minus 3x3 = 32 px) retained
    assert sums == [4, 108]
    np.testing.assert_array_equal(per_contour[0] + per_contour[1], mask)
