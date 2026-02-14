from __future__ import annotations

from pathlib import Path

import numpy as np

import hs2p.wsi as wsi_api
from params import make_filter_params, make_sampling_params, make_segment_params, make_tiling_params


def test_extract_coordinates_returns_exact_coordinates_for_rectangular_tissue(fake_backend):
    mask_l0 = np.zeros((16, 16, 1), dtype=np.uint8)
    mask_l0[4:12, 4:12, 0] = 1
    fake_backend(mask_l0)

    coordinates, contour_indices, tile_level, resize_factor, tile_size_lv0 = wsi_api.extract_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        segment_params=make_segment_params(),
        tiling_params=make_tiling_params(min_tissue_percentage=0.0),
        filter_params=make_filter_params(),
        sampling_params=make_sampling_params(
            pixel_mapping={"background": 0, "tissue": 1},
            tissue_percentage={"background": None, "tissue": 0.0},
        ),
        disable_tqdm=True,
        num_workers=1,
    )

    assert coordinates == [(16, 16), (16, 8), (8, 16), (8, 8)]
    assert contour_indices == [0, 0, 0, 0]
    assert tile_level == 0
    assert resize_factor == 1.0
    assert tile_size_lv0 == 8


def test_extract_coordinates_respects_50_vs_51_percent_tissue_threshold(fake_backend):
    mask_l0 = np.zeros((16, 16, 1), dtype=np.uint8)
    mask_l0[4:12, 4:10, 0] = 1  # produces two 100%-tissue tiles and two 50%-tissue tiles
    fake_backend(mask_l0)

    sampling_at_50 = make_sampling_params(
        pixel_mapping={"background": 0, "tissue": 1},
        tissue_percentage={"background": None, "tissue": 0.50},
    )
    sampling_above_50 = make_sampling_params(
        pixel_mapping={"background": 0, "tissue": 1},
        tissue_percentage={"background": None, "tissue": 0.51},
    )

    coordinates_50, *_ = wsi_api.extract_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        segment_params=make_segment_params(),
        tiling_params=make_tiling_params(min_tissue_percentage=0.50),
        filter_params=make_filter_params(),
        sampling_params=sampling_at_50,
        disable_tqdm=True,
        num_workers=1,
    )
    coordinates_51, *_ = wsi_api.extract_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        segment_params=make_segment_params(),
        tiling_params=make_tiling_params(min_tissue_percentage=0.51),
        filter_params=make_filter_params(),
        sampling_params=sampling_above_50,
        disable_tqdm=True,
        num_workers=1,
    )

    assert coordinates_50 == [(16, 16), (16, 8), (8, 16), (8, 8)]
    assert coordinates_51 == [(8, 16), (8, 8)]
    assert len(coordinates_51) < len(coordinates_50)


def test_extract_coordinates_match_expected_coordinates_across_spacings(fake_backend):
    mask_l0 = np.zeros((16, 16, 1), dtype=np.uint8)
    mask_l0[4:12, 4:12, 0] = 1
    fake_backend(mask_l0)

    expected = {
        1.0: {
            "coordinates": [(16, 16), (16, 8), (8, 16), (8, 8)],
            "tile_level": 0,
            "resize_factor": 1.0,
            "tile_size_lv0": 8,
        },
        1.5: {
            "coordinates": [(20, 20), (20, 8), (8, 20), (8, 8)],
            "tile_level": 0,
            "resize_factor": 1.5,
            "tile_size_lv0": 12,
        },
        2.0: {
            "coordinates": [(8, 8)],
            "tile_level": 1,
            "resize_factor": 1.0,
            "tile_size_lv0": 16,
        },
    }

    for spacing, exp in expected.items():
        coordinates, _, tile_level, resize_factor, tile_size_lv0 = wsi_api.extract_coordinates(
            wsi_path=Path("synthetic-slide.tif"),
            mask_path=Path("synthetic-mask.tif"),
            backend="asap",
            segment_params=make_segment_params(),
            tiling_params=make_tiling_params(
                spacing=spacing,
                tolerance=0.01,
                tile_size=8,
                overlap=0.0,
                min_tissue_percentage=0.0,
                drop_holes=False,
                use_padding=False,
            ),
            filter_params=make_filter_params(),
            sampling_params=make_sampling_params(
                pixel_mapping={"background": 0, "tissue": 1},
                tissue_percentage={"background": None, "tissue": 0.0},
            ),
            disable_tqdm=True,
            num_workers=1,
        )

        assert coordinates == exp["coordinates"]
        assert len(coordinates) == len(exp["coordinates"])
        assert tile_level == exp["tile_level"]
        assert resize_factor == exp["resize_factor"]
        assert tile_size_lv0 == exp["tile_size_lv0"]
