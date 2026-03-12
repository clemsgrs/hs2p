from __future__ import annotations

from pathlib import Path

import numpy as np

from hs2p.api import SegmentationConfig, TilingConfig
import hs2p.wsi as wsi_api
from hs2p.wsi import SamplingParameters
from hs2p.wsi.wsi import WholeSlideImage


def _segmentation_config(*, downsample: int = 2) -> SegmentationConfig:
    return SegmentationConfig(
        downsample=downsample,
        sthresh=8,
        sthresh_up=255,
        mthresh=3,
        close=0,
        use_otsu=False,
        use_hsv=False,
    )


def _tiling_config() -> TilingConfig:
    return TilingConfig(
        target_spacing_um=1.0,
        target_tile_size_px=8,
        tolerance=0.01,
        overlap=0.0,
        tissue_threshold=0.0,
        drop_holes=False,
        use_padding=False,
        backend="asap",
    )


def _sampling_parameters() -> SamplingParameters:
    return SamplingParameters(
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2},
        color_mapping={"background": None, "tumor": None, "stroma": None},
        tissue_percentage={"background": None, "tumor": 0.5, "stroma": 0.5},
    )


def test_filter_coordinates_returns_expected_per_class_subsets(fake_backend):
    mask_l0 = np.zeros((16, 16, 1), dtype=np.uint8)
    mask_l0[4:8, 4:8, 0] = 1
    mask_l0[4:8, 8:12, 0] = 2
    mask_l0[8:12, 8:10, 0] = 1
    mask_l0[8:12, 10:12, 0] = 2
    fake_backend(mask_l0)

    coordinates = [(8, 8), (16, 8), (8, 16), (16, 16)]
    contour_indices = [0, 0, 0, 0]

    filtered, filtered_indices = wsi_api.filter_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        coordinates=coordinates,
        contour_indices=contour_indices,
        tile_level=0,
        segment_params=_segmentation_config(),
        tiling_params=_tiling_config(),
        sampling_params=_sampling_parameters(),
        disable_tqdm=True,
    )

    assert filtered["tumor"] == [(8, 8), (16, 16)]
    assert filtered["stroma"] == [(16, 8), (16, 16)]
    assert filtered_indices["tumor"] == [0, 0]
    assert filtered_indices["stroma"] == [0, 0]


def test_load_segmentation_preserves_discrete_labels_with_nearest_neighbor(fake_backend):
    mask_l0 = np.zeros((16, 16, 1), dtype=np.uint8)
    mask_l0[4:12, 4:8, 0] = 1
    mask_l0[4:12, 8:12, 0] = 2
    fake_backend(mask_l0)

    sampling_params = SamplingParameters(
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2},
        color_mapping={"background": None, "tumor": None, "stroma": None},
        tissue_percentage={"background": None, "tumor": 0.0, "stroma": 0.0},
    )

    wsi = WholeSlideImage(
        path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        segment=True,
        segment_params=_segmentation_config(downsample=1),
        sampling_params=sampling_params,
    )

    assert set(np.unique(wsi.annotation_mask["tumor"]).tolist()) <= {0, 255}
    assert set(np.unique(wsi.annotation_mask["stroma"]).tolist()) <= {0, 255}
