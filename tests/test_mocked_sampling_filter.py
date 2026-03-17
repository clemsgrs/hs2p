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
    )

    assert filtered["tumor"] == [(8, 8), (16, 16)]
    assert filtered["stroma"] == [(16, 8), (16, 16)]
    assert filtered_indices["tumor"] == [0, 0]
    assert filtered_indices["stroma"] == [0, 0]


def test_filter_coordinates_reuses_loaded_mask_and_avoids_per_tile_mask_reads(
    monkeypatch,
):
    constructor_calls = []

    class FakeMaskBackend:
        def __init__(self):
            self.spacings = [1.0]
            self.downsamplings = [1.0]
            self.shapes = [(4, 4)]

        def get_slide(self, spacing):
            assert spacing == 1.0
            return np.array(
                [
                    [[1], [1], [2], [2]],
                    [[1], [1], [2], [2]],
                    [[0], [0], [2], [2]],
                    [[0], [0], [2], [2]],
                ],
                dtype=np.uint8,
            )

        def get_patch(self, *args, **kwargs):
            raise AssertionError(
                "filter_coordinates should not read one mask patch per tile"
            )

    class FakeWholeSlideImage:
        def __init__(
            self,
            path,
            mask_path=None,
            backend="asap",
            spacing_at_level_0=None,
            segment=False,
            segment_params=None,
            sampling_params=None,
            pixel_mapping=None,
        ):
            del (
                backend,
                spacing_at_level_0,
                segment,
                segment_params,
                sampling_params,
                pixel_mapping,
            )
            constructor_calls.append(
                (Path(path), None if mask_path is None else Path(mask_path))
            )
            self.path = Path(path)
            self.spacings = [1.0]
            self.level_dimensions = [(4, 4)]
            self.level_downsamples = [(1.0, 1.0)]
            self.mask = FakeMaskBackend() if mask_path is not None else None

        def get_level_spacing(self, level):
            assert level == 0
            return 1.0

    monkeypatch.setattr(wsi_api, "WholeSlideImage", FakeWholeSlideImage)

    filtered, filtered_indices = wsi_api.filter_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        coordinates=[(0, 0), (2, 0), (0, 2)],
        contour_indices=[0, 1, 2],
        tile_level=0,
        segment_params=_segmentation_config(),
        tiling_params=TilingConfig(
            target_spacing_um=1.0,
            target_tile_size_px=2,
            tolerance=0.01,
            overlap=0.0,
            tissue_threshold=0.0,
            drop_holes=False,
            use_padding=False,
            backend="asap",
        ),
        sampling_params=_sampling_parameters(),
    )

    assert constructor_calls == [
        (Path("synthetic-slide.tif"), Path("synthetic-mask.tif")),
    ]
    assert filtered["tumor"] == [(0, 0)]
    assert filtered["stroma"] == [(2, 0)]
    assert filtered_indices["tumor"] == [0]
    assert filtered_indices["stroma"] == [1]


def test_filter_coordinates_vectorized_path_avoids_per_tile_crops_and_handles_borders(
    monkeypatch,
):
    class FakeMaskBackend:
        def __init__(self):
            self.spacings = [1.0]
            self.downsamplings = [1.0]
            self.shapes = [(3, 3)]

        def get_slide(self, spacing):
            assert spacing == 1.0
            return np.array(
                [
                    [[1], [1], [0]],
                    [[1], [1], [0]],
                    [[0], [0], [0]],
                ],
                dtype=np.uint8,
            )

    class FakeWholeSlideImage:
        def __init__(
            self,
            path,
            mask_path=None,
            backend="asap",
            spacing_at_level_0=None,
            segment=False,
            segment_params=None,
            sampling_params=None,
            pixel_mapping=None,
        ):
            del (
                path,
                backend,
                spacing_at_level_0,
                segment,
                segment_params,
                sampling_params,
                pixel_mapping,
            )
            self.spacings = [1.0]
            self.level_dimensions = [(3, 3)]
            self.level_downsamples = [(1.0, 1.0)]
            self.mask = FakeMaskBackend() if mask_path is not None else None

        def get_level_spacing(self, level):
            assert level == 0
            return 1.0

    monkeypatch.setattr(wsi_api, "WholeSlideImage", FakeWholeSlideImage)
    monkeypatch.setattr(
        wsi_api,
        "_extract_padded_crop",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("vectorized mask scoring should not crop one tile at a time")
        ),
    )

    filtered, filtered_indices = wsi_api.filter_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        coordinates=[(0, 0), (1, 0), (2, 2)],
        contour_indices=[0, 1, 2],
        tile_level=0,
        segment_params=_segmentation_config(),
        tiling_params=TilingConfig(
            target_spacing_um=1.0,
            target_tile_size_px=2,
            tolerance=0.01,
            overlap=0.0,
            tissue_threshold=0.0,
            drop_holes=False,
            use_padding=False,
            backend="asap",
        ),
        sampling_params=SamplingParameters(
            pixel_mapping={"background": 0, "tumor": 1},
            color_mapping={"background": None, "tumor": None},
            tissue_percentage={"background": None, "tumor": 0.5},
        ),
    )

    assert filtered["tumor"] == [(0, 0), (1, 0)]
    assert filtered_indices["tumor"] == [0, 1]


def test_load_segmentation_preserves_discrete_labels_with_nearest_neighbor(
    fake_backend,
):
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
