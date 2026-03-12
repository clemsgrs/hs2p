from __future__ import annotations

from pathlib import Path

import numpy as np

from hs2p.api import FilterConfig, SegmentationConfig, TilingConfig
import hs2p.wsi as wsi_api
import hs2p.wsi.wsi as wsimod
from hs2p.wsi import SamplingParameters
from tests.helpers.fake_wsi_backend import FakePyramidWSI, PyramidSpec


def _segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(
        downsample=2,
        sthresh=8,
        sthresh_up=255,
        mthresh=3,
        close=0,
        use_otsu=False,
        use_hsv=False,
    )


def _filter_config() -> FilterConfig:
    return FilterConfig(
        ref_tile_size=4,
        a_t=0,
        a_h=0,
        max_n_holes=8,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def _tiling_config(
    *,
    spacing: float = 1.0,
    tolerance: float = 0.01,
    tile_size: int = 8,
    tissue_threshold: float = 0.0,
) -> TilingConfig:
    return TilingConfig(
        target_spacing_um=spacing,
        target_tile_size_px=tile_size,
        tolerance=tolerance,
        overlap=0.0,
        tissue_threshold=tissue_threshold,
        drop_holes=False,
        use_padding=False,
        backend="asap",
    )


def _sampling_parameters(*, tissue_threshold: float) -> SamplingParameters:
    return SamplingParameters(
        pixel_mapping={"background": 0, "tissue": 1},
        color_mapping={"background": None, "tissue": None},
        tissue_percentage={"background": None, "tissue": tissue_threshold},
    )


def test_extract_coordinates_returns_exact_coordinates_for_rectangular_tissue(fake_backend):
    mask_l0 = np.zeros((16, 16, 1), dtype=np.uint8)
    mask_l0[4:12, 4:12, 0] = 1
    fake_backend(mask_l0)

    result = wsi_api.extract_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        segment_params=_segmentation_config(),
        tiling_params=_tiling_config(tissue_threshold=0.0),
        filter_params=_filter_config(),
        sampling_params=_sampling_parameters(tissue_threshold=0.0),
        disable_tqdm=True,
        num_workers=1,
    )

    assert result.coordinates == [(16, 16), (16, 8), (8, 16), (8, 8)]
    assert result.contour_indices == [0, 0, 0, 0]
    assert result.read_level == 0
    assert result.resize_factor == 1.0
    assert result.tile_size_lv0 == 8


def test_extract_coordinates_respects_50_vs_51_percent_tissue_threshold(fake_backend):
    mask_l0 = np.zeros((16, 16, 1), dtype=np.uint8)
    mask_l0[4:12, 4:10, 0] = 1  # produces two 100%-tissue tiles and two 50%-tissue tiles
    fake_backend(mask_l0)

    sampling_at_50 = _sampling_parameters(tissue_threshold=0.50)
    sampling_above_50 = _sampling_parameters(tissue_threshold=0.51)

    result_50 = wsi_api.extract_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        segment_params=_segmentation_config(),
        tiling_params=_tiling_config(tissue_threshold=0.50),
        filter_params=_filter_config(),
        sampling_params=sampling_at_50,
        disable_tqdm=True,
        num_workers=1,
    )
    result_51 = wsi_api.extract_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        segment_params=_segmentation_config(),
        tiling_params=_tiling_config(tissue_threshold=0.51),
        filter_params=_filter_config(),
        sampling_params=sampling_above_50,
        disable_tqdm=True,
        num_workers=1,
    )

    assert result_50.coordinates == [(16, 16), (16, 8), (8, 16), (8, 8)]
    assert result_51.coordinates == [(8, 16), (8, 8)]
    assert len(result_51.coordinates) < len(result_50.coordinates)


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
        result = wsi_api.extract_coordinates(
            wsi_path=Path("synthetic-slide.tif"),
            mask_path=Path("synthetic-mask.tif"),
            backend="asap",
            segment_params=_segmentation_config(),
            tiling_params=_tiling_config(
                spacing=spacing,
                tolerance=0.01,
                tile_size=8,
                tissue_threshold=0.0,
            ),
            filter_params=_filter_config(),
            sampling_params=_sampling_parameters(tissue_threshold=0.0),
            disable_tqdm=True,
            num_workers=1,
        )

        assert result.coordinates == exp["coordinates"]
        assert len(result.coordinates) == len(exp["coordinates"])
        assert result.read_level == exp["tile_level"]
        assert result.resize_factor == exp["resize_factor"]
        assert result.tile_size_lv0 == exp["tile_size_lv0"]


def test_extract_coordinates_segments_maskless_slides_without_annotation_pct_crash(monkeypatch):
    slide_l0 = np.full((32, 32, 3), 255, dtype=np.uint8)
    slide_l1 = slide_l0[::2, ::2, :]
    tissue_mask = np.zeros((32, 32), dtype=np.uint8)
    tissue_mask[8:24, 8:24] = 255

    def _fake_wholeslide(path: Path, backend: str = "asap"):
        del path, backend
        return FakePyramidWSI(PyramidSpec(spacings=[1.0, 2.0], levels=[slide_l0, slide_l1]))

    def _fake_segment_tissue(self, segment_params):
        del segment_params
        self.annotation_mask = {"tissue": tissue_mask}
        return 0

    monkeypatch.setattr(wsimod.wsd, "WholeSlideImage", _fake_wholeslide)
    monkeypatch.setattr(wsimod.WholeSlideImage, "segment_tissue", _fake_segment_tissue)

    result = wsi_api.extract_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=None,
        backend="asap",
        segment_params=SegmentationConfig(
            downsample=2,
            sthresh=8,
            sthresh_up=255,
            mthresh=3,
            close=0,
            use_otsu=False,
            use_hsv=False,
        ),
        tiling_params=_tiling_config(tissue_threshold=0.0),
        filter_params=_filter_config(),
        disable_tqdm=True,
        num_workers=1,
    )

    assert len(result.coordinates) > 0
    assert len(result.coordinates) == len(result.tissue_percentages)
