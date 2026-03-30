from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hs2p.api import FilterConfig, SegmentationConfig, TilingConfig
import hs2p.wsi as wsi_api
from hs2p.wsi import api as wsi_api_mod
import hs2p.wsi.wsi as wsimod
from hs2p.wsi import ResolvedSamplingSpec
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
        use_padding=False,
        backend="asap",
    )


def _sampling_spec(*, tissue_threshold: float) -> ResolvedSamplingSpec:
    return ResolvedSamplingSpec(
        pixel_mapping={"background": 0, "tissue": 1},
        color_mapping={"background": None, "tissue": None},
        tissue_percentage={"background": None, "tissue": tissue_threshold},
        active_annotations=("tissue",),
    )


def test_extract_coordinates_returns_exact_coordinates_for_rectangular_tissue(
    fake_backend,
):
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
        sampling_spec=_sampling_spec(tissue_threshold=0.0),
        disable_tqdm=True,
        num_workers=1,
    )

    assert result.coordinates == [(8, 8), (8, 16), (16, 8), (16, 16)]
    assert result.contour_indices == [0, 0, 0, 0]
    assert result.read_level == 0
    assert result.resize_factor == 1.0
    assert result.tile_size_lv0 == 8


def test_extract_coordinates_respects_50_vs_51_percent_tissue_threshold(fake_backend):
    mask_l0 = np.zeros((16, 16, 1), dtype=np.uint8)
    mask_l0[4:12, 4:10, 0] = (
        1  # produces two 100%-tissue tiles and two 50%-tissue tiles
    )
    fake_backend(mask_l0)

    sampling_at_50 = _sampling_spec(tissue_threshold=0.50)
    sampling_above_50 = _sampling_spec(tissue_threshold=0.51)

    result_50 = wsi_api.extract_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        segment_params=_segmentation_config(),
        tiling_params=_tiling_config(tissue_threshold=0.50),
        filter_params=_filter_config(),
        sampling_spec=sampling_at_50,
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
        sampling_spec=sampling_above_50,
        disable_tqdm=True,
        num_workers=1,
    )

    assert result_50.coordinates == [(8, 8), (8, 16), (16, 8), (16, 16)]
    assert result_51.coordinates == [(8, 8), (8, 16)]
    assert len(result_51.coordinates) < len(result_50.coordinates)


def test_extract_coordinates_match_expected_coordinates_across_spacings(fake_backend):
    mask_l0 = np.zeros((16, 16, 1), dtype=np.uint8)
    mask_l0[4:12, 4:12, 0] = 1
    fake_backend(mask_l0)

    expected = {
        1.0: {
            "coordinates": [(8, 8), (8, 16), (16, 8), (16, 16)],
            "tile_level": 0,
            "resize_factor": 1.0,
            "tile_size_lv0": 8,
        },
        1.5: {
            "coordinates": [(8, 8), (8, 20), (20, 8), (20, 20)],
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
            sampling_spec=_sampling_spec(tissue_threshold=0.0),
            disable_tqdm=True,
            num_workers=1,
        )

        assert result.coordinates == exp["coordinates"]
        assert len(result.coordinates) == len(exp["coordinates"])
        assert result.read_level == exp["tile_level"]
        assert result.resize_factor == exp["resize_factor"]
        assert result.tile_size_lv0 == exp["tile_size_lv0"]


def test_extract_coordinate_result_preserves_stride_when_contours_have_offset_origins():
    class FakeWSI:
        level_downsamples = [(1.0, 1.0)]

        def get_tile_coordinates(
            self,
            tiling_params,
            filter_params,
            annotation=None,
            disable_tqdm=False,
            num_workers=1,
        ):
            del tiling_params, filter_params, annotation, disable_tqdm, num_workers
            return (
                [
                    (0, 0),
                    (0, 224),
                    (224, 0),
                    (224, 224),
                    (225, 0),
                    (225, 224),
                ],
                [1.0] * 6,
                [0, 0, 0, 0, 1, 1],
                0,
                1.0,
                224,
            )

        def get_level_spacing(self, level):
            assert level == 0
            return 0.5

    result = wsi_api_mod._extract_coordinate_result_from_wsi(
        wsi=FakeWSI(),
        tiling_params=SimpleNamespace(
            target_tile_size_px=224,
            overlap=0.0,
        ),
        filter_params=SimpleNamespace(),
        disable_tqdm=True,
        num_workers=1,
    )

    assert result.read_step_px == 224
    assert result.step_px_lv0 == 224


def test_extract_coordinate_result_uses_actual_overlap_stride_in_level0_pixels():
    class FakeWSI:
        level_downsamples = [(1.0, 1.0), (2.0, 2.0)]

        def get_tile_coordinates(
            self,
            tiling_params,
            filter_params,
            annotation=None,
            disable_tqdm=False,
            num_workers=1,
        ):
            del tiling_params, filter_params, annotation, disable_tqdm, num_workers
            return (
                [
                    (100, 100),
                    (100, 502),
                    (502, 100),
                    (502, 502),
                ],
                [1.0] * 4,
                [0, 0, 0, 0],
                1,
                1.0,
                448,
            )

        def get_level_spacing(self, level):
            return [0.5, 1.0][level]

    result = wsi_api_mod._extract_coordinate_result_from_wsi(
        wsi=FakeWSI(),
        tiling_params=SimpleNamespace(
            target_tile_size_px=224,
            overlap=0.1,
        ),
        filter_params=SimpleNamespace(),
        disable_tqdm=True,
        num_workers=1,
    )

    assert result.read_step_px == 201
    assert result.step_px_lv0 == 402


def test_extract_coordinates_segments_maskless_slides_without_annotation_pct_crash(
    monkeypatch,
):
    slide_l0 = np.full((32, 32, 3), 255, dtype=np.uint8)
    slide_l1 = slide_l0[::2, ::2, :]
    tissue_mask = np.zeros((32, 32), dtype=np.uint8)
    tissue_mask[8:24, 8:24] = 255

    def _fake_wholeslide(path: Path, backend: str = "asap"):
        del path, backend
        return FakePyramidWSI(
            PyramidSpec(spacings=[1.0, 2.0], levels=[slide_l0, slide_l1])
        )

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


def test_extract_coordinates_returns_zero_tile_result_for_tissue_free_maskless_slide(
    monkeypatch,
):
    slide_l0 = np.full((32, 32, 3), 255, dtype=np.uint8)
    slide_l1 = slide_l0[::2, ::2, :]

    def _fake_wholeslide(path: Path, backend: str = "asap"):
        del path, backend
        return FakePyramidWSI(
            PyramidSpec(spacings=[1.0, 2.0], levels=[slide_l0, slide_l1])
        )

    monkeypatch.setattr(wsimod.wsd, "WholeSlideImage", _fake_wholeslide)

    result = wsi_api.extract_coordinates(
        wsi_path=Path("empty-slide.tif"),
        mask_path=None,
        backend="asap",
        segment_params=_segmentation_config(),
        tiling_params=_tiling_config(tissue_threshold=0.01),
        filter_params=_filter_config(),
        disable_tqdm=True,
        num_workers=1,
    )

    assert result.coordinates == []
    assert result.contour_indices == []
    assert result.tissue_percentages == []
    assert result.read_level == 0
    assert result.read_spacing_um == 1.0
    assert result.read_tile_size_px == 8
    assert result.resize_factor == 1.0
    assert result.tile_size_lv0 == 8


def test_sample_coordinates_returns_zero_tile_result_for_tissue_free_annotation(
    fake_backend,
):
    mask_l0 = np.zeros((16, 16, 1), dtype=np.uint8)
    fake_backend(mask_l0)

    result = wsi_api.sample_coordinates(
        wsi_path=Path("synthetic-slide.tif"),
        mask_path=Path("synthetic-mask.tif"),
        backend="asap",
        segment_params=_segmentation_config(),
        tiling_params=_tiling_config(tissue_threshold=0.01),
        filter_params=_filter_config(),
        sampling_spec=_sampling_spec(tissue_threshold=0.01),
        annotation="tissue",
        disable_tqdm=True,
        num_workers=1,
    )

    assert result.coordinates == []
    assert result.contour_indices == []
    assert result.tissue_percentages == []
    assert result.read_level == 0
    assert result.read_spacing_um == 1.0
    assert result.read_tile_size_px == 8
    assert result.resize_factor == 1.0
    assert result.tile_size_lv0 == 8
