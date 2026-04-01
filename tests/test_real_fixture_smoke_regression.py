from pathlib import Path

import numpy as np
import pytest

from hs2p.api import FilterConfig, SegmentationConfig, TilingConfig
import hs2p.wsi as wsi_api
from hs2p.wsi import SamplingSpec

pytestmark = pytest.mark.integration


def _choose_backend(wsi_path: Path) -> str:
    wsd = pytest.importorskip("wholeslidedata")
    for backend in ("asap", "openslide"):
        try:
            wsd.WholeSlideImage(wsi_path, backend=backend)
            return backend
        except Exception:
            continue
    pytest.skip("No supported WholeSlideData backend is available for TIFF fixtures")


def _run_extract(wsi_path: Path, mask_path: Path, backend: str, tissue_pct: float):
    return wsi_api.extract_coordinates(
        wsi_path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        segment_params=SegmentationConfig(
            downsample=64,
            sthresh=8,
            sthresh_up=255,
            mthresh=7,
            close=4,
            use_otsu=False,
            use_hsv=True,
        ),
        tiling_params=TilingConfig(
            target_spacing_um=0.5,
            target_tile_size_px=224,
            tolerance=0.07,
            overlap=0.0,
            tissue_threshold=tissue_pct,
            backend=backend,
        ),
        filter_params=FilterConfig(
            ref_tile_size=224,
            a_t=4,
            a_h=2,
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
        sampling_spec=SamplingSpec(
            pixel_mapping={"background": 0, "tissue": 1},
            color_mapping={"background": None, "tissue": None},
            tissue_percentage={"background": None, "tissue": tissue_pct},
            active_annotations=("tissue",),
        ),
        disable_tqdm=True,
        num_workers=1,
    )


def test_real_fixture_is_deterministic_and_matches_expected_counts(real_fixture_paths):
    wsi_path, mask_path = real_fixture_paths
    backend = _choose_backend(wsi_path)

    run1 = _run_extract(wsi_path, mask_path, backend, tissue_pct=0.10)
    run2 = _run_extract(wsi_path, mask_path, backend, tissue_pct=0.10)

    assert run1.read_level == run2.read_level
    assert run1.read_spacing_um == pytest.approx(run2.read_spacing_um)
    assert run1.read_tile_size_px == run2.read_tile_size_px
    assert run1.resize_factor == pytest.approx(run2.resize_factor)
    assert run1.tile_size_lv0 == run2.tile_size_lv0
    assert run1.coordinates == run2.coordinates
    assert run1.contour_indices == run2.contour_indices
    assert len(run1.tissue_percentages) == len(run1.coordinates)
    assert len(run2.tissue_percentages) == len(run2.coordinates)
    assert all(0.0 <= value <= 1.0 for value in run1.tissue_percentages)
    assert all(0.0 <= value <= 1.0 for value in run2.tissue_percentages)
    np.testing.assert_array_equal(run1.x, run2.x)
    np.testing.assert_array_equal(run1.y, run2.y)


def test_real_fixture_outputs_sane_level0_coordinates(real_fixture_paths):
    wsi_path, mask_path = real_fixture_paths
    backend = _choose_backend(wsi_path)

    result = _run_extract(wsi_path, mask_path, backend, tissue_pct=0.10)

    assert len(result.coordinates) > 0
    assert len(result.coordinates) == len(result.contour_indices)
    assert len(result.coordinates) == len(result.tissue_percentages)
    assert len(result.coordinates) == len(set(result.coordinates))
    assert result.read_level >= 0
    assert result.read_spacing_um > 0
    assert result.read_tile_size_px > 0
    assert result.resize_factor > 0
    assert result.tile_size_lv0 > 0
    np.testing.assert_array_equal(
        result.x,
        np.array([x for x, _ in result.coordinates], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result.y,
        np.array([y for _, y in result.coordinates], dtype=np.int64),
    )
