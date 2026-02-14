from __future__ import annotations

from pathlib import Path

import pytest

import hs2p.wsi as wsi_api
from params import make_filter_params, make_sampling_params, make_segment_params, make_tiling_params


EXPECTED_TILE_COUNTS = {
    0.10: 459,
    0.50: 407,
}


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
        segment_params=make_segment_params(
            downsample=64,
            sthresh=8,
            sthresh_up=255,
            mthresh=7,
            close=4,
            use_otsu=False,
            use_hsv=True,
        ),
        tiling_params=make_tiling_params(
            spacing=0.5,
            tolerance=0.07,
            tile_size=224,
            overlap=0.0,
            min_tissue_percentage=tissue_pct,
            drop_holes=False,
            use_padding=True,
        ),
        filter_params=make_filter_params(
            ref_tile_size=224,
            a_t=4,
            a_h=2,
            max_n_holes=8,
        ),
        sampling_params=make_sampling_params(
            pixel_mapping={"background": 0, "tissue": 1},
            tissue_percentage={"background": None, "tissue": tissue_pct},
        ),
        disable_tqdm=True,
        num_workers=1,
    )


def test_real_fixture_is_deterministic_and_matches_expected_counts(real_fixture_paths):
    wsi_path, mask_path = real_fixture_paths
    backend = _choose_backend(wsi_path)

    run1 = _run_extract(wsi_path, mask_path, backend, tissue_pct=0.10)
    run2 = _run_extract(wsi_path, mask_path, backend, tissue_pct=0.10)

    coordinates1, contour_indices1, tile_level1, resize_factor1, tile_size_lv01 = run1
    coordinates2, contour_indices2, tile_level2, resize_factor2, tile_size_lv02 = run2

    assert coordinates1 == coordinates2
    assert contour_indices1 == contour_indices2
    assert tile_level1 == tile_level2
    assert resize_factor1 == resize_factor2
    assert tile_size_lv01 == tile_size_lv02

    assert len(coordinates1) == EXPECTED_TILE_COUNTS[0.10]
    assert tile_level1 >= 0
    assert tile_size_lv01 > 0
    assert resize_factor1 > 0


def test_real_fixture_stricter_threshold_never_increases_tile_count(real_fixture_paths):
    wsi_path, mask_path = real_fixture_paths
    backend = _choose_backend(wsi_path)

    loose_coordinates, _, _, _, _ = _run_extract(wsi_path, mask_path, backend, tissue_pct=0.10)
    strict_coordinates, _, _, _, _ = _run_extract(wsi_path, mask_path, backend, tissue_pct=0.50)

    assert len(loose_coordinates) == EXPECTED_TILE_COUNTS[0.10]
    assert len(strict_coordinates) == EXPECTED_TILE_COUNTS[0.50]
    assert len(strict_coordinates) <= len(loose_coordinates)
