from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hs2p.wsi as wsi_api
from params import make_filter_params, make_sampling_params, make_segment_params, make_tiling_params


def _require_asap_backend(wsi_path: Path) -> str:
    wsd = pytest.importorskip("wholeslidedata")
    try:
        wsd.WholeSlideImage(wsi_path, backend="asap")
        return "asap"
    except Exception:
        pytest.skip("ASAP backend is unavailable; golden coordinate regression requires ASAP")


def test_generated_coordinates_match_legacy_golden(real_fixture_paths, tmp_path: Path):
    wsi_path, mask_path = real_fixture_paths
    gt_path = wsi_path.parent.parent / "gt" / "test-wsi.npy"
    assert gt_path.is_file(), f"Missing golden coordinates: {gt_path}"

    backend = _require_asap_backend(wsi_path)
    tissue_pct = 0.1

    coordinates, contour_indices, tile_level, resize_factor, tile_size_lv0 = wsi_api.extract_coordinates(
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

    generated_path = tmp_path / "generated-test-wsi.npy"
    wsi_api.save_coordinates(
        coordinates=coordinates,
        contour_indices=contour_indices,
        target_spacing=0.5,
        tile_level=tile_level,
        target_tile_size=224,
        resize_factor=resize_factor,
        tile_size_lv0=tile_size_lv0,
        save_path=generated_path,
    )

    legacy = np.load(gt_path, allow_pickle=False)
    generated = np.load(generated_path, allow_pickle=False)
    np.testing.assert_array_equal(generated, legacy)
