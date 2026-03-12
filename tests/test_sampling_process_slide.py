from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np


sampling_mod = pytest.importorskip("hs2p.sampling")


def test_independent_sampling_no_visualization_does_not_crash(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        visualize=False,
        output_dir=str(tmp_path),
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params=SimpleNamespace(),
            filter_params=SimpleNamespace(),
            visu_params=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(independant_sampling=True),
        ),
    )
    sampling_params = sampling_mod.SamplingParameters(
        pixel_mapping={"tumor": 1},
        color_mapping=None,
        tissue_percentage={"tumor": 0.1},
    )

    def _fake_sample_coordinates(**kwargs):
        return SimpleNamespace(
            coordinates=[(0, 0)],
            contour_indices=[0],
            read_level=0,
            read_spacing_um=0.5,
            read_tile_size_px=256,
            tile_size_lv0=256,
        )

    monkeypatch.setattr(sampling_mod, "sample_coordinates", _fake_sample_coordinates)
    monkeypatch.setattr(sampling_mod, "_save_sampling_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(sampling_mod, "visualize_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(sampling_mod, "overlay_mask_on_slide", lambda **kwargs: None)

    _, status_info = sampling_mod.process_slide(
        sample_id="sample-1",
        wsi_path=Path("fake-wsi.tif"),
        mask_path=Path("fake-mask.tif"),
        cfg=cfg,
        tiling_config=sampling_mod.TilingConfig(
            target_spacing_um=0.5,
            target_tile_size_px=256,
            tolerance=0.05,
            overlap=0.0,
            tissue_threshold=0.1,
            drop_holes=False,
            use_padding=True,
            backend="asap",
        ),
        segmentation_config=sampling_mod.SegmentationConfig(
            downsample=64,
            sthresh=8,
            sthresh_up=255,
            mthresh=7,
            close=4,
            use_otsu=False,
            use_hsv=True,
        ),
        filter_config=sampling_mod.FilterConfig(
            ref_tile_size=16,
            a_t=4,
            a_h=2,
            max_n_holes=8,
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
        mask_visualize_dir=None,
        sampling_visualize_dir=None,
        sampling_params=sampling_params,
    )

    assert status_info["status"] == "success"
