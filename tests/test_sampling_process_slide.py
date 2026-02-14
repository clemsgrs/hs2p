from pathlib import Path
from types import SimpleNamespace

import pytest


sampling_mod = pytest.importorskip("hs2p.sampling")


def test_independent_sampling_no_visualization_does_not_crash(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        visualize=False,
        output_dir=str(tmp_path),
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(spacing=0.5, tile_size=256),
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
        return ([(0, 0)], [0], 0, 1.0, 256)

    monkeypatch.setattr(sampling_mod, "sample_coordinates", _fake_sample_coordinates)
    monkeypatch.setattr(sampling_mod, "save_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(sampling_mod, "visualize_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(sampling_mod, "overlay_mask_on_slide", lambda **kwargs: None)

    _, status_info = sampling_mod.process_slide(
        wsi_path=Path("fake-wsi.tif"),
        mask_path=Path("fake-mask.tif"),
        cfg=cfg,
        mask_visualize_dir=None,
        sampling_visualize_dir=None,
        sampling_params=sampling_params,
    )

    assert status_info["status"] == "success"
