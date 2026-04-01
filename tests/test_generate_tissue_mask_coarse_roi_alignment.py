
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.fake_wsi_backend import FakePyramidWSI, PyramidSpec

pytestmark = pytest.mark.script


def _load_script_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "generate_tissue_mask.py"
    spec = spec_from_file_location("generate_tissue_mask", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coarse_roi_shortcut_preserves_spatial_alignment(monkeypatch, tmp_path):
    script = _load_script_module()

    level0 = np.full((64, 64, 3), 255, dtype=np.uint8)
    level0[16:40, 36:60] = np.array([180, 110, 180], dtype=np.uint8)
    level1 = level0[::2, ::2]

    spec = PyramidSpec(spacings=[0.25, 0.5], levels=[level0, level1])

    def fake_wsi_factory(path, backend="asap"):
        del path, backend
        return FakePyramidWSI(spec)

    monkeypatch.setattr(script.wsd, "WholeSlideImage", fake_wsi_factory)

    mask_l0, effective_spacing, mode = script._compute_level0_mask_with_coarse_roi_shortcut(
        wsi_path=tmp_path / "slide.tif",
        target_spacing=0.5,
        tolerance=0.01,
        backend="asap",
        spacing_at_level_0=None,
        coarse_spacing=0.5,
        coarse_roi_margin_um=0.0,
        processing_tile_size=8,
        min_component_area_um2=0.0,
        min_hole_area_um2=0.0,
        gaussian_sigma_um=0.0,
        open_radius_um=0.0,
        close_radius_um=0.0,
    )

    expected = script.segment_tissue_hsv(level1, gaussian_sigma_px=0.0)

    assert mode == "coarse-roi"
    assert effective_spacing == 0.5
    assert mask_l0.shape == expected.shape
    assert np.array_equal(mask_l0, expected)
