import importlib
import sys
import types

import pytest


def test_validation_accepts_omegaconf_listconfig_rgb_values():
    omegaconf = pytest.importorskip("omegaconf")
    list_config = omegaconf.ListConfig([243, 229, 171])
    sys.modules.setdefault("seaborn", types.SimpleNamespace(color_palette=lambda *a, **k: []))
    sampling_mod = importlib.import_module("hs2p.sampling")

    pixel_mapping = {"background": 0, "gleason-3": 3}
    color_mapping = {"background": None, "gleason-3": list_config}

    sampling_mod._validate_visualization_color_mapping(
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )
