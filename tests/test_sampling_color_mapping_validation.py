import pytest

from hs2p.configs.resolvers import validate_color_mapping


def test_validation_accepts_omegaconf_listconfig_rgb_values():
    omegaconf = pytest.importorskip("omegaconf")
    list_config = omegaconf.ListConfig([243, 229, 171])

    pixel_mapping = {"background": 0, "gleason-3": 3}
    color_mapping = {"background": None, "gleason-3": list_config}

    validate_color_mapping(
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )
