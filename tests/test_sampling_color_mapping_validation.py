import pytest

from hs2p.configs.resolvers import validate_color_mapping, validate_pixel_mapping


def test_validate_pixel_mapping_accepts_uint16_labels_without_background():
    # background is optional, and values up to the uint16 ceiling are allowed
    validate_pixel_mapping({"grade_4": 300, "grade_5": 600})


def test_validate_pixel_mapping_rejects_duplicate_values():
    with pytest.raises(ValueError, match="unique"):
        validate_pixel_mapping({"background": 0, "tumor": 1, "stroma": 1})


def test_validate_pixel_mapping_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="range"):
        validate_pixel_mapping({"background": 0, "tumor": 70000})
    with pytest.raises(ValueError, match="range"):
        validate_pixel_mapping({"background": 0, "tumor": -1})


def test_validate_pixel_mapping_rejects_non_integer_values():
    with pytest.raises(ValueError, match="integer"):
        validate_pixel_mapping({"background": 0, "tumor": 1.5})


@pytest.mark.parametrize("bad", ["../escape", "/tmp/owned", "a/b", "..", ".", "x\\y", ""])
def test_validate_pixel_mapping_rejects_unsafe_label_names(bad):
    # label names become output path components, so traversal/separators must be rejected
    with pytest.raises(ValueError, match="path component"):
        validate_pixel_mapping({"background": 0, bad: 1})


def test_validation_accepts_omegaconf_listconfig_rgb_values():
    omegaconf = pytest.importorskip("omegaconf")
    list_config = omegaconf.ListConfig([243, 229, 171])

    pixel_mapping = {"background": 0, "gleason-3": 3}
    color_mapping = {"background": None, "gleason-3": list_config}

    validate_color_mapping(
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )
