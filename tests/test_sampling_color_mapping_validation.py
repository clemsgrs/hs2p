import pytest

from hs2p.configs.resolvers import validate_color_mapping, validate_pixel_mapping


def test_validate_pixel_mapping_accepts_preview_safe_boundaries_without_background():
    validate_pixel_mapping({"grade_4": 0, "grade_5": 255})


def test_validate_pixel_mapping_rejects_duplicate_values():
    with pytest.raises(ValueError, match="unique"):
        validate_pixel_mapping({"background": 0, "tumor": 1, "stroma": 1})


@pytest.mark.parametrize("invalid_value", [-1, 256])
def test_validate_pixel_mapping_rejects_value_outside_preview_safe_range(invalid_value):
    with pytest.raises(
        ValueError,
        match=rf"tumor.*{invalid_value}.*range \[0, 255\]",
    ):
        validate_pixel_mapping({"background": 0, "tumor": invalid_value})


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
