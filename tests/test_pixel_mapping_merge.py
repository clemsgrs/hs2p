"""``pixel_mapping`` may map one label to several raw mask values, merged into a single class
whose coverage is the sum of its values."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from hs2p.configs.resolvers import resolve_sampling_spec, validate_pixel_mapping
from hs2p.wsi.masks import compose_overlay_mask_from_annotations
from hs2p.wsi.preview import build_overlay_alpha, build_palette
from hs2p.wsi.visualization import _combine_label_masks


def test_validate_pixel_mapping_rejects_value_under_two_labels():
    with pytest.raises(ValueError, match=r"'stroma' and 'tumor' both map to 2"):
        validate_pixel_mapping({"tumor": [1, 2], "stroma": 2})


def test_validate_pixel_mapping_rejects_empty_list():
    with pytest.raises(ValueError, match=r"pixel_mapping\['tumor'\]"):
        validate_pixel_mapping({"tumor": []})


@pytest.mark.parametrize("bad_member", [256, -1, True, 1.0, "2"])
def test_validate_pixel_mapping_rejects_invalid_list_member(bad_member):
    with pytest.raises(ValueError, match=r"pixel_mapping\['tumor'\]"):
        validate_pixel_mapping({"tumor": [1, bad_member]})


def _cfg(pixel_mapping, min_coverage):
    return OmegaConf.create(
        {"tiling": {"masks": {"pixel_mapping": pixel_mapping, "min_coverage": min_coverage}}}
    )


def test_config_list_value_resolves_to_plain_list_of_ints():
    spec = resolve_sampling_spec(
        _cfg({"background": 0, "tumor": [1, 2], "stroma": 3}, {"tumor": 0.5}),
        tiling=None,
    )
    assert spec.pixel_mapping == {"background": 0, "tumor": [1, 2], "stroma": 3}
    assert type(spec.pixel_mapping["tumor"]) is list
    assert [type(v) for v in spec.pixel_mapping["tumor"]] == [int, int]


def test_config_null_still_drops_a_label_next_to_list_values():
    spec = resolve_sampling_spec(
        _cfg({"background": 0, "tissue": None, "tumor": [1, 2]}, {"tumor": 0.5}),
        tiling=None,
    )
    assert spec.pixel_mapping == {"background": 0, "tumor": [1, 2]}


def test_palette_colors_every_listed_value():
    palette = build_palette(
        pixel_mapping={"background": 0, "tumor": [1, 3]},
        color_mapping={"background": None, "tumor": [255, 0, 0]},
    )
    expected = np.zeros(768, dtype=np.uint8)
    expected[3:6] = [255, 0, 0]
    expected[9:12] = [255, 0, 0]
    np.testing.assert_array_equal(palette, expected)


def test_overlay_alpha_covers_every_listed_value():
    alpha = build_overlay_alpha(
        mask_arr=np.array([[0, 1, 2, 3]], dtype=np.uint8),
        alpha=1.0,
        pixel_mapping={"background": 0, "tumor": [1, 3], "stroma": 2},
        color_mapping={"background": None, "tumor": [255, 0, 0], "stroma": None},
    )
    # 0 where the label overlay is drawn, 255 where the slide shows through.
    np.testing.assert_array_equal(np.array(alpha), [[255, 0, 255, 0]])


def test_recomposed_label_raster_paints_first_listed_value():
    tumor = np.array([[0, 255]], dtype=np.uint8)
    combined = _combine_label_masks(
        masks={"tumor": tumor}, pixel_mapping={"background": 0, "tumor": [2, 5]}
    )
    np.testing.assert_array_equal(combined, [[0, 2]])


def test_overlay_mask_from_annotations_paints_first_listed_value():
    overlay = compose_overlay_mask_from_annotations(
        annotation_mask={
            "tissue": np.array([[255, 255, 0]], dtype=np.uint8),
            "tumor": np.array([[0, 255, 0]], dtype=np.uint8),
        },
        pixel_mapping={"background": [7, 8], "tissue": 1, "tumor": [2, 5]},
    )
    np.testing.assert_array_equal(overlay, [[1, 2, 7]])
