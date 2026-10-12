"""MaskSamplingConfig is the typed home of the annotation-mask vocabulary: label values,
overlay colors, per-label coverage thresholds, output mode and selection strategy."""

import pytest

import hs2p
from hs2p import MaskSamplingConfig


def test_mask_sampling_config_is_exported_from_hs2p():
    assert hs2p.MaskSamplingConfig is MaskSamplingConfig


def test_pixel_values_are_stored_as_tuples():
    masks = MaskSamplingConfig(
        pixel_mapping={"background": 0, "tumor": [1, 2]},
        min_coverage={"tumor": 0.5},
    )

    assert masks.pixel_mapping == {"background": (0,), "tumor": (1, 2)}


@pytest.mark.parametrize("value", [-1, 256, [3, 300]])
def test_pixel_values_outside_the_label_range_are_rejected(value):
    with pytest.raises(ValueError, match=r"outside the supported label range \[0, 255\]"):
        MaskSamplingConfig(pixel_mapping={"tumor": value}, min_coverage={})


def test_a_pixel_value_listed_under_two_labels_is_rejected():
    with pytest.raises(ValueError, match="'stroma' and 'tumor' both map to 2"):
        MaskSamplingConfig(
            pixel_mapping={"tumor": [1, 2], "stroma": 2}, min_coverage={}
        )


@pytest.mark.parametrize("coverage", [-0.1, 1.5])
def test_coverage_outside_zero_one_is_rejected(coverage):
    with pytest.raises(ValueError, match=r"min_coverage\['tumor'\].*\[0, 1\]"):
        MaskSamplingConfig(pixel_mapping={"tumor": 1}, min_coverage={"tumor": coverage})


@pytest.mark.parametrize("coverage", [0, 1, 0.0, 1.0])
def test_coverage_bounds_are_inclusive(coverage):
    masks = MaskSamplingConfig(
        pixel_mapping={"tumor": 1}, min_coverage={"tumor": coverage}
    )

    assert masks.min_coverage == {"tumor": float(coverage)}


@pytest.mark.parametrize("coverage", [None, True, "0.5"])
def test_coverage_must_be_a_number(coverage):
    with pytest.raises(ValueError, match=r"min_coverage\['tumor'\] must be a float"):
        MaskSamplingConfig(pixel_mapping={"tumor": 1}, min_coverage={"tumor": coverage})


def test_coverage_for_an_undeclared_label_is_rejected():
    with pytest.raises(ValueError, match="min_coverage references unknown labels: stroma"):
        MaskSamplingConfig(pixel_mapping={"tumor": 1}, min_coverage={"stroma": 0.5})


def test_colors_default_to_no_overlay_and_store_rgb_as_tuples():
    assert (
        MaskSamplingConfig(pixel_mapping={"tumor": 1}, min_coverage={}).colors is None
    )
    masks = MaskSamplingConfig(
        pixel_mapping={"background": 0, "tumor": 1},
        colors={"background": None, "tumor": [255, 0, 0]},
        min_coverage={"tumor": 0.5},
    )

    assert masks.colors == {"background": None, "tumor": (255, 0, 0)}


@pytest.mark.parametrize(
    "colors, message",
    [
        ({"tumor": [255, 0, 0]}, "missing annotation keys required by pixel_mapping: background"),
        (
            {"background": None, "tumor": None, "stroma": None},
            "unexpected annotation keys: stroma",
        ),
        ({"background": None, "tumor": [255, 0]}, "length-3 RGB sequence"),
        ({"background": None, "tumor": [255, 0, 256]}, r"integers in \[0, 255\]"),
    ],
)
def test_invalid_colors_are_rejected(colors, message):
    with pytest.raises(ValueError, match=message):
        MaskSamplingConfig(
            pixel_mapping={"background": 0, "tumor": 1},
            colors=colors,
            min_coverage={},
        )


@pytest.mark.parametrize(
    "field, value",
    [("output_mode", "per_label"), ("strategy", "independent_sampling")],
)
def test_unknown_output_mode_or_strategy_is_rejected(field, value):
    with pytest.raises(ValueError, match=f"MaskSamplingConfig.{field} must be one of"):
        MaskSamplingConfig(
            pixel_mapping={"tumor": 1}, min_coverage={}, **{field: value}
        )


@pytest.mark.parametrize("label", ["merged", "../tumor", ""])
def test_unsafe_or_reserved_label_names_are_rejected(label):
    with pytest.raises(ValueError, match="reserved|safe path component"):
        MaskSamplingConfig(pixel_mapping={label: 1}, min_coverage={})


def test_default_block_is_binary_tissue_at_one_percent_coverage():
    assert hs2p.DEFAULT_MASK_SAMPLING == MaskSamplingConfig(
        pixel_mapping={"background": 0, "tissue": 1},
        colors={"background": None, "tissue": (157, 219, 129)},
        min_coverage={"tissue": 0.01},
        output_mode="per_annotation",
        strategy="joint",
    )
