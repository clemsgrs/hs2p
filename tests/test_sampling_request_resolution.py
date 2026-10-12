"""CLI wiring for annotation sampling: resolve_mask_sampling_config builds the typed
MaskSamplingConfig from the merged ``tiling.masks`` block, and resolve_sampling_request
decides binary-tissue tiling vs annotation sampling from it."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from hs2p import DEFAULT_MASK_SAMPLING, MaskSamplingConfig
from hs2p.configs.loader import default_config
from hs2p.configs.resolvers import (
    require_tissue_fraction,
    resolve_mask_sampling_config,
    resolve_sampling_request,
    resolve_tiling_config,
)
from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy


def _cfg(overrides: dict | None = None):
    cfg = OmegaConf.create(default_config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _request(cfg):
    return resolve_sampling_request(masks=resolve_mask_sampling_config(cfg))


def test_untouched_default_masks_block_resolves_to_the_default_constant():
    masks = resolve_mask_sampling_config(_cfg())

    assert masks == DEFAULT_MASK_SAMPLING
    assert _request(_cfg()) == (None, None, None)


def test_changing_only_the_tissue_threshold_stays_binary_tissue_tiling():
    cfg = _cfg({"tiling": {"masks": {"min_coverage": {"tissue": 0.5}}}})

    masks = resolve_mask_sampling_config(cfg)

    assert masks.min_coverage == {"tissue": 0.5}
    assert require_tissue_fraction(masks) == 0.5
    assert _request(cfg) == (None, None, None)


def test_annotation_config_triggers_sampling_without_tissue_threshold():
    cfg = _cfg(
        {
            "tiling": {
                "masks": {
                    "output_mode": "merged",
                    "pixel_mapping": {"grade_4": 4, "grade_5": 5},
                    "colors": {"grade_4": [255, 0, 0], "grade_5": [0, 0, 255]},
                    # null out the default tissue threshold → pure-grade sampling
                    "min_coverage": {"tissue": None, "grade_4": 0.25, "grade_5": 0.25},
                }
            }
        }
    )

    sampling, strategy, output_mode = _request(cfg)

    assert sampling is not None
    assert set(sampling.active_annotations) == {"grade_4", "grade_5"}
    assert strategy == CoordinateSelectionStrategy.JOINT_SAMPLING
    assert output_mode == CoordinateOutputMode.MERGED


def test_masks_strategy_selects_independent_sampling():
    cfg = _cfg(
        {
            "tiling": {
                "masks": {
                    "strategy": "independent",
                    "pixel_mapping": {"grade_4": 4, "grade_5": 5},
                    "colors": {"grade_4": [255, 0, 0], "grade_5": [0, 0, 255]},
                    "min_coverage": {"tissue": None, "grade_4": 0.25, "grade_5": 0.25},
                },
            }
        }
    )

    _, strategy, _ = _request(cfg)

    assert strategy == CoordinateSelectionStrategy.INDEPENDENT_SAMPLING


def test_retired_independent_sampling_key_points_to_masks_strategy():
    cfg = _cfg({"tiling": {"independent_sampling": True}})

    with pytest.raises(ValueError, match=r"tiling\.masks\.strategy"):
        resolve_mask_sampling_config(cfg)


def test_label_reusing_value_one_works_when_default_tissue_is_nulled():
    """Configs deep-merge over the default {background:0, tissue:1}; nulling the tissue pixel
    value drops it so a 0/1 annotation mask (tumor:1) configures without a duplicate-value error."""
    cfg = _cfg(
        {
            "tiling": {
                "masks": {
                    "pixel_mapping": {"tissue": None, "tumor": 1},
                    "colors": {"tissue": None, "tumor": [1, 2, 3]},
                    "min_coverage": {"tissue": None, "tumor": 0.5},
                }
            }
        }
    )

    sampling, _, _ = _request(cfg)

    assert sampling is not None
    assert dict(sampling.pixel_mapping) == {"background": (0,), "tumor": (1,)}
    assert set(sampling.active_annotations) == {"tumor"}


def test_background_label_is_not_reserved_at_activation():
    """No label name is special at the CLI activation boundary: a thresholded 'background'
    class enables annotation sampling instead of falling through to binary tissue."""
    cfg = _cfg(
        {
            "tiling": {
                "masks": {
                    "pixel_mapping": {"tissue": None},
                    "colors": {"tissue": None},
                    "min_coverage": {"tissue": None, "background": 0.5},
                }
            }
        }
    )

    sampling, _, _ = _request(cfg)

    assert sampling is not None
    assert set(sampling.active_annotations) == {"background"}


def test_annotation_config_rejects_reserved_name_before_null_to_drop():
    cfg = _cfg(
        {
            "tiling": {
                "masks": {
                    "pixel_mapping": {"merged": None},
                    "colors": {"merged": None},
                    "min_coverage": {"merged": 0.5},
                }
            }
        }
    )

    with pytest.raises(ValueError, match="'merged'.*reserved"):
        resolve_mask_sampling_config(cfg)


def test_tissue_tiling_requires_a_tissue_threshold():
    """Binary tissue tiling hard-errors on a missing tissue threshold (no silent 0.0),
    honours an explicit 0.0 opt-out, and carries a specified value through."""
    assert require_tissue_fraction(
        replace(DEFAULT_MASK_SAMPLING, min_coverage={"tissue": 0.2})
    ) == 0.2
    # Explicit 0.0 is a deliberate "no tissue filtering" opt-out, not an inferred default.
    assert require_tissue_fraction(
        replace(DEFAULT_MASK_SAMPLING, min_coverage={"tissue": 0.0})
    ) == 0.0
    with pytest.raises(ValueError, match="min_coverage.tissue is required"):
        require_tissue_fraction(replace(DEFAULT_MASK_SAMPLING, min_coverage={}))


def test_masks_output_mode_is_validated():
    with pytest.raises(ValueError, match=r"tiling\.masks\.output_mode must be one of"):
        resolve_mask_sampling_config(
            _cfg({"tiling": {"masks": {"output_mode": "bogus"}}})
        )


_LEGACY_SAMPLING_PARAMS = {
    "pixel_mapping": [{"background": 0}, {"tissue": 1}, {"tumor": 2}],
    "color_mapping": [{"background": None}, {"tissue": None}, {"tumor": [255, 0, 0]}],
    "tissue_percentage": [{"background": None}, {"tissue": None}, {"tumor": 0.5}],
}

_CURRENT_MASKS = {
    "pixel_mapping": {"background": 0, "tissue": 1},
    "colors": {"background": None, "tissue": [157, 219, 129]},
    "min_coverage": {"background": None, "tissue": 0.01},
}


@pytest.mark.parametrize(
    "resolve",
    [resolve_tiling_config, resolve_mask_sampling_config],
    ids=["resolve_tiling_config", "resolve_mask_sampling_config"],
)
@pytest.mark.parametrize(
    "tiling_section",
    [
        {"sampling_params": _LEGACY_SAMPLING_PARAMS},
        {"sampling_params": _LEGACY_SAMPLING_PARAMS, "masks": _CURRENT_MASKS},
    ],
    ids=["alone", "alongside_masks"],
)
def test_direct_resolvers_reject_retired_sampling_params(resolve, tiling_section):
    """Unmerged configs passed straight to a resolver never reach a legacy parser: the
    retired key is refused whether or not a current masks section sits next to it."""
    cfg = OmegaConf.create(
        {
            "tiling": {
                "params": dict(default_config.tiling.params),
                "backend": "auto",
                **tiling_section,
            }
        }
    )

    with pytest.raises(ValueError, match=r"tiling\.sampling_params.*tiling\.masks"):
        resolve(cfg)


def test_masks_section_carries_colors_and_per_label_coverage():
    """The masks schema keeps colors and per-label coverage, and samples exactly the labels
    with a non-null coverage threshold."""
    cfg = OmegaConf.create(
        {
            "tiling": {
                "masks": {
                    "pixel_mapping": [{"background": 0}, {"tumor": 2}, {"stroma": [3, 4]}],
                    "colors": [
                        {"background": None},
                        {"tumor": [255, 0, 0]},
                        {"stroma": [0, 0, 255]},
                    ],
                    "min_coverage": [{"background": None}, {"tumor": 0.5}, {"stroma": 0.25}],
                }
            }
        }
    )

    masks = resolve_mask_sampling_config(cfg)

    assert masks == MaskSamplingConfig(
        pixel_mapping={"background": (0,), "tumor": (2,), "stroma": (3, 4)},
        colors={"background": None, "tumor": (255, 0, 0), "stroma": (0, 0, 255)},
        min_coverage={"tumor": 0.5, "stroma": 0.25},
    )
    sampling, _, _ = resolve_sampling_request(masks=masks)
    assert sampling.active_annotations == ("tumor", "stroma")


def test_config_without_masks_section_gets_the_default_block():
    cfg = SimpleNamespace(tiling=SimpleNamespace())

    assert resolve_mask_sampling_config(cfg) is DEFAULT_MASK_SAMPLING
