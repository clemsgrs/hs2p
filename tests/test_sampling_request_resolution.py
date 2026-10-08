"""CLI wiring for annotation sampling: resolve_sampling_request decides binary-tissue vs
annotation sampling from the merged config, resolve_output_mode validates the mode, and
resolve_tiling_config tolerates configs that declare no 'tissue' threshold."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from hs2p.configs.loader import default_config
from hs2p.configs.models import TilingConfig
from hs2p.configs.resolvers import (
    build_default_sampling_spec,
    resolve_output_mode,
    resolve_sampling_request,
    resolve_sampling_spec,
    resolve_tiling_config,
)
from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy


def _tiling(min_coverage):
    return TilingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        min_coverage=min_coverage,
    )


def _cfg(overrides: dict | None = None):
    cfg = OmegaConf.create(default_config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


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
    # No 'tissue' threshold must not raise here (it used to KeyError).
    tiling = resolve_tiling_config(cfg)
    assert (tiling.min_coverage.get("tissue") or 0.0) == 0.0

    sampling, strategy, output_mode = resolve_sampling_request(cfg, tiling=tiling)
    assert sampling is not None
    assert set(sampling.active_annotations) == {"grade_4", "grade_5"}
    assert strategy == CoordinateSelectionStrategy.JOINT_SAMPLING
    assert output_mode == CoordinateOutputMode.MERGED


def test_independent_sampling_flag_selects_strategy():
    cfg = _cfg(
        {
            "tiling": {
                "independent_sampling": True,
                "masks": {
                    "pixel_mapping": {"grade_4": 4, "grade_5": 5},
                    "colors": {"grade_4": [255, 0, 0], "grade_5": [0, 0, 255]},
                    "min_coverage": {"tissue": None, "grade_4": 0.25, "grade_5": 0.25},
                },
            }
        }
    )
    tiling = resolve_tiling_config(cfg)
    _, strategy, _ = resolve_sampling_request(cfg, tiling=tiling)
    assert strategy == CoordinateSelectionStrategy.INDEPENDENT_SAMPLING


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
    sampling, _, _ = resolve_sampling_request(cfg, tiling=resolve_tiling_config(cfg))
    assert sampling is not None
    assert dict(sampling.pixel_mapping) == {"background": 0, "tumor": 1}
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
    sampling, _, _ = resolve_sampling_request(cfg, tiling=resolve_tiling_config(cfg))
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
        resolve_sampling_request(cfg, tiling=resolve_tiling_config(cfg))


def test_build_default_sampling_spec_requires_tissue_coverage():
    """The binary-tissue default spec hard-errors on a missing tissue threshold (no silent
    0.0), honours an explicit 0.0 opt-out, and carries a specified value through."""
    assert build_default_sampling_spec(_tiling({"tissue": 0.2})).tissue_percentage[
        "tissue"
    ] == 0.2
    # Explicit 0.0 is a deliberate "no tissue filtering" opt-out, not an inferred default.
    assert build_default_sampling_spec(_tiling({"tissue": 0.0})).tissue_percentage[
        "tissue"
    ] == 0.0
    with pytest.raises(ValueError, match="min_coverage.tissue is required"):
        build_default_sampling_spec(_tiling({}))


def test_resolve_output_mode_default_and_validation():
    assert resolve_output_mode(_cfg()) == CoordinateOutputMode.PER_ANNOTATION
    with pytest.raises(ValueError, match="output_mode"):
        resolve_output_mode(_cfg({"tiling": {"masks": {"output_mode": "bogus"}}}))


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


def _resolve_tiling(cfg):
    return resolve_tiling_config(cfg)


def _resolve_spec(cfg):
    return resolve_sampling_spec(cfg, tiling=_tiling({"tissue": 0.01}))


def _resolve_request(cfg):
    return resolve_sampling_request(cfg, tiling=_tiling({"tissue": 0.01}))


@pytest.mark.parametrize(
    "resolve", [_resolve_tiling, _resolve_spec, _resolve_request],
    ids=["resolve_tiling_config", "resolve_sampling_spec", "resolve_sampling_request"],
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
                "independent_sampling": False,
                "backend": "auto",
                **tiling_section,
            }
        }
    )

    with pytest.raises(ValueError, match=r"tiling\.sampling_params.*tiling\.masks"):
        resolve(cfg)


def test_masks_section_carries_colors_and_per_label_coverage():
    """The masks schema (the only sampling schema) keeps colors and per-label coverage, and
    samples exactly the labels with a non-null coverage threshold."""
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

    spec = resolve_sampling_spec(cfg, tiling=_tiling({}))

    assert dict(spec.pixel_mapping) == {"background": 0, "tumor": 2, "stroma": [3, 4]}
    assert dict(spec.color_mapping) == {
        "background": None,
        "tumor": [255, 0, 0],
        "stroma": [0, 0, 255],
    }
    assert dict(spec.tissue_percentage) == {
        "background": None,
        "tumor": 0.5,
        "stroma": 0.25,
    }
    assert spec.active_annotations == ("tumor", "stroma")


def test_config_without_masks_section_samples_default_tissue():
    """An adapter that declares no masks section gets binary tissue sampling at the
    resolved tissue threshold."""
    cfg = SimpleNamespace(tiling=SimpleNamespace())

    spec = resolve_sampling_spec(cfg, tiling=_tiling({"tissue": 0.2}))

    assert dict(spec.pixel_mapping) == {"background": 0, "tissue": 1}
    assert spec.active_annotations == ("tissue",)
    assert spec.tissue_percentage["tissue"] == 0.2
