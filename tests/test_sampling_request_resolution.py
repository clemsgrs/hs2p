"""CLI wiring for annotation sampling: resolve_sampling_request decides binary-tissue vs
annotation sampling from the merged config, resolve_output_mode validates the mode, and
resolve_tiling_config tolerates configs that declare no 'tissue' threshold."""

import pytest
from omegaconf import OmegaConf

from hs2p.configs.loader import default_config
from hs2p.configs.models import TilingConfig
from hs2p.configs.resolvers import (
    build_default_sampling_spec,
    resolve_output_mode,
    resolve_sampling_request,
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


def test_default_config_routes_to_binary_tissue():
    cfg = _cfg()
    tiling = resolve_tiling_config(cfg)
    sampling, strategy, output_mode = resolve_sampling_request(cfg, tiling=tiling)
    assert sampling is None and strategy is None and output_mode is None
    assert (tiling.min_coverage.get("tissue") or 0.0) == 0.01


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


def test_annotation_request_does_not_require_tissue_coverage():
    """resolve_sampling_request must not build the tissue-requiring default spec for an
    annotation-only config — it detects the untouched default via constants instead."""
    cfg = _cfg(
        {
            "tiling": {
                "masks": {
                    "pixel_mapping": {"grade_4": 4, "grade_5": 5},
                    "colors": {"grade_4": [255, 0, 0], "grade_5": [0, 0, 255]},
                    "min_coverage": {"tissue": None, "grade_4": 0.25, "grade_5": 0.25},
                }
            }
        }
    )
    sampling, _, _ = resolve_sampling_request(cfg, tiling=resolve_tiling_config(cfg))
    assert sampling is not None
    assert set(sampling.active_annotations) == {"grade_4", "grade_5"}


def test_resolve_output_mode_default_and_validation():
    assert resolve_output_mode(_cfg()) == CoordinateOutputMode.PER_ANNOTATION
    with pytest.raises(ValueError, match="output_mode"):
        resolve_output_mode(_cfg({"tiling": {"masks": {"output_mode": "bogus"}}}))
