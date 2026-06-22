"""CLI wiring for annotation sampling: resolve_sampling_request decides binary-tissue vs
annotation sampling from the merged config, resolve_output_mode validates the mode, and
resolve_tiling_config tolerates configs that declare no 'tissue' threshold."""

import pytest
from omegaconf import OmegaConf

from hs2p.configs.loader import default_config
from hs2p.configs.resolvers import (
    resolve_output_mode,
    resolve_sampling_request,
    resolve_tiling_config,
)
from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy


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
    assert tiling.tissue_threshold == 0.01


def test_annotation_config_triggers_sampling_without_tissue_threshold():
    cfg = _cfg(
        {
            "tiling": {
                "masks": {
                    "output_mode": "single_output",
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
    assert tiling.tissue_threshold == 0.0

    sampling, strategy, output_mode = resolve_sampling_request(cfg, tiling=tiling)
    assert sampling is not None
    assert set(sampling.active_annotations) == {"grade_4", "grade_5"}
    assert strategy == CoordinateSelectionStrategy.JOINT_SAMPLING
    assert output_mode == CoordinateOutputMode.SINGLE_OUTPUT


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


def test_resolve_output_mode_default_and_validation():
    assert resolve_output_mode(_cfg()) == CoordinateOutputMode.PER_ANNOTATION
    with pytest.raises(ValueError, match="output_mode"):
        resolve_output_mode(_cfg({"tiling": {"masks": {"output_mode": "bogus"}}}))
