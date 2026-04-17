"""TDD tests for build_per_annotation_tiling_results with INDEPENDENT_SAMPLING strategy."""

from types import SimpleNamespace

import numpy as np
import pytest

from hs2p.preprocessing import ResolvedAnnotationMasks, build_per_annotation_tiling_results
from hs2p.wsi.types import CoordinateSelectionStrategy, CoordinateOutputMode, SamplingSpec

BASE_SPACING = 0.5
SLIDE_W, SLIDE_H = 400, 400


def _mock_slide():
    return SimpleNamespace(
        dimensions=(SLIDE_W, SLIDE_H),
        spacing=BASE_SPACING,
        level_downsamples=[1.0],
        level_dimensions=[(SLIDE_W, SLIDE_H)],
    )


def _annotation_mask():
    """
    400×400 mask:
      - tumor (value=1): top-left quadrant [row 0:200, col 0:200]
      - stroma (value=2): bottom-right quadrant [row 200:400, col 200:400]
      - background (value=0): everywhere else
    """
    mask = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint8)
    mask[0:200, 0:200] = 1
    mask[200:400, 200:400] = 2
    return mask


def _resolved_masks(mask):
    return ResolvedAnnotationMasks(
        masks={
            "tumor": np.where(mask == 1, 255, 0).astype(np.uint8),
            "stroma": np.where(mask == 2, 255, 0).astype(np.uint8),
        },
        tissue_method="precomputed_mask",
        seg_downsample=1,
        seg_level=0,
        seg_spacing_um=BASE_SPACING,
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2},
        mask_path=None,
        mask_level=None,
        mask_spacing_um=None,
    )


def _sampling_spec(tumor_threshold=0.1, stroma_threshold=0.1):
    return SamplingSpec(
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2},
        color_mapping=None,
        tissue_percentage={"background": None, "tumor": tumor_threshold, "stroma": stroma_threshold},
        active_annotations=("tumor", "stroma"),
    )


_COMMON_KWARGS = dict(
    image_path="/fake/slide.tiff",
    backend="mock",
    requested_backend="auto",
    sample_id="test_slide",
    requested_tile_size_px=64,
    requested_spacing_um=BASE_SPACING,
    overlap=0.0,
    tolerance=0.05,
    ref_tile_size_px=16,
    a_t=0,
    a_h=0,
)


def test_independent_sampling_returns_key_per_active_annotation():
    """Returns exactly one TilingResult per active_annotation."""
    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=_resolved_masks(_annotation_mask()),
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        **_COMMON_KWARGS,
    )
    assert set(results.keys()) == {"tumor", "stroma"}


def test_independent_sampling_tumor_tiles_within_tumor_region():
    """Tumor tiles lie in the top-left quadrant (x<200, y<200)."""
    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=_resolved_masks(_annotation_mask()),
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        **_COMMON_KWARGS,
    )
    r = results["tumor"]
    assert r.num_tiles > 0
    assert np.all(r.x < 200), "tumor tile x coords should be < 200"
    assert np.all(r.y < 200), "tumor tile y coords should be < 200"


def test_independent_sampling_stroma_tiles_within_stroma_region():
    """Stroma tiles lie in the bottom-right quadrant (x>=200, y>=200)."""
    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=_resolved_masks(_annotation_mask()),
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        **_COMMON_KWARGS,
    )
    r = results["stroma"]
    assert r.num_tiles > 0
    assert np.all(r.x >= 200), "stroma tile x coords should be >= 200"
    assert np.all(r.y >= 200), "stroma tile y coords should be >= 200"


def test_independent_sampling_annotation_field_on_result():
    """Each TilingResult.annotation matches its dict key."""
    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=_resolved_masks(_annotation_mask()),
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        **_COMMON_KWARGS,
    )
    assert results["tumor"].annotation == "tumor"
    assert results["stroma"].annotation == "stroma"


def test_independent_sampling_selection_strategy_field_on_result():
    """Each TilingResult.selection_strategy == INDEPENDENT_SAMPLING."""
    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=_resolved_masks(_annotation_mask()),
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        **_COMMON_KWARGS,
    )
    for result in results.values():
        assert result.selection_strategy == CoordinateSelectionStrategy.INDEPENDENT_SAMPLING


def test_independent_sampling_high_threshold_reduces_tile_count():
    """A stricter per-label coverage threshold yields fewer tiles than a lenient one."""
    mask = _annotation_mask()
    resolved = _resolved_masks(mask)
    slide = _mock_slide()

    results_strict = build_per_annotation_tiling_results(
        slide=slide,
        resolved_masks=resolved,
        sampling_spec=_sampling_spec(tumor_threshold=0.99),
        selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        **_COMMON_KWARGS,
    )
    results_lenient = build_per_annotation_tiling_results(
        slide=slide,
        resolved_masks=resolved,
        sampling_spec=_sampling_spec(tumor_threshold=0.01),
        selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        **_COMMON_KWARGS,
    )
    assert results_strict["tumor"].num_tiles <= results_lenient["tumor"].num_tiles


def test_independent_sampling_tile_index_is_contiguous():
    """tile_index for each result is a contiguous range [0, num_tiles)."""
    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=_resolved_masks(_annotation_mask()),
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        **_COMMON_KWARGS,
    )
    for result in results.values():
        expected = np.arange(result.num_tiles, dtype=np.int32)
        np.testing.assert_array_equal(result.tile_index, expected)


def test_independent_sampling_empty_annotation_region_returns_zero_tiles():
    """An annotation that covers no slide area yields an empty TilingResult."""
    mask = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint8)
    mask[0:200, 0:200] = 1  # only tumor, no stroma
    resolved = _resolved_masks(mask)

    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=resolved,
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        **_COMMON_KWARGS,
    )
    assert results["stroma"].num_tiles == 0
