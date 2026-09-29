"""TDD tests for build_per_annotation_tiling_results with JOINT_SAMPLING strategy."""

from types import SimpleNamespace

import numpy as np
import pytest

from hs2p.preprocessing import ResolvedAnnotationMasks, build_per_annotation_tiling_results
from hs2p.wsi.types import CoordinateSelectionStrategy, SamplingSpec

BASE_SPACING = 0.5
SLIDE_W, SLIDE_H = 400, 400


def _mock_slide():
    return SimpleNamespace(
        dimensions=(SLIDE_W, SLIDE_H),
        spacing=BASE_SPACING,
        level_downsamples=[1.0],
        level_dimensions=[(SLIDE_W, SLIDE_H)],
    )


def _nonoverlapping_annotation_mask():
    """
    tumor: top-left quadrant [0:200, 0:200]
    stroma: bottom-right quadrant [200:400, 200:400]
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
        requested_seg_downsample=1,
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


def test_invalid_output_mode_fails_fast():
    """An unrecognized output_mode (e.g. an API typo) must raise, not silently fall through
    to per-annotation output."""
    with pytest.raises(ValueError, match="output_mode"):
        build_per_annotation_tiling_results(
            slide=_mock_slide(),
            resolved_masks=_resolved_masks(_nonoverlapping_annotation_mask()),
            sampling_spec=_sampling_spec(),
            selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
            output_mode="bogus_mode",
            **_COMMON_KWARGS,
        )


def test_joint_sampling_tiles_pass_per_label_coverage_threshold():
    """Every tile in each annotation result has tissue_fraction >= that annotation's threshold."""
    threshold = 0.5
    mask = _nonoverlapping_annotation_mask()
    resolved = _resolved_masks(mask)

    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=resolved,
        sampling_spec=_sampling_spec(tumor_threshold=threshold, stroma_threshold=threshold),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        **_COMMON_KWARGS,
    )

    for annotation, result in results.items():
        assert result.num_tiles > 0, f"Expected tiles for {annotation}"
        assert np.all(result.tissue_fractions >= threshold - 1e-6), (
            f"{annotation} tiles should have fraction >= {threshold}"
        )


def test_joint_sampling_selection_strategy_field_on_result():
    """Each TilingResult.selection_strategy == JOINT_SAMPLING."""
    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=_resolved_masks(_nonoverlapping_annotation_mask()),
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        **_COMMON_KWARGS,
    )
    for result in results.values():
        assert result.selection_strategy == CoordinateSelectionStrategy.JOINT_SAMPLING
