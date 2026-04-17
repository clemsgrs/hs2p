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


def _overlapping_annotation_mask():
    """
    400×400 mask with overlapping regions:
      - tumor (value=1): [row 0:300, col 0:300]
      - stroma (value=2): [row 100:400, col 100:400]
    In the overlap zone [row 100:300, col 100:300] we set stroma (2).
    So the final mask is: tumor where (row<100 or col<100), stroma elsewhere in tissue.
    """
    mask = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint8)
    mask[0:300, 0:300] = 1   # tumor
    mask[100:400, 100:400] = 2  # stroma overwrites overlap
    return mask


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


def test_joint_sampling_returns_key_per_active_annotation():
    """Returns exactly one TilingResult per active_annotation."""
    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=_resolved_masks(_nonoverlapping_annotation_mask()),
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        **_COMMON_KWARGS,
    )
    assert set(results.keys()) == {"tumor", "stroma"}


def test_joint_sampling_annotation_field_on_result():
    """Each TilingResult.annotation matches its dict key."""
    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=_resolved_masks(_nonoverlapping_annotation_mask()),
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        **_COMMON_KWARGS,
    )
    assert results["tumor"].annotation == "tumor"
    assert results["stroma"].annotation == "stroma"


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


def test_joint_sampling_tiles_lie_within_union_region():
    """All joint-sampled tiles lie within the union of annotation regions (not pure background)."""
    mask = _nonoverlapping_annotation_mask()
    union_mask = (mask > 0).astype(np.uint8)  # 1 where any annotation
    resolved = _resolved_masks(mask)
    slide = _mock_slide()

    results = build_per_annotation_tiling_results(
        slide=slide,
        resolved_masks=resolved,
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        **_COMMON_KWARGS,
    )

    tile_size = 64
    for annotation, result in results.items():
        for x, y in zip(result.x, result.y):
            x_end = min(int(x) + tile_size, SLIDE_W)
            y_end = min(int(y) + tile_size, SLIDE_H)
            tile_region = union_mask[int(y):y_end, int(x):x_end]
            assert tile_region.sum() > 0, (
                f"{annotation} tile at ({x},{y}) contains no annotation pixels"
            )



def test_joint_sampling_tile_index_is_contiguous():
    """tile_index for each result is a contiguous range [0, num_tiles)."""
    results = build_per_annotation_tiling_results(
        slide=_mock_slide(),
        resolved_masks=_resolved_masks(_nonoverlapping_annotation_mask()),
        sampling_spec=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        **_COMMON_KWARGS,
    )
    for result in results.values():
        expected = np.arange(result.num_tiles, dtype=np.int32)
        np.testing.assert_array_equal(result.tile_index, expected)


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
