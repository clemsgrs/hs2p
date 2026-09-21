"""``pixel_mapping`` may map one label to several raw mask values, merged into a single class
whose coverage is the sum of its values."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

import hs2p.tiling.mask as maskmod
import hs2p.tiling.single as singlemod
from hs2p.configs.resolvers import resolve_sampling_spec, validate_pixel_mapping
from hs2p.tiling.mask import resolve_annotation_masks
from hs2p.tiling.single import preprocess_slide_per_annotation
from hs2p.wsi.masks import compose_overlay_mask_from_annotations
from hs2p.wsi.preview import build_overlay_alpha, build_palette
from hs2p.wsi.types import CoordinateSelectionStrategy, SamplingSpec
from hs2p.wsi.visualization import _combine_label_masks

SPACING = 0.5
SLIDE_SIZE = 200
TILE_SIZE = 100


def _split_tumor_mask() -> np.ndarray:
    """Tile (0, 0) is 30% value 1 + 30% value 2; the other three tiles are all 0."""
    mask = np.zeros((SLIDE_SIZE, SLIDE_SIZE), dtype=np.uint8)
    mask[0:100, 0:30] = 1
    mask[0:100, 30:60] = 2
    return mask


class _FakeSlide:
    """Single-level slide; ``pixels`` is replicated to RGB by read_region."""

    def __init__(self, pixels: np.ndarray):
        self._pixels = pixels
        self.dimensions = (SLIDE_SIZE, SLIDE_SIZE)
        self.spacing = SPACING
        self.level_downsamples = [1.0]
        self.level_dimensions = [(SLIDE_SIZE, SLIDE_SIZE)]
        self.backend_name = "mock"

    def read_region(self, location, level, size):
        del location, level, size
        return np.repeat(self._pixels[:, :, None], 3, axis=2)

    def close(self) -> None:
        return None


@pytest.fixture
def patched_open(monkeypatch):
    from hs2p.wsi.backend import BackendSelection

    def fake_open(path, backend="auto", spacing_override=None):
        del backend, spacing_override
        if "mask" in str(path):
            return _FakeSlide(_split_tumor_mask())
        return _FakeSlide(np.full((SLIDE_SIZE, SLIDE_SIZE), 255, np.uint8))

    monkeypatch.setattr(singlemod, "open_slide", fake_open)
    monkeypatch.setattr(maskmod, "open_slide", fake_open)
    monkeypatch.setattr(
        maskmod,
        "resolve_backend",
        lambda requested, *, wsi_path, mask_path=None: BackendSelection(
            backend="mock", tried=("mock",)
        ),
    )


def _tile(pixel_mapping, min_coverage, strategy):
    sampling = SamplingSpec(
        pixel_mapping=pixel_mapping,
        color_mapping=None,
        tissue_percentage=min_coverage,
        active_annotations=tuple(k for k, v in min_coverage.items() if v is not None),
    )
    results = preprocess_slide_per_annotation(
        image_path="/fake/slide.tif",
        mask_path="/fake/slide_mask.tif",
        pixel_mapping=pixel_mapping,
        sampling_spec=sampling,
        selection_strategy=strategy,
        requested_tile_size_px=TILE_SIZE,
        requested_spacing_um=SPACING,
        seg_downsample=1,
        a_t=0,
    )
    return {
        name: list(zip(result.tiles.x.tolist(), result.tiles.y.tolist()))
        for name, result in results.items()
    }


STRATEGIES = [
    CoordinateSelectionStrategy.JOINT_SAMPLING,
    CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_merged_label_keeps_tile_whose_summed_coverage_passes(patched_open, strategy):
    tiles = _tile(
        {"background": 0, "tumor": [1, 2]},
        {"background": None, "tumor": 0.5},
        strategy,
    )
    assert tiles == {"tumor": [(0, 0)]}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_split_labels_drop_tile_that_fails_each_threshold(patched_open, strategy):
    tiles = _tile(
        {"background": 0, "a": 1, "b": 2},
        {"background": None, "a": 0.5, "b": 0.5},
        strategy,
    )
    assert tiles == {"a": [], "b": []}


def test_resolve_annotation_masks_unions_listed_values(patched_open):
    slide = SimpleNamespace(
        spacing=SPACING,
        level_downsamples=[1.0],
        level_dimensions=[(SLIDE_SIZE, SLIDE_SIZE)],
    )
    resolved = resolve_annotation_masks(
        slide=slide,
        mask_path="/fake/slide_mask.tif",
        pixel_mapping={"background": 0, "tumor": [1, 2]},
        seg_downsample=1,
    )
    expected = np.zeros((SLIDE_SIZE, SLIDE_SIZE), dtype=np.uint8)
    expected[0:100, 0:60] = 255
    np.testing.assert_array_equal(resolved.masks["tumor"], expected)
    assert resolved.pixel_mapping == {"background": 0, "tumor": [1, 2]}


def test_validate_pixel_mapping_rejects_value_under_two_labels():
    with pytest.raises(ValueError, match=r"'stroma' and 'tumor' both map to 2"):
        validate_pixel_mapping({"tumor": [1, 2], "stroma": 2})


def test_validate_pixel_mapping_rejects_empty_list():
    with pytest.raises(ValueError, match=r"pixel_mapping\['tumor'\]"):
        validate_pixel_mapping({"tumor": []})


def test_validate_pixel_mapping_rejects_duplicate_within_list():
    with pytest.raises(ValueError, match=r"pixel_mapping\['tumor'\] lists 1 more than once"):
        validate_pixel_mapping({"tumor": [1, 1]})


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


def test_config_scalar_values_stay_scalar_ints():
    spec = resolve_sampling_spec(
        _cfg({"background": 0, "tumor": 1, "stroma": 2}, {"tumor": 0.5}),
        tiling=None,
    )
    assert spec.pixel_mapping == {"background": 0, "tumor": 1, "stroma": 2}
    assert [type(v) for v in spec.pixel_mapping.values()] == [int, int, int]


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
