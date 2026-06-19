"""Tests for the annotation-mask producer (resolve_annotation_masks) and the per-class
coverage summary (summarize_annotation_coverage) — the previously-missing keystone for
build_per_annotation_tiling_results, plus the coverage utility soma's ingestion drives."""

from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.tiling.mask as maskmod
import hs2p.tiling.orchestration as orchmod
import hs2p.tiling.single as singlemod
import pandas as pd

from hs2p.api import FilterConfig, SlideSpec, TilingConfig, tile_slide, tile_slides
from hs2p.tiling.coverage import summarize_annotation_coverage
from hs2p.tiling.mask import resolve_annotation_masks
from hs2p.tiling.single import preprocess_slide_per_annotation
from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy, SamplingSpec

BASE_SPACING = 0.5
SLIDE_W, SLIDE_H = 400, 400
PIXEL_MAPPING = {"background": 0, "tumor": 1, "stroma": 2, "necrosis": 3}

# Areas (in level-0 == seg pixels here, downsample 1):
#   tumor    200x200 = 40000
#   stroma   200x200 = 40000
#   necrosis  40x40  =  1600   (a small focus in the top-right corner)
TUMOR_PX, STROMA_PX, NECROSIS_PX = 40000, 40000, 1600


def _label_mask() -> np.ndarray:
    mask = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint8)
    mask[0:200, 0:200] = 1
    mask[200:400, 200:400] = 2
    mask[0:40, 360:400] = 3
    return mask


class _FakeMaskSlide:
    """Single-level label-mask slide; read_region returns the label replicated to RGB."""

    def __init__(self, mask: np.ndarray, spacing: float):
        self._mask = mask
        self.spacing = spacing
        height, width = mask.shape
        self.level_dimensions = [(width, height)]
        self.level_downsamples = [1.0]

    def read_region(self, location, level, size):
        del location, level, size
        return np.repeat(self._mask[:, :, None], 3, axis=2)

    def close(self) -> None:
        return None


def _mock_slide() -> SimpleNamespace:
    return SimpleNamespace(
        dimensions=(SLIDE_W, SLIDE_H),
        spacing=BASE_SPACING,
        level_downsamples=[1.0],
        level_dimensions=[(SLIDE_W, SLIDE_H)],
        backend_name="mock",
    )


@pytest.fixture
def patched_mask_open(monkeypatch):
    mask = _label_mask()
    monkeypatch.setattr(
        maskmod,
        "open_slide",
        lambda path, backend=None: _FakeMaskSlide(mask, BASE_SPACING),
    )
    return mask


def test_resolve_annotation_masks_splits_per_non_background_class(patched_mask_open):
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask_path="/fake/slide_mask.tif",
        pixel_mapping=PIXEL_MAPPING,
        seg_downsample=1,
    )
    assert set(resolved.masks) == {"tumor", "stroma", "necrosis"}
    assert int(np.count_nonzero(resolved.masks["tumor"])) == TUMOR_PX
    assert int(np.count_nonzero(resolved.masks["stroma"])) == STROMA_PX
    assert int(np.count_nonzero(resolved.masks["necrosis"])) == NECROSIS_PX
    # binaries are 0 / 255
    assert set(np.unique(resolved.masks["tumor"]).tolist()) <= {0, 255}
    assert resolved.seg_spacing_um == pytest.approx(BASE_SPACING)
    assert resolved.seg_downsample == 1
    assert resolved.pixel_mapping == PIXEL_MAPPING


def test_resolve_annotation_masks_requires_background(patched_mask_open):
    with pytest.raises(ValueError, match="background"):
        resolve_annotation_masks(
            slide=_mock_slide(),
            mask_path="/fake/slide_mask.tif",
            pixel_mapping={"tumor": 1},
            seg_downsample=1,
        )


def test_summarize_annotation_coverage_area_frac_and_est_tiles(patched_mask_open):
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask_path="/fake/slide_mask.tif",
        pixel_mapping=PIXEL_MAPPING,
        seg_downsample=1,
    )
    summary = summarize_annotation_coverage(
        slide=_mock_slide(),
        resolved_masks=resolved,
        min_coverage={"tumor": 0.1, "stroma": 0.1, "necrosis": 0.1},
        requested_tile_size_px=200,
        requested_spacing_um=BASE_SPACING,
        overlap=0.0,
    )

    mm2_per_px = (BASE_SPACING / 1000.0) ** 2
    total = TUMOR_PX + STROMA_PX + NECROSIS_PX

    assert summary["tumor"]["area_mm2"] == pytest.approx(TUMOR_PX * mm2_per_px)
    assert summary["tumor"]["frac"] == pytest.approx(TUMOR_PX / total)
    assert summary["necrosis"]["frac"] == pytest.approx(NECROSIS_PX / total)

    # 2x2 grid of 200px tiles: tumor fills one tile, stroma fills one tile.
    assert summary["tumor"]["est_tiles"] == 1
    assert summary["stroma"]["est_tiles"] == 1
    # The necrosis focus covers only 1600/40000 = 0.04 of its tile (< 0.1) → ~0 tiles,
    # i.e. "present but trace" reads as no usable tiles (the design's intent).
    assert summary["necrosis"]["est_tiles"] == 0


def test_resolve_annotation_masks_recovers_via_openslide_on_degenerate_read(monkeypatch):
    """A backend that mis-decodes the mask to all-background (cucim + compressed minisblack)
    is recovered by the openslide fallback."""
    labeled = _label_mask()
    empty = np.zeros_like(labeled)

    def fake_open(path, backend=None):
        if str(backend).lower() == "openslide":
            return _FakeMaskSlide(labeled, BASE_SPACING)
        return _FakeMaskSlide(empty, BASE_SPACING)

    monkeypatch.setattr(maskmod, "open_slide", fake_open)
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask_path="/fake/slide_mask.tif",
        pixel_mapping=PIXEL_MAPPING,
        seg_downsample=1,
    )
    assert int(np.count_nonzero(resolved.masks["tumor"])) == TUMOR_PX
    assert int(np.count_nonzero(resolved.masks["stroma"])) == STROMA_PX
    assert int(np.count_nonzero(resolved.masks["necrosis"])) == NECROSIS_PX


def test_resolve_annotation_masks_genuinely_empty_stays_empty(monkeypatch):
    """When every backend agrees the mask is background, no spurious labels are invented."""
    empty = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint8)
    monkeypatch.setattr(
        maskmod, "open_slide", lambda path, backend=None: _FakeMaskSlide(empty, BASE_SPACING)
    )
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask_path="/fake/slide_mask.tif",
        pixel_mapping=PIXEL_MAPPING,
        seg_downsample=1,
    )
    assert all(int(np.count_nonzero(m)) == 0 for m in resolved.masks.values())


class _FakeSlide:
    """Minimal single-level slide reader for the per-annotation tiling path."""

    def __init__(self):
        self.dimensions = (SLIDE_W, SLIDE_H)
        self.spacing = BASE_SPACING
        self.level_downsamples = [1.0]
        self.level_dimensions = [(SLIDE_W, SLIDE_H)]
        self.backend_name = "mock"

    def read_region(self, location, level, size):
        del location, level
        width, height = int(size[0]), int(size[1])
        return np.full((height, width, 3), 255, np.uint8)

    def close(self) -> None:
        return None


def _sampling_spec():
    return SamplingSpec(
        pixel_mapping=PIXEL_MAPPING,
        color_mapping=None,
        tissue_percentage={"background": None, "tumor": 0.1, "stroma": 0.1, "necrosis": 0.1},
        active_annotations=("tumor", "stroma", "necrosis"),
    )


@pytest.fixture
def patched_slide_and_mask_open(monkeypatch):
    mask = _label_mask()

    def fake_open(path, backend="auto", spacing_override=None):
        del spacing_override
        if "mask" in str(path).lower():
            return _FakeMaskSlide(mask, BASE_SPACING)
        return _FakeSlide()

    monkeypatch.setattr(singlemod, "open_slide", fake_open)
    monkeypatch.setattr(maskmod, "open_slide", fake_open)


@pytest.mark.parametrize(
    "strategy",
    [
        CoordinateSelectionStrategy.JOINT_SAMPLING,
        CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
    ],
)
def test_preprocess_slide_per_annotation_wires_producer_to_sampler(
    patched_slide_and_mask_open, strategy
):
    results = preprocess_slide_per_annotation(
        image_path="/fake/slide.tif",
        mask_path="/fake/slide_mask.tif",
        pixel_mapping=PIXEL_MAPPING,
        sampling_spec=_sampling_spec(),
        selection_strategy=strategy,
        sample_id="slide0",
        requested_tile_size_px=64,
        requested_spacing_um=BASE_SPACING,
        seg_downsample=1,
        a_t=0,
    )
    assert set(results) == {"tumor", "stroma", "necrosis"}
    # tumor annotation occupies the top-left quadrant → its tiles stay there (origins < 200,
    # since tumor ends at row/col 200 and a tile origin >= 200 cannot overlap it).
    tumor = results["tumor"]
    assert tumor.num_tiles > 0
    assert (tumor.tiles.x < 200).all() and (tumor.tiles.y < 200).all()
    # stroma occupies the bottom-right quadrant; with a 64px grid the earliest origin that
    # can overlap the region starting at 200 is 192 (spans [192, 256)).
    stroma = results["stroma"]
    assert stroma.num_tiles > 0
    assert (stroma.tiles.x >= 192).all() and (stroma.tiles.y >= 192).all()


def test_tile_slide_with_sampling_returns_per_annotation_dict(monkeypatch):
    """tile_slide(..., sampling=spec) dispatches to the shared annotation core and returns
    one TilingResult per active annotation — same capability as tile_slides (no divergence)."""
    mask = _label_mask()

    def fake_open(path, backend="auto", spacing_override=None):
        del spacing_override
        if "mask" in str(path).lower():
            return _FakeMaskSlide(mask, BASE_SPACING)
        return _FakeSlide()

    monkeypatch.setattr(singlemod, "open_slide", fake_open)
    monkeypatch.setattr(maskmod, "open_slide", fake_open)
    monkeypatch.setattr(
        orchmod,
        "resolve_backend",
        lambda *a, **k: SimpleNamespace(backend="mock", reason=None),
    )

    whole_slide = SlideSpec(
        sample_id="slide0",
        image_path="/fake/slide.tif",
        mask_path="/fake/slide_mask.tif",
    )
    tiling = TilingConfig(
        requested_spacing_um=BASE_SPACING,
        requested_tile_size_px=64,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.0,
        backend="mock",
    )
    result = tile_slide(
        whole_slide,
        tiling=tiling,
        filtering=FilterConfig(a_t=0),
        sampling=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
    )
    assert isinstance(result, dict)
    assert set(result) == {"tumor", "stroma", "necrosis"}
    assert result["tumor"].num_tiles > 0


@pytest.mark.parametrize(
    "strategy",
    [
        CoordinateSelectionStrategy.JOINT_SAMPLING,
        CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
    ],
)
def test_single_output_merges_to_one_deduped_result_per_slide(
    patched_slide_and_mask_open, strategy
):
    """SINGLE_OUTPUT collapses the per-annotation fan-out into one merged result keyed by
    None: the dedup'd union of every tile passing any class threshold (the dense-seg
    contract — each spatial tile once, multi-class mask attached downstream)."""
    common = dict(
        image_path="/fake/slide.tif",
        mask_path="/fake/slide_mask.tif",
        pixel_mapping=PIXEL_MAPPING,
        sampling_spec=_sampling_spec(),
        selection_strategy=strategy,
        sample_id="slide0",
        requested_tile_size_px=64,
        requested_spacing_um=BASE_SPACING,
        seg_downsample=1,
        a_t=0,
    )
    per_anno = preprocess_slide_per_annotation(
        output_mode=CoordinateOutputMode.PER_ANNOTATION, **common
    )
    single = preprocess_slide_per_annotation(
        output_mode=CoordinateOutputMode.SINGLE_OUTPUT, **common
    )

    # one entry, keyed None, annotation cleared, output_mode tagged
    assert set(single) == {None}
    merged = single[None]
    assert merged.annotation is None
    assert merged.output_mode == CoordinateOutputMode.SINGLE_OUTPUT

    # merged coords == sorted unique union of the per-annotation coords
    expected = set()
    for res in per_anno.values():
        expected.update(zip(res.tiles.x.tolist(), res.tiles.y.tolist()))
    got = list(zip(merged.tiles.x.tolist(), merged.tiles.y.tolist()))
    assert len(got) == len(set(got))  # no duplicates
    assert set(got) == expected
    assert merged.num_tiles == len(expected) > 0
    # tile_index is a fresh contiguous range over the merged set
    assert merged.tiles.tile_index.tolist() == list(range(merged.num_tiles))


def _patch_tile_slides_open(monkeypatch):
    mask = _label_mask()

    def fake_open(path, backend="auto", spacing_override=None):
        del spacing_override
        if "mask" in str(path).lower():
            return _FakeMaskSlide(mask, BASE_SPACING)
        return _FakeSlide()

    monkeypatch.setattr(singlemod, "open_slide", fake_open)
    monkeypatch.setattr(maskmod, "open_slide", fake_open)
    monkeypatch.setattr(
        orchmod,
        "resolve_backend",
        lambda *a, **k: SimpleNamespace(backend="mock", reason=None),
    )


def _slides(n):
    return [
        SlideSpec(
            sample_id=f"slide{i}",
            image_path=f"/fake/slide{i}.tif",
            mask_path=f"/fake/slide{i}_mask.tif",
        )
        for i in range(n)
    ]


def _mock_tiling():
    return TilingConfig(
        requested_spacing_um=BASE_SPACING,
        requested_tile_size_px=64,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.0,
        backend="mock",
    )


def test_tile_slides_with_sampling_emits_one_artifact_per_slide_annotation(monkeypatch, tmp_path):
    _patch_tile_slides_open(monkeypatch)
    artifacts = tile_slides(
        _slides(2),
        tiling=_mock_tiling(),
        filtering=FilterConfig(a_t=0),
        output_dir=tmp_path,
        num_workers=1,
        sampling=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
    )
    # 2 slides × 3 active annotations = 6 artifacts, each tagged with its annotation.
    assert len(artifacts) == 6
    assert {a.annotation for a in artifacts} == {"tumor", "stroma", "necrosis"}

    rows = pd.read_csv(tmp_path / "process_list.csv")
    assert len(rows) == 6
    assert set(rows["annotation"]) == {"tumor", "stroma", "necrosis"}
    assert "tissue" not in set(rows["annotation"])  # sampling path, never the tissue label
    assert (rows["tiling_status"] == "success").all()


def test_tile_slides_single_output_emits_one_artifact_per_slide(monkeypatch, tmp_path):
    """With SINGLE_OUTPUT the sampling fan-out collapses: one artifact / process_list row
    per slide (annotation None), which soma's per-slide extraction path consumes unchanged."""
    _patch_tile_slides_open(monkeypatch)
    artifacts = tile_slides(
        _slides(2),
        tiling=_mock_tiling(),
        filtering=FilterConfig(a_t=0),
        output_dir=tmp_path,
        num_workers=1,
        sampling=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        output_mode=CoordinateOutputMode.SINGLE_OUTPUT,
    )
    assert len(artifacts) == 2  # one per slide, not per (slide, annotation)
    assert {a.sample_id for a in artifacts} == {"slide0", "slide1"}
    assert all(a.annotation is None for a in artifacts)

    rows = pd.read_csv(tmp_path / "process_list.csv")
    assert len(rows) == 2
    assert set(rows["sample_id"]) == {"slide0", "slide1"}
    assert (rows["tiling_status"] == "success").all()
    assert (rows["num_tiles"] > 0).all()


def test_tile_slides_sampling_rejects_unsupported_combos(monkeypatch, tmp_path):
    _patch_tile_slides_open(monkeypatch)
    with pytest.raises(NotImplementedError, match="resume"):
        tile_slides(
            _slides(1),
            tiling=_mock_tiling(),
            filtering=FilterConfig(a_t=0),
            output_dir=tmp_path,
            num_workers=1,
            sampling=_sampling_spec(),
            resume=True,
        )


def test_summarize_annotation_coverage_est_tiles_none_without_threshold(patched_mask_open):
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask_path="/fake/slide_mask.tif",
        pixel_mapping=PIXEL_MAPPING,
        seg_downsample=1,
    )
    summary = summarize_annotation_coverage(
        slide=_mock_slide(),
        resolved_masks=resolved,
        min_coverage={"tumor": 0.1},  # only tumor has a threshold
        requested_tile_size_px=200,
        requested_spacing_um=BASE_SPACING,
    )
    assert summary["tumor"]["est_tiles"] == 1
    assert summary["stroma"]["est_tiles"] is None
    assert summary["necrosis"]["est_tiles"] is None
