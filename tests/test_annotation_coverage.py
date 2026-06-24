"""Tests for the annotation-mask producer (resolve_annotation_masks) and the per-class
coverage summary (summarize_annotation_coverage) — the previously-missing keystone for
build_per_annotation_tiling_results, plus the coverage utility soma's ingestion drives."""

from pathlib import Path
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


def test_resolve_annotation_masks_splits_per_declared_label(patched_mask_open):
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask_path="/fake/slide_mask.tif",
        pixel_mapping=PIXEL_MAPPING,
        seg_downsample=1,
    )
    # No reserved name: every declared label gets a binary, including "background".
    assert set(resolved.masks) == {"background", "tumor", "stroma", "necrosis"}
    assert int(np.count_nonzero(resolved.masks["tumor"])) == TUMOR_PX
    assert int(np.count_nonzero(resolved.masks["stroma"])) == STROMA_PX
    assert int(np.count_nonzero(resolved.masks["necrosis"])) == NECROSIS_PX
    assert int(np.count_nonzero(resolved.masks["background"])) == (
        SLIDE_H * SLIDE_W - TUMOR_PX - STROMA_PX - NECROSIS_PX
    )
    # binaries are 0 / 255
    assert set(np.unique(resolved.masks["tumor"]).tolist()) <= {0, 255}
    assert resolved.seg_spacing_um == pytest.approx(BASE_SPACING)
    assert resolved.seg_downsample == 1
    assert resolved.pixel_mapping == PIXEL_MAPPING


def test_resolve_annotation_masks_preserves_uint16_labels_above_255(monkeypatch):
    """Label values above 255 stored in a uint16 raster must not wrap to uint8.

    The reader validates labels against the full pixel-value set (which is not bounded to
    255), so an unconditional uint8 downcast would wrap e.g. 300 -> 44, silently dropping a
    class or merging it into whichever class owns the wrapped value.
    """
    mask = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint16)
    mask[0:200, 0:200] = 300  # would wrap to 44 under a uint8 downcast
    mask[200:400, 200:400] = 600  # would wrap to 88
    monkeypatch.setattr(
        maskmod,
        "open_slide",
        lambda path, backend=None: _FakeMaskSlide(mask, BASE_SPACING),
    )

    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask_path="/fake/slide_mask.tif",
        pixel_mapping={"background": 0, "tumor": 300, "stroma": 600},
        seg_downsample=1,
    )
    assert int(np.count_nonzero(resolved.masks["tumor"])) == 200 * 200
    assert int(np.count_nonzero(resolved.masks["stroma"])) == 200 * 200


def test_resolve_annotation_masks_background_is_optional(monkeypatch):
    """A raster that labels every pixel (no reserved background value) needs no 'background'
    entry — every declared class still validates and is split into a binary."""
    mask = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint8)
    mask[:, : SLIDE_W // 2] = 1  # tumor over the left half
    mask[:, SLIDE_W // 2 :] = 2  # stroma over the right half
    monkeypatch.setattr(
        maskmod,
        "open_slide",
        lambda path, backend=None: _FakeMaskSlide(mask, BASE_SPACING),
    )

    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask_path="/fake/slide_mask.tif",
        pixel_mapping={"tumor": 1, "stroma": 2},
        seg_downsample=1,
    )
    assert set(resolved.masks) == {"tumor", "stroma"}
    assert int(np.count_nonzero(resolved.masks["tumor"])) == SLIDE_H * (SLIDE_W // 2)
    assert int(np.count_nonzero(resolved.masks["stroma"])) == SLIDE_H * (SLIDE_W // 2)


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


def test_openslide_fallback_recovers_when_primary_read_rejects_all_zero(monkeypatch):
    """No-background vocabulary + a degenerate all-zero primary read.

    With no declared 0, the all-zero mis-decoded level fails the discreteness guard and the
    primary read *raises* (rather than returning an empty mask). The openslide fallback must
    still kick in and recover the labels — otherwise the slide errors out.
    """
    # Fully-labeled raster (only values 1/2, no background) so the openslide read validates.
    labeled = np.empty((SLIDE_H, SLIDE_W), dtype=np.uint8)
    labeled[:, : SLIDE_W // 2] = 1
    labeled[:, SLIDE_W // 2 :] = 2
    empty = np.zeros_like(labeled)

    def fake_open(path, backend=None):
        if str(backend).lower() == "openslide":
            return _FakeMaskSlide(labeled, BASE_SPACING)
        return _FakeMaskSlide(empty, BASE_SPACING)  # cucim mis-decode → all zero → rejected

    monkeypatch.setattr(maskmod, "open_slide", fake_open)
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask_path="/fake/slide_mask.tif",
        pixel_mapping={"tumor": 1, "stroma": 2},
        seg_downsample=1,
    )
    assert int(np.count_nonzero(resolved.masks["tumor"])) == SLIDE_H * (SLIDE_W // 2)
    assert int(np.count_nonzero(resolved.masks["stroma"])) == SLIDE_H * (SLIDE_W // 2)


def test_openslide_fallback_accepts_valid_all_zero_after_primary_error(monkeypatch):
    """Primary read raises (e.g. a decode error), openslide recovers a valid all-zero mask.

    With background (0) declared, an all-zero raster is a legitimate read — the fallback must
    be accepted even though it has no foreground, instead of re-raising the primary error.
    """

    class _RaisingMaskSlide:
        spacing = BASE_SPACING
        level_dimensions = [(SLIDE_W, SLIDE_H)]
        level_downsamples = [1.0]

        def read_region(self, location, level, size):
            raise RuntimeError("backend decode error")

        def close(self):
            return None

    empty = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint8)

    def fake_open(path, backend=None):
        if str(backend).lower() == "openslide":
            return _FakeMaskSlide(empty, BASE_SPACING)  # valid: 0 is declared background
        return _RaisingMaskSlide()  # primary backend raises

    monkeypatch.setattr(maskmod, "open_slide", fake_open)
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask_path="/fake/slide_mask.tif",
        pixel_mapping={"background": 0, "tumor": 1},
        seg_downsample=1,
    )
    assert int(np.count_nonzero(resolved.masks["tumor"])) == 0
    assert int(np.count_nonzero(resolved.masks["background"])) == SLIDE_H * SLIDE_W


def test_resolve_annotation_masks_genuinely_empty_stays_empty(monkeypatch):
    """When every backend agrees the mask is background, no spurious real labels are invented.

    The background binary correctly fills the slide (every pixel is the background value); the
    point is that no foreground class picks up phantom pixels.
    """
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
    assert all(
        int(np.count_nonzero(resolved.masks[name])) == 0
        for name in ("tumor", "stroma", "necrosis")
    )
    assert int(np.count_nonzero(resolved.masks["background"])) == SLIDE_H * SLIDE_W


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
        min_coverage={"tissue": 0.0},
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
def test_merged_merges_to_one_deduped_result_per_slide(
    patched_slide_and_mask_open, strategy
):
    """MERGED collapses the per-annotation fan-out into one merged result keyed by
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
        output_mode=CoordinateOutputMode.MERGED, **common
    )

    # one entry, keyed None, annotation cleared, output_mode tagged
    assert set(single) == {None}
    merged = single[None]
    assert merged.annotation is None
    assert merged.output_mode == CoordinateOutputMode.MERGED

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
        min_coverage={"tissue": 0.0},
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


def test_tile_slides_sampling_progress_counts_slides_not_artifacts(monkeypatch, tmp_path):
    """Slide-completion progress must track slides, not per-annotation artifacts: a 2-slide ×
    3-annotation run reports completed=2/total=2, never completed=6."""
    _patch_tile_slides_open(monkeypatch)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        orchmod, "emit_progress", lambda name, **kw: events.append((name, kw))
    )
    tile_slides(
        _slides(2),
        tiling=_mock_tiling(),
        filtering=FilterConfig(a_t=0),
        output_dir=tmp_path,
        num_workers=1,
        sampling=_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
    )
    progress = [kw for name, kw in events if name == "tiling.progress"]
    assert progress, "expected tiling.progress events"
    assert all(p["total"] == 2 for p in progress)
    assert max(p["completed"] for p in progress) == 2
    assert all(p["completed"] <= p["total"] for p in progress)


def test_tile_slides_merged_emits_one_artifact_per_slide(monkeypatch, tmp_path):
    """With MERGED the sampling fan-out collapses: one artifact / process_list row
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
        output_mode=CoordinateOutputMode.MERGED,
    )
    assert len(artifacts) == 2  # one per slide, not per (slide, annotation)
    assert {a.sample_id for a in artifacts} == {"slide0", "slide1"}
    assert all(a.annotation is None for a in artifacts)

    rows = pd.read_csv(tmp_path / "process_list.csv")
    assert len(rows) == 2
    assert set(rows["sample_id"]) == {"slide0", "slide1"}
    assert (rows["tiling_status"] == "success").all()
    assert (rows["num_tiles"] > 0).all()
    # Merged single-output rows must be distinguishable from binary tissue tiling.
    assert set(rows["annotation"]) == {"merged"}
    assert (rows["output_mode"] == CoordinateOutputMode.MERGED).all()


@pytest.mark.parametrize("unsupported", ["resume", "read_coordinates_from", "save_tiles"])
def test_tile_slides_sampling_rejects_unsupported_combos(monkeypatch, tmp_path, unsupported):
    _patch_tile_slides_open(monkeypatch)
    kwargs = {
        "resume": {"resume": True},
        "read_coordinates_from": {"read_coordinates_from": tmp_path},
        "save_tiles": {"save_tiles": True},
    }[unsupported]
    with pytest.raises(NotImplementedError, match=unsupported):
        tile_slides(
            _slides(1),
            tiling=_mock_tiling(),
            filtering=FilterConfig(a_t=0),
            output_dir=tmp_path,
            num_workers=1,
            sampling=_sampling_spec(),
            **kwargs,
        )


def _sampling_spec_with_colors():
    return SamplingSpec(
        pixel_mapping=PIXEL_MAPPING,
        color_mapping={
            "background": None,
            "tumor": [255, 0, 0],
            "stroma": [0, 255, 0],
            "necrosis": None,  # null color → omitted from the overlay
        },
        tissue_percentage={"background": None, "tumor": 0.1, "stroma": 0.1, "necrosis": 0.1},
        active_annotations=("tumor", "stroma", "necrosis"),
    )


def _patch_mask_preview_renderer(monkeypatch):
    """Record every annotation mask-preview render and stub the file write so the structural
    tests need no real WSI. Returns the list of recorded render calls."""
    calls: list[dict] = []

    def _fake_save_overlay_preview(*, mask_preview_path, **kwargs):
        calls.append({"mask_preview_path": Path(mask_preview_path), **kwargs})
        Path(mask_preview_path).parent.mkdir(parents=True, exist_ok=True)
        Path(mask_preview_path).write_bytes(b"preview")

    monkeypatch.setattr(singlemod, "save_overlay_preview", _fake_save_overlay_preview)
    return calls


@pytest.mark.parametrize(
    "output_mode",
    [CoordinateOutputMode.PER_ANNOTATION, CoordinateOutputMode.MERGED],
)
def test_tile_slides_sampling_writes_one_mask_preview_per_slide(
    monkeypatch, tmp_path, output_mode
):
    from hs2p.configs import PreviewConfig

    _patch_tile_slides_open(monkeypatch)
    calls = _patch_mask_preview_renderer(monkeypatch)
    tile_slides(
        _slides(2),
        tiling=_mock_tiling(),
        filtering=FilterConfig(a_t=0),
        preview=PreviewConfig(save_mask_preview=True),
        output_dir=tmp_path,
        num_workers=1,
        sampling=_sampling_spec_with_colors(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        output_mode=output_mode,
    )
    # One preview file per slide at the conventional flat location.
    for sample_id in ("slide0", "slide1"):
        assert (tmp_path / "preview" / "mask" / f"{sample_id}.jpg").is_file()
    # Rendered once per slide, not once per active annotation.
    assert len(calls) == 2
    # Filled, alpha-blended, per-label colored overlay from the resolved binaries.
    for call in calls:
        assert call["color_mapping"]["tumor"] == [255, 0, 0]
        assert call["color_mapping"]["necrosis"] is None  # null-color label omitted
        assert call["alpha"] == PreviewConfig().mask_overlay_alpha
        assert call["pixel_mapping"] == PIXEL_MAPPING

    rows = pd.read_csv(tmp_path / "process_list.csv")
    expected = str(tmp_path / "preview" / "mask" / "slide0.jpg")
    slide0_rows = rows[rows["sample_id"] == "slide0"]
    assert len(slide0_rows) >= 1
    # mask_preview_path repeated on every label row of the slide.
    assert (slide0_rows["mask_preview_path"] == expected).all()


def test_tile_slides_sampling_without_preview_writes_no_mask_preview(monkeypatch, tmp_path):
    _patch_tile_slides_open(monkeypatch)
    calls = _patch_mask_preview_renderer(monkeypatch)
    tile_slides(
        _slides(1),
        tiling=_mock_tiling(),
        filtering=FilterConfig(a_t=0),
        output_dir=tmp_path,
        num_workers=1,
        sampling=_sampling_spec_with_colors(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
    )
    assert calls == []
    assert not (tmp_path / "preview" / "mask").exists()


def _patch_tiling_preview_renderer(monkeypatch):
    """Record every tiling-preview render and stub the file write so the structural tests need
    no real WSI. Returns the list of recorded render calls (one per non-empty tile set)."""
    calls: list[dict] = []

    def _fake_write_coordinate_preview(**kwargs):
        calls.append(dict(kwargs))
        save_dir = Path(kwargs["save_dir"])
        annotation = kwargs.get("annotation")
        if annotation is not None:
            save_dir = save_dir / annotation
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{kwargs['sample_id']}.jpg").write_bytes(b"preview")

    monkeypatch.setattr(orchmod, "write_coordinate_preview", _fake_write_coordinate_preview)
    return calls


def _zero_one_label_sampling_spec():
    """Only ``tumor`` can sample tiles; ``stroma``/``necrosis`` thresholds are impossibly high
    so they sample zero tiles."""
    return SamplingSpec(
        pixel_mapping=PIXEL_MAPPING,
        color_mapping={
            "background": None,
            "tumor": [255, 0, 0],
            "stroma": [0, 255, 0],
            "necrosis": [0, 0, 255],
        },
        tissue_percentage={"background": None, "tumor": 0.1, "stroma": 2.0, "necrosis": 2.0},
        active_annotations=("tumor", "stroma", "necrosis"),
    )


@pytest.mark.parametrize(
    "strategy",
    [
        CoordinateSelectionStrategy.JOINT_SAMPLING,
        CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
    ],
)
def test_tile_slides_per_annotation_writes_one_tiling_preview_per_label(
    monkeypatch, tmp_path, strategy
):
    """PER_ANNOTATION: one tiling preview per active label, each under that label's
    per-annotation subdir, with its tiling_preview_path recorded on the matching row.
    Identical under INDEPENDENT and JOINT selection."""
    from hs2p.configs import PreviewConfig

    _patch_tile_slides_open(monkeypatch)
    _patch_mask_preview_renderer(monkeypatch)
    calls = _patch_tiling_preview_renderer(monkeypatch)
    tile_slides(
        _slides(1),
        tiling=_mock_tiling(),
        filtering=FilterConfig(a_t=0),
        preview=PreviewConfig(save_tiling_preview=True),
        output_dir=tmp_path,
        num_workers=1,
        sampling=_sampling_spec_with_colors(),
        selection_strategy=strategy,
        output_mode=CoordinateOutputMode.PER_ANNOTATION,
    )
    tiling_root = tmp_path / "preview" / "tiling"
    for annotation in ("tumor", "stroma"):
        assert (tiling_root / annotation / "slide0.jpg").is_file()
    # One render per active label that sampled tiles; each carries its own annotation subdir.
    rendered_annotations = {c.get("annotation") for c in calls}
    assert rendered_annotations == {"tumor", "stroma", "necrosis"}

    rows = pd.read_csv(tmp_path / "process_list.csv")
    for annotation in ("tumor", "stroma"):
        row = rows[rows["annotation"] == annotation].iloc[0]
        expected = str(tiling_root / annotation / "slide0.jpg")
        assert row["tiling_preview_path"] == expected


@pytest.mark.parametrize(
    "strategy",
    [
        CoordinateSelectionStrategy.JOINT_SAMPLING,
        CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
    ],
)
def test_tile_slides_merged_writes_one_merged_tiling_preview(
    monkeypatch, tmp_path, strategy
):
    """MERGED: a single merged tiling preview at the flat preview location (no
    per-annotation subdir), with its tiling_preview_path recorded on the merged row."""
    from hs2p.configs import PreviewConfig

    _patch_tile_slides_open(monkeypatch)
    _patch_mask_preview_renderer(monkeypatch)
    calls = _patch_tiling_preview_renderer(monkeypatch)
    tile_slides(
        _slides(1),
        tiling=_mock_tiling(),
        filtering=FilterConfig(a_t=0),
        preview=PreviewConfig(save_tiling_preview=True),
        output_dir=tmp_path,
        num_workers=1,
        sampling=_sampling_spec_with_colors(),
        selection_strategy=strategy,
        output_mode=CoordinateOutputMode.MERGED,
    )
    flat = tmp_path / "preview" / "tiling" / "slide0.jpg"
    assert flat.is_file()
    # Exactly one merged render at the flat root (annotation None → no subdir).
    assert len(calls) == 1
    assert calls[0].get("annotation") is None

    rows = pd.read_csv(tmp_path / "process_list.csv")
    assert len(rows) == 1
    assert rows.iloc[0]["tiling_preview_path"] == str(flat)


def test_tile_slides_per_annotation_zero_tile_label_skips_tiling_preview(
    monkeypatch, tmp_path
):
    """A label that sampled zero tiles writes no tiling preview, but still has a manifest row
    with num_tiles=0, empty tiling_preview_path, and a populated mask_preview_path."""
    from hs2p.configs import PreviewConfig

    _patch_tile_slides_open(monkeypatch)
    _patch_mask_preview_renderer(monkeypatch)
    _patch_tiling_preview_renderer(monkeypatch)
    tile_slides(
        _slides(1),
        tiling=_mock_tiling(),
        filtering=FilterConfig(a_t=0),
        preview=PreviewConfig(save_mask_preview=True, save_tiling_preview=True),
        output_dir=tmp_path,
        num_workers=1,
        sampling=_zero_one_label_sampling_spec(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        output_mode=CoordinateOutputMode.PER_ANNOTATION,
    )
    tiling_root = tmp_path / "preview" / "tiling"
    assert (tiling_root / "tumor" / "slide0.jpg").is_file()
    assert not (tiling_root / "stroma").exists()
    assert not (tiling_root / "necrosis").exists()

    rows = pd.read_csv(tmp_path / "process_list.csv")
    mask_expected = str(tmp_path / "preview" / "mask" / "slide0.jpg")
    for annotation in ("stroma", "necrosis"):
        row = rows[rows["annotation"] == annotation].iloc[0]
        assert int(row["num_tiles"]) == 0
        assert pd.isna(row["tiling_preview_path"])
        assert row["mask_preview_path"] == mask_expected
    tumor_row = rows[rows["annotation"] == "tumor"].iloc[0]
    assert tumor_row["tiling_preview_path"] == str(tiling_root / "tumor" / "slide0.jpg")


def test_tile_slides_sampling_without_tiling_preview_writes_none(monkeypatch, tmp_path):
    _patch_tile_slides_open(monkeypatch)
    calls = _patch_tiling_preview_renderer(monkeypatch)
    tile_slides(
        _slides(1),
        tiling=_mock_tiling(),
        filtering=FilterConfig(a_t=0),
        output_dir=tmp_path,
        num_workers=1,
        sampling=_sampling_spec_with_colors(),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        output_mode=CoordinateOutputMode.PER_ANNOTATION,
    )
    assert calls == []
    assert not (tmp_path / "preview" / "tiling").exists()
    rows = pd.read_csv(tmp_path / "process_list.csv")
    assert rows["tiling_preview_path"].isna().all()


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
