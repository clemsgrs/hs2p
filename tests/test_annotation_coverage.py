"""Tests for the annotation-mask producer (resolve_annotation_masks) and the per-class
coverage summary (summarize_annotation_coverage) — the previously-missing keystone for
build_per_annotation_tiling_results, plus the coverage utility soma's ingestion drives."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.mask as maskmod
import hs2p.tiling.orchestration as orchmod
import hs2p.tiling.single as singlemod
import pandas as pd

from hs2p.api import FilterConfig, SlideSpec, TilingConfig, tile_slides
from hs2p.mask import AnnotationLabels, Mask
from hs2p.tiling.coverage import summarize_annotation_coverage
from hs2p.tiling.mask import resolve_annotation_masks
from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy, SamplingSpec

BASE_SPACING = 0.5
SLIDE_W, SLIDE_H = 400, 400
PIXEL_MAPPING = {"background": 0, "tumor": 1, "stroma": 2, "necrosis": 3}

# Areas (in level-0 == seg pixels here, downsample 1):
#   tumor    200x200 = 40000
#   stroma   200x200 = 40000
#   necrosis  40x40  =  1600   (a small focus in the top-right corner)
TUMOR_PX, STROMA_PX, NECROSIS_PX = 40000, 40000, 1600


def _fake_resolve_backends(
    *,
    requested_slide_backend,
    requested_mask_backend,
    wsi_path,
    mask_path=None,
    slide_spacing_override=None,
):
    del slide_spacing_override
    from hs2p.wsi.backend import BackendSelection, ResolvedBackends

    sel = BackendSelection(backend="asap", reason=None, tried=("asap",))
    return ResolvedBackends(
        slide=sel,
        mask=None if mask_path is None else sel,
        requested_slide_backend=requested_slide_backend,
        requested_mask_backend=None if mask_path is None else requested_mask_backend,
    )


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
        self.native_spacing = spacing
        height, width = mask.shape
        self.level_dimensions = [(width, height)]
        self.level_downsamples = [(1.0, 1.0)]

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


def _open_annotation_mask(
    monkeypatch, native: np.ndarray, *, pixel_mapping=PIXEL_MAPPING
) -> Mask:
    """Open an annotation ``Mask`` whose ``mock`` backend decodes to ``native``."""
    monkeypatch.setattr(
        maskmod,
        "open_slide",
        lambda path, backend=None, **kwargs: _FakeMaskSlide(native, BASE_SPACING),
    )
    return Mask(
        path="/fake/slide_mask.tif",
        labels=AnnotationLabels(pixel_mapping=pixel_mapping),
        backend="mock",
    )


def test_resolve_annotation_masks_splits_per_declared_label(monkeypatch):
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask=_open_annotation_mask(monkeypatch, _label_mask()),
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
    assert resolved.mask_path == Path("/fake/slide_mask.tif")
    assert (resolved.mask_level, resolved.mask_spacing_um) == (0, BASE_SPACING)


def test_resolve_annotation_masks_accepts_uint16_storage_for_preview_safe_labels(
    monkeypatch,
):
    native = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint16)
    native[0:200, 0:200] = 254
    native[200:400, 200:400] = 255

    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask=_open_annotation_mask(
            monkeypatch,
            native,
            pixel_mapping={"background": 0, "tumor": 254, "stroma": 255},
        ),
        seg_downsample=1,
    )
    assert int(np.count_nonzero(resolved.masks["tumor"])) == 200 * 200
    assert int(np.count_nonzero(resolved.masks["stroma"])) == 200 * 200


def test_resolve_annotation_masks_background_is_optional(monkeypatch):
    """A raster that labels every pixel (no reserved background value) needs no 'background'
    entry — every declared class still validates and is split into a binary."""
    native = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint8)
    native[:, : SLIDE_W // 2] = 1  # tumor over the left half
    native[:, SLIDE_W // 2 :] = 2  # stroma over the right half

    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask=_open_annotation_mask(
            monkeypatch, native, pixel_mapping={"tumor": 1, "stroma": 2}
        ),
        seg_downsample=1,
    )
    assert set(resolved.masks) == {"tumor", "stroma"}
    assert int(np.count_nonzero(resolved.masks["tumor"])) == SLIDE_H * (SLIDE_W // 2)
    assert int(np.count_nonzero(resolved.masks["stroma"])) == SLIDE_H * (SLIDE_W // 2)


def test_summarize_annotation_coverage_area_frac_and_est_tiles(monkeypatch):
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask=_open_annotation_mask(monkeypatch, _label_mask()),
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


def _slide_at(spacing_um: float) -> SimpleNamespace:
    return SimpleNamespace(
        dimensions=(SLIDE_W, SLIDE_H),
        spacing=spacing_um,
        level_downsamples=[1.0],
        level_dimensions=[(SLIDE_W, SLIDE_H)],
        backend_name="mock",
    )


def _left_column_tumor_est_tiles(monkeypatch, *, spacing_um: float, **kwargs) -> int:
    """``est_tiles`` for tumor filling the left 200 px column of a 400x400 slide at
    ``spacing_um``, with 200 px tiles requested at 0.5 um/px and a 0.99 threshold."""
    native = np.zeros((SLIDE_H, SLIDE_W), dtype=np.uint8)
    native[:, :200] = 1
    slide = _slide_at(spacing_um)
    resolved = resolve_annotation_masks(
        slide=slide,
        mask=_open_annotation_mask(monkeypatch, native, pixel_mapping={"background": 0, "tumor": 1}),
        seg_downsample=1,
    )
    summary = summarize_annotation_coverage(
        slide=slide,
        resolved_masks=resolved,
        min_coverage={"tumor": 0.99},
        requested_tile_size_px=200,
        requested_spacing_um=BASE_SPACING,
        **kwargs,
    )
    return summary["tumor"]["est_tiles"]


def test_est_tiles_uses_the_tiling_footprint_within_tolerance(monkeypatch):
    """#226: a level within tolerance is read natively, so tiling's footprint on a
    0.485 um/px slide is 200 level-0 px, not round(200 * 0.5 / 0.485) = 206. With the
    tiling footprint both left-column tiles are pure tumor; with 206 px they are 97%."""
    assert _left_column_tumor_est_tiles(monkeypatch, spacing_um=0.485) == 2


def test_est_tiles_keeps_the_requested_footprint_outside_tolerance(monkeypatch):
    # 0.485 is 3% off 0.5: outside a 1% tolerance the read is resized, and the
    # footprint stays round(200 * 0.5 / 0.485) = 206, so the column tiles are 97% tumor.
    assert _left_column_tumor_est_tiles(monkeypatch, spacing_um=0.485, tolerance=0.01) == 0


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


def _patch_tile_slides_open(monkeypatch):
    mask = _label_mask()

    def fake_open(path, backend="auto", spacing_override=None, **kwargs):
        del spacing_override
        if "mask" in str(path).lower():
            return _FakeMaskSlide(mask, BASE_SPACING)
        return _FakeSlide()

    monkeypatch.setattr(singlemod, "open_slide", fake_open)
    monkeypatch.setattr(maskmod, "open_slide", fake_open)
    monkeypatch.setattr(orchmod, "resolve_backends", _fake_resolve_backends)


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
        backend="asap",
    )


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
    assert all(
        artifact.coordinates_meta_path.parent == tmp_path / "tiles"
        for artifact in artifacts
    )
    assert not (tmp_path / "tiles" / "merged").exists()

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


@pytest.mark.parametrize("invalid_value", [-1, 256])
def test_tile_slides_rejects_invalid_label_id_before_slide_io_or_output(
    monkeypatch, tmp_path, invalid_value
):
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        orchmod,
        "resolve_backends",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("slide backend resolution must not run")
        ),
    )
    sampling = SamplingSpec(
        pixel_mapping={"background": 0, "tumor": invalid_value},
        color_mapping=None,
        tissue_percentage={"background": None, "tumor": 0.1},
        active_annotations=("tumor",),
    )

    with pytest.raises(ValueError, match=rf"tumor.*{invalid_value}"):
        tile_slides(
            _slides(1),
            tiling=_mock_tiling(),
            filtering=FilterConfig(a_t=0),
            output_dir=output_dir,
            sampling=sampling,
        )

    assert not output_dir.exists()


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


def test_summarize_annotation_coverage_est_tiles_none_without_threshold(monkeypatch):
    resolved = resolve_annotation_masks(
        slide=_mock_slide(),
        mask=_open_annotation_mask(monkeypatch, _label_mask()),
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
