"""End-to-end provenance, persistence, resume, and progress coverage for the independent
mask backend (#163)."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.tiling.orchestration as orchestration_mod
import hs2p.wsi.wsi as wsi_mod
import hs2p.wsi.visualization as vis_mod
from hs2p.api import (
    CompatibilitySpec,
    FilterConfig,
    SegmentationConfig,
    SlideSpec,
    TilingArtifacts,
    TilingConfig,
    load_tiling_result,
    save_tiling_result,
    validate_tiling_artifacts,
)
from hs2p.artifacts import load_whole_slides_from_rows
from hs2p.tiling.result import TileGeometry, TilingResult
from hs2p.wsi.backend import BackendSelection, ResolvedBackends
from tests.test_progress import RecordingReporter


def _tiles() -> TileGeometry:
    return TileGeometry(
        x=np.array([10, 30], dtype=np.int64),
        y=np.array([20, 40], dtype=np.int64),
        tissue_fractions=np.array([0.3, 0.7], dtype=np.float32),
        tile_index=np.array([0, 1], dtype=np.int32),
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        read_level=0,
        read_tile_size_px=224,
        read_spacing_um=0.5,
        tile_size_lv0=224,
        is_within_tolerance=True,
        base_spacing_um=0.25,
        slide_dimensions=[1000, 1200],
        level_downsamples=[1.0],
        overlap=0.0,
        min_tissue_fraction=0.1,
    )


def _result(**overrides) -> TilingResult:
    params = dict(
        tiles=_tiles(),
        sample_id="slide-1",
        image_path="slide-1.svs",
        backend="cucim",
        requested_backend="auto",
        mask_backend="openslide",
        requested_mask_backend="auto",
        tolerance=0.07,
        step_px_lv0=224,
        tissue_method="precomputed_mask",
        requested_seg_downsample=64,
        seg_downsample=64,
        seg_level=0,
        seg_spacing_um=0.5,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=224,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
        mask_path="slide-1-mask.tif",
    )
    params.update(overrides)
    return TilingResult(**params)


def test_metadata_round_trips_all_four_backends_without_conflation(tmp_path: Path):
    result = _result()
    artifacts = save_tiling_result(result, output_dir=tmp_path)
    loaded = load_tiling_result(
        artifacts.coordinates_npz_path, artifacts.coordinates_meta_path
    )
    # requested and resolved, slide and mask, all distinct and preserved separately.
    assert loaded.requested_backend == "auto"
    assert loaded.backend == "cucim"
    assert loaded.requested_mask_backend == "auto"
    assert loaded.mask_backend == "openslide"
    # artifacts carry the mask provenance too
    assert artifacts.mask_backend == "openslide"
    assert artifacts.requested_mask_backend == "auto"


def test_maskless_result_persists_null_mask_provenance(tmp_path: Path):
    result = _result(
        tissue_method="hsv",
        mask_path=None,
        mask_backend=None,
        requested_mask_backend=None,
    )
    artifacts = save_tiling_result(result, output_dir=tmp_path)
    loaded = load_tiling_result(
        artifacts.coordinates_npz_path, artifacts.coordinates_meta_path
    )
    assert loaded.mask_backend is None
    assert loaded.requested_mask_backend is None
    assert artifacts.mask_backend is None
    assert artifacts.requested_mask_backend is None


def test_success_process_row_records_mask_backends_symmetrically():
    artifact = TilingArtifacts(
        sample_id="slide-1",
        coordinates_npz_path=Path("tiles/slide-1.coordinates.npz"),
        coordinates_meta_path=Path("tiles/slide-1.coordinates.meta.json"),
        num_tiles=2,
        backend="cucim",
        requested_backend="auto",
        mask_backend="openslide",
        requested_mask_backend="auto",
    )
    row = orchestration_mod._build_success_process_row(
        whole_slide=SlideSpec(
            sample_id="slide-1",
            image_path=Path("slide-1.svs"),
            mask_path=Path("slide-1-mask.tif"),
        ),
        artifact=artifact,
    )
    assert row["requested_backend"] == "auto"
    assert row["backend"] == "cucim"
    assert row["requested_mask_backend"] == "auto"
    assert row["mask_backend"] == "openslide"


def test_failure_process_row_records_mask_backends_when_known():
    row = orchestration_mod._build_failure_process_row(
        whole_slide=SlideSpec(
            sample_id="slide-1",
            image_path=Path("slide-1.svs"),
            mask_path=Path("slide-1-mask.tif"),
        ),
        error="boom",
        traceback_text="tb",
        requested_backend="auto",
        backend="cucim",
        requested_mask_backend="auto",
        mask_backend="openslide",
    )
    assert row["requested_mask_backend"] == "auto"
    assert row["mask_backend"] == "openslide"


def test_resume_rejects_on_resolved_mask_backend_mismatch(tmp_path: Path):
    result = _result()
    artifacts = save_tiling_result(result, output_dir=tmp_path)
    whole_slide = SlideSpec(
        sample_id="slide-1",
        image_path=Path("slide-1.svs"),
        mask_path=Path("slide-1-mask.tif"),
    )
    seg = SegmentationConfig(method="precomputed_mask", downsample=64, sthresh=8, sthresh_up=255, mthresh=7, close=4)
    filt = FilterConfig(ref_tile_size=224, a_t=4, a_h=2, filter_white=False, filter_black=False, white_threshold=220, black_threshold=25, fraction_threshold=0.9)
    tiling = TilingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=224, tolerance=0.07, overlap=0.0,
        min_coverage={"tissue": 0.1}, backend="cucim", mask_backend="auto",
    )
    # Compatible: resolved mask backend matches (openslide), requested differs (auto vs asap).
    compatible = CompatibilitySpec(
        tiling=tiling, segmentation=seg, filtering=filt, mask_backend="openslide",
    )
    ok = validate_tiling_artifacts(
        whole_slide=whole_slide,
        coordinates_npz_path=artifacts.coordinates_npz_path,
        coordinates_meta_path=artifacts.coordinates_meta_path,
        compatibility=compatible,
    )
    assert ok.mask_backend == "openslide"
    # Incompatible: resolved mask backend differs.
    incompatible = replace(compatible, mask_backend="asap")
    with pytest.raises(ValueError, match="mask_backend mismatch"):
        validate_tiling_artifacts(
            whole_slide=whole_slide,
            coordinates_npz_path=artifacts.coordinates_npz_path,
            coordinates_meta_path=artifacts.coordinates_meta_path,
            compatibility=incompatible,
        )


def test_resume_ignores_mask_backend_for_maskless_slide(tmp_path: Path):
    result = _result(tissue_method="hsv", mask_path=None, mask_backend=None, requested_mask_backend=None)
    artifacts = save_tiling_result(result, output_dir=tmp_path)
    whole_slide = SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs"))
    seg = SegmentationConfig(method="hsv", downsample=64, sthresh=8, sthresh_up=255, mthresh=7, close=4)
    filt = FilterConfig(ref_tile_size=224, a_t=4, a_h=2, filter_white=False, filter_black=False, white_threshold=220, black_threshold=25, fraction_threshold=0.9)
    tiling = TilingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=224, tolerance=0.07, overlap=0.0,
        min_coverage={"tissue": 0.1}, backend="cucim",
    )
    # A mismatched compatibility.mask_backend must not reject a maskless slide.
    compatible = CompatibilitySpec(
        tiling=tiling, segmentation=seg, filtering=filt, mask_backend="asap",
    )
    ok = validate_tiling_artifacts(
        whole_slide=whole_slide,
        coordinates_npz_path=artifacts.coordinates_npz_path,
        coordinates_meta_path=artifacts.coordinates_meta_path,
        compatibility=compatible,
    )
    assert ok.mask_backend is None


def test_resume_rejects_pre_163_process_list_missing_mask_columns(
    tmp_path: Path,
):
    import pandas as pd

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # A pre-#163 process_list.csv has slide backend columns but no mask backend columns.
    pd.DataFrame(
        [
            {
                "sample_id": "slide-1",
                "annotation": "tissue",
                "image_path": "slide-1.svs",
                "mask_path": np.nan,
                "requested_backend": "asap",
                "backend": "asap",
                "tiling_status": "success",
                "num_tiles": 1,
                "coordinates_npz_path": "x.npz",
                "coordinates_meta_path": "x.meta.json",
                "error": np.nan,
                "traceback": np.nan,
            }
        ]
    ).to_csv(run_dir / "process_list.csv", index=False)
    seg = SegmentationConfig(method="hsv", downsample=64, sthresh=8, sthresh_up=255, mthresh=7, close=4)
    filt = FilterConfig(ref_tile_size=224, a_t=4, a_h=2, filter_white=False, filter_black=False, white_threshold=220, black_threshold=25, fraction_threshold=0.9)
    tiling = TilingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=224, tolerance=0.07, overlap=0.0,
        min_coverage={"tissue": 0.1}, backend="asap",
    )
    with pytest.raises(ValueError, match="missing required columns:.*mask_backend"):
        orchestration_mod.tile_slides(
            [SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs"))],
            tiling=tiling,
            segmentation=seg,
            filtering=filt,
            output_dir=run_dir,
            resume=True,
        )


def test_load_whole_slides_from_rows_preserves_mask_path():
    slides = load_whole_slides_from_rows(
        [{"sample_id": "s1", "image_path": "s1.svs", "mask_path": "s1-mask.tif"}]
    )
    assert slides[0].mask_path == Path("s1-mask.tif")


def test_auto_mask_selection_emits_distinct_progress_event(monkeypatch):
    reporter = RecordingReporter()

    def _fake_resolve_backends(*, requested_slide_backend, requested_mask_backend, wsi_path, mask_path=None):
        slide = BackendSelection(backend="asap", reason=None, tried=("asap",))
        mask = BackendSelection(
            backend="cucim", reason="selected cuCIM for auto backend", tried=("cucim",)
        )
        return ResolvedBackends(
            slide=slide, mask=mask,
            requested_slide_backend="asap", requested_mask_backend="auto",
        )

    monkeypatch.setattr(orchestration_mod, "resolve_backends", _fake_resolve_backends)
    tiling = TilingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=224, tolerance=0.07, overlap=0.0,
        min_coverage={"tissue": 0.1}, backend="asap", mask_backend="auto",
    )
    whole_slide = SlideSpec(
        sample_id="slide-9", image_path=Path("slide-9.svs"), mask_path=Path("slide-9-mask.tif")
    )
    import hs2p.progress as progress

    with progress.activate_progress_reporter(reporter):
        effective = orchestration_mod._resolve_effective_backends(whole_slide, tiling)

    kinds = [e.kind for e in reporter.events]
    assert "mask_backend.selected" in kinds
    event = next(e for e in reporter.events if e.kind == "mask_backend.selected")
    assert event.payload["sample_id"] == "slide-9"
    assert event.payload["backend"] == "cucim"
    assert event.payload["mask_path"] == str(whole_slide.mask_path)
    assert "cuCIM" in event.payload["reason"]
    # slide backend resolved independently and folded into the effective config
    assert effective.backend == "asap"
    assert effective.mask_backend == "cucim"
    assert effective.requested_mask_backend == "auto"


def test_maskless_slide_emits_no_mask_backend_event(monkeypatch):
    reporter = RecordingReporter()

    def _fake_resolve_backends(*, requested_slide_backend, requested_mask_backend, wsi_path, mask_path=None):
        return ResolvedBackends(
            slide=BackendSelection(backend="cucim", reason="selected cuCIM for auto backend", tried=("cucim",)),
            mask=None, requested_slide_backend="auto", requested_mask_backend=None,
        )

    monkeypatch.setattr(orchestration_mod, "resolve_backends", _fake_resolve_backends)
    tiling = TilingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=224, tolerance=0.07, overlap=0.0,
        min_coverage={"tissue": 0.1}, backend="auto", mask_backend="auto",
    )
    whole_slide = SlideSpec(sample_id="slide-x", image_path=Path("slide-x.svs"))
    import hs2p.progress as progress

    with progress.activate_progress_reporter(reporter):
        orchestration_mod._resolve_effective_backends(whole_slide, tiling)

    kinds = [e.kind for e in reporter.events]
    assert "mask_backend.selected" not in kinds
    assert "backend.selected" in kinds


def test_wsi_resolves_mask_backend_independently(monkeypatch):
    opened: list[tuple[str, str]] = []

    def _fake_resolve_backend(requested, *, wsi_path, mask_path=None):
        # slide path -> asap; mask path -> openslide (independent)
        backend = "openslide" if "mask" in str(wsi_path) else "asap"
        return BackendSelection(backend=backend, reason=None, tried=(backend,))

    class _Reader:
        backend_name = "asap"
        spacings = [0.5]
        level_dimensions = [(10, 10)]
        level_downsamples = [(1.0, 1.0)]

        def close(self):
            return None

    def _fake_open_slide(path, backend, *, spacing_override=None, gpu_decode=False):
        opened.append((str(path), backend))
        return _Reader()

    monkeypatch.setattr(wsi_mod, "resolve_backend", _fake_resolve_backend)
    monkeypatch.setattr(wsi_mod, "open_slide", _fake_open_slide)
    # The attached mask now opens through the centralized ``open_mask_reader`` helper (#163),
    # which resolves + opens via the reader module's own globals — patch those too.
    import hs2p.wsi.reader as reader_mod

    monkeypatch.setattr(reader_mod, "resolve_backend", _fake_resolve_backend)
    monkeypatch.setattr(reader_mod, "open_slide", _fake_open_slide)

    wsi = wsi_mod.WSI(
        path=Path("/data/slide.svs"),
        backend="auto",
        mask_path=Path("/data/slide-mask.tif"),
        mask_backend="auto",
    )
    assert wsi.backend == "asap"
    assert wsi.mask_backend == "openslide"
    assert wsi.requested_mask_backend == "auto"
    # the mask reader was opened with the mask's own resolved backend
    assert ("/data/slide-mask.tif", "openslide") in opened
    assert ("/data/slide.svs", "asap") in opened


def test_overlay_mask_on_slide_reads_mask_with_its_own_backend(monkeypatch):
    backends: list[str] = []

    class _FakeWSI:
        def __init__(self, path, backend, mask_path=None, spacing_at_level_0=None, mask_backend="auto"):
            backends.append(backend)
            self.reader = SimpleNamespace(
                read_level=lambda level: np.zeros((10, 10, 3), np.uint8),
                spacings=[0.5],
                level_downsamples=[(1.0, 1.0)],
            )
            self.level_downsamples = [(1.0, 1.0)]

        def get_best_level_for_downsample_custom(self, downsample):
            return 0

        def get_slide(self, level):
            return np.zeros((10, 10, 3), np.uint8)

        def get_level_spacing(self, level):
            return 0.5

    def _fake_open_mask_reader(mask_path, *, mask_backend="auto"):
        # The mask opens through the centralized helper with its own resolved backend (#163).
        backends.append(mask_backend)
        reader = SimpleNamespace(
            read_level=lambda level: np.zeros((10, 10, 3), np.uint8),
            spacings=[0.5],
            level_downsamples=[(1.0, 1.0)],
        )
        return reader, mask_backend

    monkeypatch.setattr(vis_mod, "WSI", _FakeWSI)
    monkeypatch.setattr(vis_mod, "open_mask_reader", _fake_open_mask_reader)
    monkeypatch.setattr(
        vis_mod, "read_aligned_mask", lambda **kwargs: np.zeros((10, 10), np.uint8)
    )
    vis_mod.overlay_mask_on_slide(
        wsi_path=Path("/data/slide.svs"),
        annotation_mask_path=Path("/data/mask.tif"),
        downsample=32,
        backend="asap",
        mask_backend="openslide",
    )
    # first the slide WSI (asap), then the mask via open_mask_reader (openslide)
    assert backends == ["asap", "openslide"]
