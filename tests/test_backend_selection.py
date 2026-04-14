from pathlib import Path
from types import SimpleNamespace

import numpy as np

import hs2p.api as api_mod
import hs2p.preprocessing as preprocessing_mod
import hs2p.wsi.backend as backend_mod
import hs2p.wsi.reader as reader_mod
import hs2p.wsi.wsi as wsi_mod
from tests.test_progress import RecordingReporter


def _make_tiling_result(sample_id: str = "slide-1") -> preprocessing_mod.TilingResult:
    return preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tissue_fractions=np.array([0.0], dtype=np.float32),
            tile_index=np.array([0], dtype=np.int32),
            requested_tile_size_px=256,
            requested_spacing_um=0.5,
            read_level=0,
            read_tile_size_px=256,
            read_spacing_um=0.5,
            tile_size_lv0=256,
            is_within_tolerance=True,
            base_spacing_um=0.5,
            slide_dimensions=[1000, 1000],
            level_downsamples=[1.0],
            overlap=0.0,
            min_tissue_fraction=0.1,
        ),
        sample_id=sample_id,
        image_path=Path("slide.svs"),
        backend="cucim",
        requested_backend="cucim",
        tolerance=0.05,
        step_px_lv0=256,
        tissue_method="hsv",
        seg_downsample=64,
        seg_level=0,
        seg_spacing_um=0.5,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=16,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def _cucim_auto_backend_selection(
    requested_backend: str, *, wsi_path: Path, mask_path
) -> backend_mod.BackendSelection:
    del requested_backend, wsi_path, mask_path
    return backend_mod.BackendSelection(
        backend="cucim",
        reason="selected cuCIM for auto backend",
        tried=("cucim",),
    )


def test_resolve_backend_prefers_cucim_when_supported(monkeypatch):
    calls: list[str] = []

    def _fake_can_open_slide(*, wsi_path: str, mask_path: str | None, backend: str):
        del wsi_path, mask_path
        calls.append(backend)
        return backend == "cucim"

    monkeypatch.setattr(reader_mod, "_backend_can_open_slide", _fake_can_open_slide)

    selection = backend_mod.resolve_backend("auto", wsi_path=Path("slide.svs"))

    assert selection.backend == "cucim"
    assert selection.tried == ("cucim",)
    assert "cuCIM" in (selection.reason or "")
    assert calls == ["cucim"]


def test_resolve_backend_skips_cucim_for_known_unsupported_suffix(monkeypatch):
    calls: list[str] = []

    def _fake_can_open_slide(*, wsi_path: str, mask_path: str | None, backend: str):
        del wsi_path, mask_path
        calls.append(backend)
        return backend == "asap"

    monkeypatch.setattr(reader_mod, "_backend_can_open_slide", _fake_can_open_slide)

    selection = backend_mod.resolve_backend("auto", wsi_path=Path("slide.mrxs"))

    assert selection.backend == "asap"
    assert selection.tried == ("asap",)
    assert calls == ["asap"]


def test_resolve_backend_respects_explicit_override(monkeypatch):
    calls: list[str] = []

    def _fake_can_open_slide(*, wsi_path: str, mask_path: str | None, backend: str):
        del wsi_path, mask_path, backend
        calls.append("called")
        return False

    monkeypatch.setattr(reader_mod, "_backend_can_open_slide", _fake_can_open_slide)

    selection = backend_mod.resolve_backend("asap", wsi_path=Path("slide.svs"))

    assert selection.backend == "asap"
    assert selection.tried == ("asap",)
    assert selection.reason is None
    assert calls == []


def test_reader_resolve_backend_prefers_cucim_when_supported(monkeypatch):
    calls: list[str] = []

    def _fake_can_open_slide(*, wsi_path: str, mask_path: str | None, backend: str):
        del wsi_path, mask_path
        calls.append(backend)
        return backend == "cucim"

    monkeypatch.setattr(reader_mod, "_backend_can_open_slide", _fake_can_open_slide)

    selection = reader_mod.resolve_backend("auto", wsi_path=Path("slide.svs"))

    assert selection.backend == "cucim"
    assert selection.tried == ("cucim",)
    assert "cuCIM" in (selection.reason or "")
    assert calls == ["cucim"]


def test_reader_backend_probe_uses_backend_openers(monkeypatch):
    seen_paths: list[str] = []

    def _fake_opener(path, *, spacing_override=None):
        del spacing_override
        seen_paths.append(str(path))
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        reader_mod,
        "_BACKENDS",
        {
            **reader_mod._BACKENDS,
            "cucim": reader_mod._BackendSpec(
                name="cucim",
                opener=_fake_opener,
                supports_path=lambda path: True,
            ),
        },
    )
    reader_mod._backend_can_open_slide.cache_clear()

    assert reader_mod._backend_can_open_slide(
        wsi_path="/tmp/slide.tiff",
        mask_path="/tmp/mask.tiff",
        backend="cucim",
    )
    assert seen_paths == ["/tmp/slide.tiff", "/tmp/mask.tiff"]



def test_wsi_opens_slide_and_mask_readers_with_resolved_backend(monkeypatch):
    seen_calls: list[tuple[object, str, float | None]] = []

    class _FakeSlideReader:
        backend_name = "cucim"
        dimensions = (100, 100)
        spacing = 0.5
        spacings = [0.5]
        level_dimensions = [(100, 100)]
        level_downsamples = [(1.0, 1.0)]
        level_count = 1

        def read_level(self, level: int):
            del level
            return np.zeros((100, 100, 3), dtype=np.uint8)

        def read_region(self, location, level, size):
            del location, level
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        def get_thumbnail(self, size):
            del size
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            return None

    def _fake_open_slide(path, backend: str, *, spacing_override=None, gpu_decode=False):
        del gpu_decode
        seen_calls.append((path, backend, spacing_override))
        return _FakeSlideReader()

    monkeypatch.setattr(
        wsi_mod,
        "resolve_backend",
        lambda requested_backend, *, wsi_path, mask_path=None: backend_mod.BackendSelection(
            backend="cucim",
            tried=("cucim",),
        ),
    )
    monkeypatch.setattr(wsi_mod, "open_slide", _fake_open_slide)
    monkeypatch.setattr(wsi_mod.WSI, "load_segmentation", lambda *args, **kwargs: 0)

    wsi_mod.WSI(
        path=Path("/tmp/slide.tiff"),
        mask_path=Path("/tmp/mask.tiff"),
        backend="auto",
        sampling_spec=wsi_mod.SamplingSpec(
            pixel_mapping={"background": 0, "tumor": 1},
            color_mapping={"background": None, "tumor": None},
            tissue_percentage={"background": None, "tumor": 0.1},
            active_annotations=("tumor",),
        ),
        segment_params=SimpleNamespace(),
    )

    assert seen_calls == [
        (Path("/tmp/slide.tiff"), "cucim", None),
        (Path("/tmp/mask.tiff"), "cucim", None),
    ]


def test_tile_slide_uses_resolved_backend_for_hash_and_result(monkeypatch):
    captured: dict[str, str] = {}

    def _fake_preprocess_slide(**kwargs):
        captured["backend"] = kwargs["backend"]
        return _make_tiling_result()

    monkeypatch.setattr(api_mod, "resolve_backend", _cucim_auto_backend_selection)
    monkeypatch.setattr(api_mod, "preprocess_slide", _fake_preprocess_slide)

    result = api_mod.tile_slide(
        api_mod.SlideSpec(sample_id="slide-1", image_path=Path("slide.svs")),
        tiling=api_mod.TilingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=256,
            tolerance=0.05,
            overlap=0.0,
            tissue_threshold=0.1,
            backend="auto",
        ),
        segmentation=api_mod.SegmentationConfig(method="hsv", downsample=64, sthresh=8, sthresh_up=255, mthresh=7, close=4),
        filtering=api_mod.FilterConfig(16, 4, 2, False, False, 220, 25, 0.9),
        num_workers=1,
    )

    assert result.backend == "cucim"
    assert captured["backend"] == "cucim"


def test_tile_slide_emits_backend_selection_progress_event(monkeypatch):
    import hs2p.progress as progress

    reporter = RecordingReporter()

    monkeypatch.setattr(api_mod, "resolve_backend", _cucim_auto_backend_selection)
    monkeypatch.setattr(
        api_mod,
        "preprocess_slide",
        lambda **kwargs: _make_tiling_result(sample_id="slide-quiet"),
    )

    with progress.activate_progress_reporter(reporter):
        api_mod.tile_slide(
            api_mod.SlideSpec(sample_id="slide-quiet", image_path=Path("slide.svs")),
            tiling=api_mod.TilingConfig(
                requested_spacing_um=0.5,
                requested_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
                backend="auto",
            ),
            segmentation=api_mod.SegmentationConfig(method="hsv", downsample=64, sthresh=8, sthresh_up=255, mthresh=7, close=4),
            filtering=api_mod.FilterConfig(16, 4, 2, False, False, 220, 25, 0.9),
            num_workers=1,
        )

    assert [event.kind for event in reporter.events] == ["backend.selected"]
    assert reporter.events[0].payload == {
        "sample_id": "slide-quiet",
        "backend": "cucim",
        "reason": "selected cuCIM for auto backend",
    }
