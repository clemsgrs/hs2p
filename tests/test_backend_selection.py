from pathlib import Path
from types import SimpleNamespace

import numpy as np

import hs2p.api as api_mod
import hs2p.preprocessing as preprocessing_mod
import hs2p.wsi.backend as backend_mod
import hs2p.wsi.reader as reader_mod
import hs2p.wsi.wsi as wsi_mod


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
    assert "CuCIM" in (selection.reason or "")
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
    assert "CuCIM" in (selection.reason or "")
    assert calls == ["cucim"]


def test_reader_backend_probe_uses_backend_openers(monkeypatch):
    seen_paths: list[str] = []

    def _fake_opener(path, *, spacing_override=None, gpu_decode=False):
        del spacing_override, gpu_decode
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



def test_wholeslideimage_coerces_cucim_paths_to_strings(monkeypatch):
    seen_paths: list[tuple[object, str]] = []

    class _FakeSlide:
        spacings = [0.5]
        shapes = [(100, 100)]

    def _fake_wholeslideimage(path, *, backend: str):
        seen_paths.append((path, backend))
        return _FakeSlide()

    monkeypatch.setattr(
        wsi_mod,
        "resolve_backend",
        lambda requested_backend, *, wsi_path, mask_path=None: backend_mod.BackendSelection(
            backend="cucim",
            tried=("cucim",),
        ),
    )
    monkeypatch.setattr(wsi_mod.wsd, "WholeSlideImage", _fake_wholeslideimage)
    monkeypatch.setattr(wsi_mod.WholeSlideImage, "load_segmentation", lambda *args, **kwargs: 0)

    wsi_mod.WholeSlideImage(
        path=Path("/tmp/slide.tiff"),
        mask_path=Path("/tmp/mask.tiff"),
        backend="auto",
        sampling_spec=wsi_mod.ResolvedSamplingSpec(
            pixel_mapping={"background": 0, "tumor": 1},
            color_mapping={"background": None, "tumor": None},
            tissue_percentage={"background": None, "tumor": 0.1},
            active_annotations=("tumor",),
        ),
        segment_params=SimpleNamespace(),
    )

    assert seen_paths == [
        ("/tmp/slide.tiff", "cucim"),
        ("/tmp/mask.tiff", "cucim"),
    ]


def test_tile_slide_uses_resolved_backend_for_hash_and_result(monkeypatch):
    captured: dict[str, str] = {}

    def _fake_resolve_backend(requested_backend: str, *, wsi_path: Path, mask_path):
        del requested_backend, wsi_path, mask_path
        return backend_mod.BackendSelection(
            backend="cucim",
            reason="selected CuCIM for auto backend",
            tried=("cucim",),
        )

    def _fake_preprocess_slide(**kwargs):
        captured["backend"] = kwargs["backend"]
        return preprocessing_mod.TilingResult(
            tiles=preprocessing_mod.TileGeometry(
                coordinates=np.array([[0, 0]], dtype=np.int64),
                tissue_fractions=np.array([0.0], dtype=np.float32),
                tile_index=np.array([0], dtype=np.int32),
                requested_tile_size_px=256,
                requested_spacing_um=0.5,
                read_level=0,
                effective_tile_size_px=256,
                effective_spacing_um=0.5,
                tile_size_lv0=256,
                is_within_tolerance=True,
                base_spacing_um=0.5,
                slide_dimensions=[1000, 1000],
                level_downsamples=[1.0],
                overlap=0.0,
                min_tissue_fraction=0.1,
                use_padding=True,
            ),
            sample_id="slide-1",
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

    monkeypatch.setattr(api_mod, "resolve_backend", _fake_resolve_backend)
    monkeypatch.setattr(api_mod, "preprocess_slide", _fake_preprocess_slide)

    result = api_mod.tile_slide(
        api_mod.SlideSpec(sample_id="slide-1", image_path=Path("slide.svs")),
        tiling=api_mod.TilingConfig(
            target_spacing_um=0.5,
            target_tile_size_px=256,
            tolerance=0.05,
            overlap=0.0,
            tissue_threshold=0.1,
            backend="auto",
        ),
        segmentation=api_mod.SegmentationConfig(64, 8, 255, 7, 4, False, True),
        filtering=api_mod.FilterConfig(16, 4, 2, False, False, 220, 25, 0.9),
        num_workers=1,
    )

    assert result.backend == "cucim"
    assert captured["backend"] == "cucim"
