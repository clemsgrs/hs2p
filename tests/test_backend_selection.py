from pathlib import Path
from types import SimpleNamespace

import numpy as np

import hs2p.api as api_mod
import hs2p.wsi.backend as backend_mod
import hs2p.wsi.reader as reader_mod
import hs2p.wsi.wsi as wsi_mod


def test_resolve_backend_prefers_cucim_when_supported(monkeypatch):
    calls: list[str] = []

    def _fake_can_open_slide(*, wsi_path: str, mask_path: str | None, backend: str):
        del wsi_path, mask_path
        calls.append(backend)
        return backend == "cucim"

    monkeypatch.setattr(backend_mod, "_backend_can_open_slide", _fake_can_open_slide)

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

    monkeypatch.setattr(backend_mod, "_backend_can_open_slide", _fake_can_open_slide)

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

    monkeypatch.setattr(backend_mod, "_backend_can_open_slide", _fake_can_open_slide)

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


def test_backend_probe_coerces_cucim_paths_to_strings(monkeypatch):
    seen_paths: list[tuple[object, str]] = []

    def _fake_wholeslideimage(path, *, backend: str):
        seen_paths.append((path, backend))
        return SimpleNamespace()

    backend_mod._backend_can_open_slide.cache_clear()
    monkeypatch.setattr(backend_mod.wsd, "WholeSlideImage", _fake_wholeslideimage)

    assert backend_mod._backend_can_open_slide(
        wsi_path="/tmp/slide.tiff",
        mask_path="/tmp/mask.tiff",
        backend="cucim",
    )
    assert seen_paths == [
        ("/tmp/slide.tiff", "cucim"),
        ("/tmp/mask.tiff", "cucim"),
    ]


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

    def _fake_extract_coordinates(
        *,
        wsi_path: Path,
        mask_path: Path | None,
        backend: str,
        segment_params,
        tiling_params,
        filter_params,
        sampling_spec,
        mask_preview_path,
        spacing_at_level_0,
        disable_tqdm,
        num_workers,
    ):
        del (
            wsi_path,
            mask_path,
            segment_params,
            tiling_params,
            filter_params,
            sampling_spec,
            mask_preview_path,
            spacing_at_level_0,
            disable_tqdm,
            num_workers,
        )
        captured["backend"] = backend
        return SimpleNamespace(
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tissue_percentages=None,
            read_level=0,
            read_spacing_um=0.5,
            read_tile_size_px=256,
            read_step_px=256,
            tile_size_lv0=256,
            step_px_lv0=256,
        )

    def _fake_hash(*, tiling, segmentation, filtering, sampling_spec=None, selection_strategy=None, output_mode=None, annotation=None):
        del segmentation, filtering, sampling_spec, selection_strategy, output_mode, annotation
        captured["hash_backend"] = tiling.backend
        return "hash"

    monkeypatch.setattr(api_mod, "resolve_backend", _fake_resolve_backend)
    monkeypatch.setattr(api_mod, "extract_coordinates", _fake_extract_coordinates)
    monkeypatch.setattr(api_mod, "compute_effective_config_hash", _fake_hash)

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
        filtering=api_mod.FilterConfig(16, 4, 2, 8, False, False, 220, 25, 0.9),
        num_workers=1,
    )

    assert result.backend == "cucim"
    assert captured["backend"] == "cucim"
    assert captured["hash_backend"] == "cucim"
