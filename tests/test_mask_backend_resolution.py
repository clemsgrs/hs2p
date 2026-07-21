"""Independent slide/mask backend resolution seam + source-mask read threading (#163)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.tiling.mask as mask_mod
import hs2p.wsi.reader as reader_mod
from hs2p.wsi.backend import resolve_backends
from hs2p.tiling.mask import (
    load_annotation_label_mask,
    load_precomputed_tissue_mask,
    resolve_annotation_masks,
    resolve_tissue_mask,
)


class _ArrayMaskSlide:
    spacing = 0.25
    level_downsamples = [1.0]

    def __init__(self, mask: np.ndarray, *, backend_name: str = "asap"):
        self._mask = mask
        self.level_dimensions = [(mask.shape[1], mask.shape[0])]
        self.backend_name = backend_name

    def read_region(self, location, level, size):
        return self._mask

    def close(self):
        return None


def _wsi(*, backend_name: str = "cucim"):
    return SimpleNamespace(
        spacing=0.25,
        level_downsamples=[1.0],
        level_dimensions=[(2, 2)],
        backend_name=backend_name,
    )


# --- seam: independent resolution -------------------------------------------------------


def test_seam_resolves_slide_and_mask_from_own_paths(monkeypatch):
    seen: list[tuple[str, str]] = []

    def _fake_can_open(*, wsi_path, mask_path, backend):
        del mask_path
        seen.append((str(wsi_path), backend))
        # slide.svs → cucim opens; mask.tif → only asap opens
        if "mask" in str(wsi_path):
            return backend == "asap"
        return backend == "cucim"

    monkeypatch.setattr(reader_mod, "_backend_can_open_slide", _fake_can_open)

    resolved = resolve_backends(
        requested_slide_backend="auto",
        requested_mask_backend="auto",
        wsi_path=Path("slide.svs"),
        mask_path=Path("annotation-mask.tif"),
    )

    assert resolved.slide_backend == "cucim"
    assert resolved.mask_backend == "asap"
    assert resolved.requested_slide_backend == "auto"
    assert resolved.requested_mask_backend == "auto"


def test_seam_maskless_has_null_mask_provenance(monkeypatch):
    monkeypatch.setattr(
        reader_mod,
        "_backend_can_open_slide",
        lambda *, wsi_path, mask_path, backend: backend == "cucim",
    )
    resolved = resolve_backends(
        requested_slide_backend="auto",
        requested_mask_backend="auto",
        wsi_path=Path("slide.svs"),
        mask_path=None,
    )
    assert resolved.slide_backend == "cucim"
    assert resolved.mask is None
    assert resolved.mask_backend is None
    assert resolved.requested_mask_backend is None


def test_seam_slide_resolution_ignores_mask_openability(monkeypatch):
    """Slide backend must resolve from the slide path only — a mask that cannot open
    with the slide's chosen backend must not perturb slide selection."""
    probed_paths: list[str] = []

    def _fake_can_open(*, wsi_path, mask_path, backend):
        probed_paths.append(str(wsi_path))
        # cucim can open the slide; the mask is a different format only asap opens
        if "mask" in str(wsi_path):
            return backend == "asap"
        return backend == "cucim"

    monkeypatch.setattr(reader_mod, "_backend_can_open_slide", _fake_can_open)
    resolved = resolve_backends(
        requested_slide_backend="auto",
        requested_mask_backend="asap",
        wsi_path=Path("slide.svs"),
        mask_path=Path("mask.tif"),
    )
    assert resolved.slide_backend == "cucim"
    # The mask path was never probed against the slide's cucim decision.
    assert all("mask" not in p for p in probed_paths)


def test_seam_explicit_backends_are_authoritative_without_probe(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("explicit backend must not trigger an openability probe")

    monkeypatch.setattr(reader_mod, "_backend_can_open_slide", _boom)
    resolved = resolve_backends(
        requested_slide_backend="openslide",
        requested_mask_backend="cucim",
        wsi_path=Path("slide.svs"),
        mask_path=Path("mask.tif"),
    )
    assert resolved.slide_backend == "openslide"
    assert resolved.mask_backend == "cucim"


# --- explicit mask backend threads into every read path ---------------------------------


def test_precomputed_tissue_mask_opens_with_explicit_mask_backend(monkeypatch):
    opened: list[str] = []

    def _open(path, backend=None):
        opened.append(backend)
        return _ArrayMaskSlide(np.array([[0, 1], [0, 0]], dtype=np.uint8))

    monkeypatch.setattr(mask_mod, "open_slide", _open)
    load_precomputed_tissue_mask(
        mask_path="/masks/m.tif",
        slide=_wsi(backend_name="cucim"),
        seg_level=0,
        tissue_value=1,
        mask_backend="openslide",
    )
    assert opened == ["openslide"]


def test_annotation_mask_opens_with_explicit_mask_backend(monkeypatch):
    opened: list[str] = []

    def _open(path, backend=None):
        opened.append(backend)
        return _ArrayMaskSlide(np.array([[0, 1], [0, 0]], dtype=np.uint8))

    monkeypatch.setattr(mask_mod, "open_slide", _open)
    load_annotation_label_mask(
        mask_path="/masks/a.tif",
        slide=_wsi(backend_name="cucim"),
        seg_level=0,
        valid_values={0, 1},
        mask_backend="asap",
    )
    assert opened == ["asap"]


def test_resolve_tissue_mask_uses_mask_backend_not_slide_backend(monkeypatch):
    opened: list[str] = []

    def _open(path, backend=None):
        opened.append(backend)
        return _ArrayMaskSlide(np.array([[0, 1], [0, 0]], dtype=np.uint8))

    monkeypatch.setattr(mask_mod, "open_slide", _open)
    resolve_tissue_mask(
        slide=_wsi(backend_name="cucim"),
        tissue_mask_path="/masks/m.tif",
        tissue_mask_tissue_value=1,
        seg_downsample=1,
        mask_backend="vips",
    )
    assert opened == ["vips"]


def test_resolve_annotation_masks_uses_mask_backend(monkeypatch):
    opened: list[str] = []

    def _open(path, backend=None):
        opened.append(backend)
        return _ArrayMaskSlide(np.array([[0, 1], [0, 0]], dtype=np.uint8))

    monkeypatch.setattr(mask_mod, "open_slide", _open)
    resolve_annotation_masks(
        slide=_wsi(backend_name="cucim"),
        mask_path="/masks/a.tif",
        pixel_mapping={"background": 0, "tumor": 1},
        seg_downsample=1,
        mask_backend="openslide",
    )
    assert opened == ["openslide"]


def test_resolve_tissue_mask_direct_call_omitting_backend_resolves_from_mask_path(monkeypatch):
    """A direct caller who omits ``mask_backend`` resolves the mask independently from the mask
    path via ``auto`` — NOT forced to the slide's resolved backend — and records the requested
    provenance as ``"auto"``."""
    opened: list[str] = []

    def _open(path, backend=None):
        opened.append(backend)
        return _ArrayMaskSlide(np.array([[0, 1], [0, 0]], dtype=np.uint8))

    def _fake_resolve(requested, *, wsi_path, mask_path=None):
        # ``auto`` resolution over the mask's own path selects openslide; the slide's resolved
        # backend (asap) must not be consulted.
        if requested == "auto":
            return reader_mod.BackendSelection(backend="openslide")
        return reader_mod.BackendSelection(backend=requested)

    monkeypatch.setattr(mask_mod, "open_slide", _open)
    monkeypatch.setattr(mask_mod, "resolve_backend", _fake_resolve)
    result = resolve_tissue_mask(
        slide=_wsi(backend_name="asap"),
        tissue_mask_path="/masks/m.tif",
        tissue_mask_tissue_value=1,
        seg_downsample=1,
    )
    assert opened == ["openslide"]
    assert result.requested_mask_backend == "auto"
    assert result.mask_backend == "openslide"


def test_resolve_annotation_masks_direct_call_omitting_backend_resolves_from_mask_path(monkeypatch):
    """The annotation counterpart: omitting ``mask_backend`` resolves the mask from its own
    path via ``auto`` (not the slide backend) and records requested provenance ``"auto"``."""
    opened: list[str] = []

    def _open(path, backend=None):
        opened.append(backend)
        return _ArrayMaskSlide(np.array([[0, 1], [0, 0]], dtype=np.uint8))

    def _fake_resolve(requested, *, wsi_path, mask_path=None):
        if requested == "auto":
            return reader_mod.BackendSelection(backend="openslide")
        return reader_mod.BackendSelection(backend=requested)

    monkeypatch.setattr(mask_mod, "open_slide", _open)
    monkeypatch.setattr(mask_mod, "resolve_backend", _fake_resolve)
    result = resolve_annotation_masks(
        slide=_wsi(backend_name="asap"),
        mask_path="/masks/a.tif",
        pixel_mapping={"background": 0, "tumor": 1},
        seg_downsample=1,
    )
    assert opened == ["openslide"]
    assert result.requested_mask_backend == "auto"
    assert result.mask_backend == "openslide"


def test_mask_decode_error_names_resolved_mask_backend(monkeypatch):
    def _open(path, backend=None):
        raise RuntimeError("codec unavailable")

    monkeypatch.setattr(mask_mod, "open_slide", _open)
    with pytest.raises(RuntimeError) as excinfo:
        load_precomputed_tissue_mask(
            mask_path="/masks/broken.tif",
            slide=_wsi(backend_name="cucim"),
            seg_level=0,
            tissue_value=1,
            mask_backend="openslide",
        )
    message = str(excinfo.value)
    assert "/masks/broken.tif" in message
    assert "backend=openslide" in message
    assert "cucim" not in message


# --- centralized mask-open helper (Finding 6) -------------------------------------------


def test_open_mask_reader_incompatible_backend_raises_actionable_error(monkeypatch):
    """An open failure names the mask path and the requested backend, not a raw codec error."""

    def _boom(path, backend=reader_mod.AUTO_BACKEND, **kwargs):
        raise RuntimeError("codec: unsupported compression scheme")

    monkeypatch.setattr(reader_mod, "open_slide", _boom)
    with pytest.raises(RuntimeError) as excinfo:
        reader_mod.open_mask_reader("/masks/incompatible.tif", mask_backend="cucim")
    message = str(excinfo.value)
    assert "/masks/incompatible.tif" in message
    assert "cucim" in message
    assert "codec" in message
    assert "Select another mask backend" in message


def test_open_mask_reader_value_error_cause_reraises_as_value_error(monkeypatch):
    def _boom(path, backend=reader_mod.AUTO_BACKEND, **kwargs):
        raise ValueError("bad mask geometry")

    monkeypatch.setattr(reader_mod, "open_slide", _boom)
    with pytest.raises(ValueError) as excinfo:
        reader_mod.open_mask_reader("/masks/bad.tif", mask_backend="openslide")
    message = str(excinfo.value)
    assert "/masks/bad.tif" in message
    assert "openslide" in message


def test_open_mask_reader_returns_reader_and_resolved_backend(monkeypatch):
    sentinel = object()

    def _open(path, backend=reader_mod.AUTO_BACKEND, **kwargs):
        return sentinel

    monkeypatch.setattr(reader_mod, "open_slide", _open)
    reader, resolved = reader_mod.open_mask_reader("/masks/m.tif", mask_backend="asap")
    assert reader is sentinel
    assert resolved == "asap"


def test_empty_precomputed_warning_names_resolved_mask_backend(monkeypatch, caplog):
    monkeypatch.setattr(
        mask_mod,
        "open_slide",
        lambda path, backend=None: _ArrayMaskSlide(np.zeros((2, 2), dtype=np.uint8)),
    )
    caplog.set_level("WARNING", logger="hs2p.tiling.mask")
    resolve_tissue_mask(
        slide=_wsi(backend_name="cucim"),
        sample_id="case-9",
        tissue_mask_path="/masks/empty.tif",
        tissue_mask_tissue_value=1,
        seg_downsample=1,
        mask_backend="openslide",
    )
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "backend=openslide" in warnings[0]
    assert "genuinely" in warnings[0].lower() or "intentionally" in warnings[0].lower()
    assert "decode" in warnings[0].lower()
