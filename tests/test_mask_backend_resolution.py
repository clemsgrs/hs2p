"""Independent slide/mask backend resolution seam + resolver backend provenance (#163)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import hs2p.mask as source_mask_mod
import hs2p.wsi.reader as reader_mod
from hs2p.mask import AnnotationLabels, Mask, TissueLabels
from hs2p.wsi.backend import resolve_backends
from hs2p.tiling.mask import resolve_annotation_masks, resolve_tissue_mask


class _ArrayMaskSlide:
    spacing = 0.25
    native_spacing = 0.25
    level_downsamples = [(1.0, 1.0)]

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

    def _fake_can_open(
        *,
        source_path,
        companion_path,
        backend,
        spacing_override=None,
        require_spacing=True,
    ):
        del companion_path, spacing_override
        seen.append((str(source_path), backend))
        # slide.svs → cucim opens; mask.tif → only asap opens
        if "mask" in str(source_path):
            return backend == "asap"
        return backend == "cucim"

    monkeypatch.setattr(reader_mod, "_backend_can_open_source", _fake_can_open)

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
        "_backend_can_open_source",
        lambda *, backend, **kwargs: backend == "cucim",
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

    def _fake_can_open(
        *,
        source_path,
        companion_path,
        backend,
        spacing_override=None,
        require_spacing=True,
    ):
        del companion_path, spacing_override
        probed_paths.append(str(source_path))
        # cucim can open the slide; the mask is a different format only asap opens
        if "mask" in str(source_path):
            return backend == "asap"
        return backend == "cucim"

    monkeypatch.setattr(reader_mod, "_backend_can_open_source", _fake_can_open)
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

    monkeypatch.setattr(reader_mod, "_backend_can_open_source", _boom)
    resolved = resolve_backends(
        requested_slide_backend="openslide",
        requested_mask_backend="cucim",
        wsi_path=Path("slide.svs"),
        mask_path=Path("mask.tif"),
    )
    assert resolved.slide_backend == "openslide"
    assert resolved.mask_backend == "cucim"


# --- the resolvers record the mask's own backend ----------------------------------------


def _open_tissue_mask(monkeypatch, native: np.ndarray, *, path: str, backend: str) -> Mask:
    monkeypatch.setattr(
        source_mask_mod,
        "open_slide",
        lambda path, backend=None, **kwargs: _ArrayMaskSlide(native),
    )
    return Mask(path=path, labels=TissueLabels(background=0, tissue=1), backend=backend)


def test_resolve_tissue_mask_records_the_mask_backend_not_the_slide_backend(monkeypatch):
    mask = _open_tissue_mask(
        monkeypatch,
        np.array([[0, 1], [0, 0]], dtype=np.uint8),
        path="/masks/m.tif",
        backend="vips",
    )
    result = resolve_tissue_mask(
        slide=_wsi(backend_name="cucim"), mask=mask, seg_downsample=1
    )
    assert result.mask_backend == "vips"
    # The mask only knows the backend it opened; no request was supplied.
    assert result.requested_mask_backend == "vips"


def _open_annotation_mask(monkeypatch, native: np.ndarray, *, path: str, backend: str) -> Mask:
    monkeypatch.setattr(
        source_mask_mod,
        "open_slide",
        lambda path, backend=None, **kwargs: _ArrayMaskSlide(native),
    )
    return Mask(
        path=path,
        labels=AnnotationLabels(pixel_mapping={"background": 0, "tumor": 1}),
        backend=backend,
    )


def test_resolve_annotation_masks_records_the_mask_backend_not_the_slide_backend(monkeypatch):
    mask = _open_annotation_mask(
        monkeypatch,
        np.array([[0, 1], [0, 0]], dtype=np.uint8),
        path="/masks/a.tif",
        backend="openslide",
    )
    result = resolve_annotation_masks(
        slide=_wsi(backend_name="cucim"), mask=mask, seg_downsample=1
    )
    assert result.mask_backend == "openslide"
    # The mask only knows the backend it opened; no request was supplied.
    assert result.requested_mask_backend == "openslide"


def test_resolve_tissue_mask_records_the_callers_requested_mask_backend(monkeypatch):
    """Requested provenance is the caller's: the mask exposes only the backend it opened."""
    mask = _open_tissue_mask(
        monkeypatch,
        np.array([[0, 1], [0, 0]], dtype=np.uint8),
        path="/masks/m.tif",
        backend="openslide",
    )
    result = resolve_tissue_mask(
        slide=_wsi(backend_name="asap"),
        mask=mask,
        seg_downsample=1,
        requested_mask_backend="auto",
    )
    assert result.requested_mask_backend == "auto"
    assert result.mask_backend == "openslide"


def test_resolve_annotation_masks_records_the_callers_requested_mask_backend(monkeypatch):
    """The annotation counterpart: requested provenance is the caller's."""
    mask = _open_annotation_mask(
        monkeypatch,
        np.array([[0, 1], [0, 0]], dtype=np.uint8),
        path="/masks/a.tif",
        backend="openslide",
    )
    result = resolve_annotation_masks(
        slide=_wsi(backend_name="asap"),
        mask=mask,
        seg_downsample=1,
        requested_mask_backend="auto",
    )
    assert result.requested_mask_backend == "auto"
    assert result.mask_backend == "openslide"


def test_empty_precomputed_warning_names_resolved_mask_backend(monkeypatch, caplog):
    mask = _open_tissue_mask(
        monkeypatch,
        np.zeros((2, 2), dtype=np.uint8),
        path="/masks/empty.tif",
        backend="openslide",
    )
    caplog.set_level("WARNING", logger="hs2p.tiling.mask")
    resolve_tissue_mask(
        slide=_wsi(backend_name="cucim"),
        sample_id="case-9",
        mask=mask,
        seg_downsample=1,
    )
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "backend=openslide" in warnings[0]
    assert "genuinely" in warnings[0].lower() or "intentionally" in warnings[0].lower()
    assert "decode" in warnings[0].lower()
