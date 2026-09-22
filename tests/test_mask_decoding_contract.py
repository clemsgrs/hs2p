from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import hs2p.mask as source_mask_mod
import hs2p.tiling.mask as mask_mod
from hs2p.mask import AnnotationLabels, Mask, TissueLabels
from hs2p.tiling.mask import resolve_annotation_masks, resolve_tissue_mask


class _ArrayMaskSlide:
    spacing = 0.25
    native_spacing = 0.25
    level_downsamples = [(1.0, 1.0)]

    def __init__(self, mask: np.ndarray):
        self._mask = mask
        self.level_dimensions = [(mask.shape[1], mask.shape[0])]

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


def _open_tissue_mask(monkeypatch, native: np.ndarray, *, path: str, tissue: int) -> Mask:
    """Open a tissue ``Mask`` whose ``cucim`` backend decodes to ``native``."""
    monkeypatch.setattr(
        source_mask_mod,
        "open_slide",
        lambda path, backend=None, **kwargs: _ArrayMaskSlide(native),
    )
    return Mask(
        path=path, labels=TissueLabels(background=0, tissue=tissue), backend="cucim"
    )


def test_invalid_tissue_labels_fail_without_a_direct_tiff_read(monkeypatch):
    direct_reads = []
    mask = _open_tissue_mask(
        monkeypatch,
        np.array([[0, 1], [2, 0]], dtype=np.uint8),
        path="/masks/sparse.tif",
        tissue=1,
    )
    monkeypatch.setattr(
        Image,
        "open",
        lambda *args, **kwargs: direct_reads.append((args, kwargs))
        or (_ for _ in ()).throw(AssertionError("direct TIFF read attempted")),
    )

    with pytest.raises(ValueError) as excinfo:
        resolve_tissue_mask(slide=_wsi(), mask=mask, seg_downsample=1)

    assert direct_reads == []
    message = str(excinfo.value)
    assert "/masks/sparse.tif" in message
    assert "backend=cucim" in message
    assert "undeclared label IDs [2]" in message


def test_tissue_backend_exception_fails_without_an_alternate_read(monkeypatch):
    opened_backends = []

    def fail_open(path, backend=None, **kwargs):
        opened_backends.append(backend)
        raise RuntimeError("codec unavailable")

    monkeypatch.setattr(source_mask_mod, "open_slide", fail_open)

    with pytest.raises(RuntimeError) as excinfo:
        Mask(
            path="/masks/decode-error.tif",
            labels=TissueLabels(background=0, tissue=1),
            backend="cucim",
        )

    assert opened_backends == ["cucim"]
    message = str(excinfo.value)
    assert "/masks/decode-error.tif" in message
    assert "backend=cucim" in message
    assert "codec unavailable" in message


def test_empty_precomputed_tissue_mask_succeeds_with_one_contextual_warning(
    monkeypatch, caplog
):
    mask_path = "/masks/empty.tif"
    mask = _open_tissue_mask(
        monkeypatch, np.zeros((2, 2), dtype=np.uint8), path=mask_path, tissue=7
    )
    caplog.set_level("WARNING", logger="hs2p.tiling.mask")

    resolved = resolve_tissue_mask(
        slide=_wsi(),
        sample_id="case-017",
        mask=mask,
        seg_downsample=1,
    )

    assert np.array_equal(resolved.tissue_mask, np.zeros((2, 2), dtype=np.uint8))
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    warning = warnings[0]
    assert "case-017" in warning
    assert mask_path in warning
    assert "cucim" in warning
    assert "mask_level=0" in warning
    assert "tissue_value=7" in warning


def test_nonempty_precomputed_tissue_mask_uses_one_backend_without_warning(
    monkeypatch, caplog
):
    opened_backends = []

    def open_mask(path, backend=None, **kwargs):
        opened_backends.append(backend)
        return _ArrayMaskSlide(np.array([[0, 7], [0, 0]], dtype=np.uint8))

    monkeypatch.setattr(source_mask_mod, "open_slide", open_mask)
    caplog.set_level("WARNING", logger="hs2p.tiling.mask")

    with Mask(
        path="/masks/nonempty.tif",
        labels=TissueLabels(background=0, tissue=7),
        backend="cucim",
    ) as mask:
        resolved = resolve_tissue_mask(
            slide=_wsi(),
            sample_id="case-018",
            mask=mask,
            seg_downsample=1,
        )

    assert opened_backends == ["cucim"]
    assert np.array_equal(
        resolved.tissue_mask,
        np.array([[0, 255], [0, 0]], dtype=np.uint8),
    )
    assert [record for record in caplog.records if record.levelname == "WARNING"] == []


@pytest.mark.parametrize("method", ["hsv", "sam2"])
def test_empty_generated_tissue_masks_do_not_warn(monkeypatch, caplog, method):
    slide = SimpleNamespace(
        spacing=0.25,
        level_downsamples=[1.0],
        level_dimensions=[(2, 2)],
        dimensions=(2, 2),
        backend_name="cucim",
        read_region=lambda location, level, size: np.zeros(
            (size[1], size[0], 3), dtype=np.uint8
        ),
    )
    monkeypatch.setattr(
        mask_mod,
        "segment_tissue_image",
        lambda image, config: np.zeros(image.shape[:2], dtype=np.uint8),
    )
    caplog.set_level("WARNING", logger="hs2p.tiling.mask")

    resolved = resolve_tissue_mask(
        slide=slide,
        sample_id=f"generated-{method}",
        tissue_method=method,
        seg_downsample=1,
    )

    assert not np.any(resolved.tissue_mask)
    assert [record for record in caplog.records if record.levelname == "WARNING"] == []


def test_single_value_annotation_mask_succeeds_without_empty_mask_warning(
    monkeypatch, caplog
):
    monkeypatch.setattr(
        source_mask_mod,
        "open_slide",
        lambda path, backend=None, **kwargs: _ArrayMaskSlide(
            np.zeros((2, 2), dtype=np.uint8)
        ),
    )
    caplog.set_level("WARNING", logger="hs2p.tiling.mask")

    with Mask(
        path="/masks/annotations.tif",
        labels=AnnotationLabels(pixel_mapping={"background": 0, "tumor": 1}),
        backend="cucim",
    ) as mask:
        resolved = resolve_annotation_masks(slide=_wsi(), mask=mask, seg_downsample=1)

    assert np.all(resolved.masks["background"] == 255)
    assert not np.any(resolved.masks["tumor"])
    assert [record for record in caplog.records if record.levelname == "WARNING"] == []
