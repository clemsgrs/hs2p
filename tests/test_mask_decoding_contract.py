from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import hs2p.tiling.mask as mask_mod
from hs2p.tiling.mask import (
    load_precomputed_tissue_mask,
    resolve_annotation_masks,
    resolve_tissue_mask,
)


class _ArrayMaskSlide:
    spacing = 0.25
    level_downsamples = [1.0]

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


def test_invalid_tissue_labels_fail_without_a_direct_tiff_read(monkeypatch):
    backend_mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    direct_reads = []
    monkeypatch.setattr(
        mask_mod,
        "open_slide",
        lambda path, backend=None: _ArrayMaskSlide(backend_mask),
    )
    monkeypatch.setattr(
        Image,
        "open",
        lambda *args, **kwargs: direct_reads.append((args, kwargs))
        or (_ for _ in ()).throw(AssertionError("direct TIFF read attempted")),
    )

    with pytest.raises(ValueError) as excinfo:
        load_precomputed_tissue_mask(
            mask_path="/masks/sparse.tif",
            slide=_wsi(),
            seg_level=0,
            tissue_value=1,
            mask_backend="cucim",
        )

    assert direct_reads == []
    message = str(excinfo.value)
    assert "Precomputed tissue mask" in message
    assert "/masks/sparse.tif" in message
    assert "backend=cucim" in message
    assert "select another backend" in message.lower()
    assert "regenerate" in message.lower()


def test_tissue_backend_exception_fails_without_an_alternate_read(monkeypatch):
    opened_backends = []

    def fail_open(path, backend=None):
        opened_backends.append(backend)
        raise RuntimeError("codec unavailable")

    monkeypatch.setattr(mask_mod, "open_slide", fail_open)

    with pytest.raises(RuntimeError) as excinfo:
        load_precomputed_tissue_mask(
            mask_path="/masks/decode-error.tif",
            slide=_wsi(),
            seg_level=0,
            tissue_value=1,
            mask_backend="cucim",
        )

    assert opened_backends == ["cucim"]
    message = str(excinfo.value)
    assert "Precomputed tissue mask" in message
    assert "/masks/decode-error.tif" in message
    assert "backend=cucim" in message
    assert "codec unavailable" in message
    assert "select another backend" in message.lower()
    assert "regenerate" in message.lower()


def test_empty_precomputed_tissue_mask_succeeds_with_one_contextual_warning(
    monkeypatch, caplog
):
    mask_path = "/masks/empty.tif"
    monkeypatch.setattr(
        mask_mod,
        "open_slide",
        lambda path, backend=None: _ArrayMaskSlide(
            np.zeros((2, 2), dtype=np.uint8)
        ),
    )
    caplog.set_level("WARNING", logger="hs2p.tiling.mask")

    resolved = resolve_tissue_mask(
        slide=_wsi(),
        sample_id="case-017",
        tissue_mask_path=mask_path,
        tissue_mask_tissue_value=7,
        seg_downsample=1,
        mask_backend="cucim",
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

    def open_mask(path, backend=None):
        opened_backends.append(backend)
        return _ArrayMaskSlide(np.array([[0, 7], [0, 0]], dtype=np.uint8))

    monkeypatch.setattr(mask_mod, "open_slide", open_mask)
    caplog.set_level("WARNING", logger="hs2p.tiling.mask")

    resolved = resolve_tissue_mask(
        slide=_wsi(),
        sample_id="case-018",
        tissue_mask_path="/masks/nonempty.tif",
        tissue_mask_tissue_value=7,
        seg_downsample=1,
        mask_backend="cucim",
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
        tissue_mask_path=None,
        seg_downsample=1,
    )

    assert not np.any(resolved.tissue_mask)
    assert [record for record in caplog.records if record.levelname == "WARNING"] == []


def test_single_value_annotation_mask_succeeds_without_empty_mask_warning(
    monkeypatch, caplog
):
    monkeypatch.setattr(
        mask_mod,
        "open_slide",
        lambda path, backend=None: _ArrayMaskSlide(
            np.zeros((2, 2), dtype=np.uint8)
        ),
    )
    caplog.set_level("WARNING", logger="hs2p.tiling.mask")

    resolved = resolve_annotation_masks(
        slide=_wsi(),
        mask_path="/masks/annotations.tif",
        pixel_mapping={"background": 0, "tumor": 1},
        seg_downsample=1,
        mask_backend="cucim",
    )

    assert np.all(resolved.masks["background"] == 255)
    assert not np.any(resolved.masks["tumor"])
    assert [record for record in caplog.records if record.levelname == "WARNING"] == []
