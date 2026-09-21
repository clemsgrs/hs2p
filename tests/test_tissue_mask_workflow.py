"""The precomputed tissue workflow through its configuration-driven entrypoints (#191).

A real flat PNG slide is tiled against an in-memory mask pyramid served by the
``openslide`` backend slot, so ``tile_slide`` / ``tile_slides`` run end to end without a
native backend.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import hs2p.progress as progress
import hs2p.wsi.reader as reader_mod
from hs2p.api import SlideSpec, TilingConfig, tile_slide, tile_slides
from tests.test_progress import RecordingReporter


class _MaskPyramid:
    """An in-memory mask source; ``levels`` are the arrays each level decodes to."""

    def __init__(self, levels, *, spacing, decode_error=None):
        self._levels = levels
        self._decode_error = decode_error
        self.spacing = spacing
        self.native_spacing = spacing
        self.level_dimensions = [(level.shape[1], level.shape[0]) for level in levels]
        width_0 = self.level_dimensions[0][0]
        self.level_downsamples = [
            (width_0 / width, width_0 / width) for width, _ in self.level_dimensions
        ]
        self.read_levels: list[int] = []
        self.close_count = 0

    def read_region(self, location, level, size):
        self.read_levels.append(level)
        if self._decode_error is not None:
            raise self._decode_error
        return self._levels[level]

    def close(self):
        self.close_count += 1


def _pyramid(level_1: np.ndarray, **kwargs) -> _MaskPyramid:
    """A 512 px / 256 px mask pyramid at 0.25 / 0.5 um/px from its 256 px level."""
    level_0 = np.kron(level_1, np.ones((2, 2), dtype=level_1.dtype))
    return _MaskPyramid([level_0, level_1], spacing=0.25, **kwargs)


def _top_left_quadrant(*, background: int, tissue: int) -> np.ndarray:
    labels = np.full((256, 256), background, dtype=np.uint8)
    labels[:128, :128] = tissue
    return labels


def _serve_mask(monkeypatch, reader: _MaskPyramid) -> list[str]:
    """Serve ``reader`` from the ``openslide`` backend slot; returns the opened paths."""
    opened: list[str] = []

    def _open(path, **kwargs):
        opened.append(str(path))
        return reader

    monkeypatch.setitem(
        reader_mod._BACKENDS,
        "openslide",
        reader_mod._BackendSpec("openslide", _open, lambda path: True),
    )
    return opened


def _slide_spec(tmp_path: Path) -> SlideSpec:
    """A 256 px flat slide at 0.5 um/px with a (never decoded) mask path."""
    image_path = tmp_path / "slide.png"
    Image.fromarray(np.full((256, 256, 3), 200, dtype=np.uint8)).save(image_path)
    mask_path = tmp_path / "mask.tif"
    mask_path.write_bytes(b"")
    return SlideSpec(
        sample_id="slide-1",
        image_path=image_path,
        mask_path=mask_path,
        spacing_at_level_0=0.5,
    )


def _tiling(*, mask_backend: str = "openslide") -> TilingConfig:
    return TilingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=64,
        tolerance=0.05,
        overlap=0.0,
        min_coverage={"tissue": 0.5},
        backend="pil",
        mask_backend=mask_backend,
    )


def test_tile_slide_tiles_the_tissue_of_a_precomputed_mask(monkeypatch, tmp_path):
    reader = _pyramid(_top_left_quadrant(background=0, tissue=1))
    whole_slide = _slide_spec(tmp_path)
    opened = _serve_mask(monkeypatch, reader)

    result = tile_slide(whole_slide, tiling=_tiling())

    assert sorted(zip(result.x.tolist(), result.y.tolist())) == [
        (0, 0),
        (0, 64),
        (64, 0),
        (64, 64),
    ]
    expected_mask = np.zeros((256, 256), dtype=np.uint8)
    expected_mask[:128, :128] = 255
    np.testing.assert_array_equal(result.tissue_mask, expected_mask)
    assert result.tissue_method == "precomputed_mask"
    assert result.tissue_mask_tissue_value == 1
    assert (result.seg_level, result.seg_spacing_um) == (0, 0.5)
    # The 0.5 um/px level is read, not the finer level 0.
    assert reader.read_levels == [1]
    assert (result.mask_level, result.mask_spacing_um) == (1, 0.5)
    assert opened == [str(whole_slide.mask_path)]
    assert reader.close_count == 1


def test_tile_slide_closes_the_mask_when_its_decode_fails(monkeypatch, tmp_path):
    reader = _pyramid(
        _top_left_quadrant(background=0, tissue=1),
        decode_error=RuntimeError("codec unavailable"),
    )
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)

    with pytest.raises(RuntimeError, match="codec unavailable") as excinfo:
        tile_slide(whole_slide, tiling=_tiling())

    assert str(whole_slide.mask_path) in str(excinfo.value)
    assert "backend=openslide" in str(excinfo.value)
    assert reader.close_count == 1


def test_tile_slide_opens_the_mask_with_the_preflight_backend(monkeypatch, tmp_path):
    reader = _pyramid(_top_left_quadrant(background=0, tissue=1))
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)
    probed: list[str] = []

    def _only_openslide_opens(*, backend, **kwargs):
        probed.append(backend)
        return backend == "openslide"

    monkeypatch.setattr(reader_mod, "_backend_can_open_source", _only_openslide_opens)

    result = tile_slide(whole_slide, tiling=_tiling(mask_backend="auto"))

    # One preflight probe sequence; opening the mask does not resolve ``auto`` again.
    assert probed == ["cucim", "vips", "openslide"]
    assert result.requested_mask_backend == "auto"
    assert result.mask_backend == "openslide"


def test_tile_slides_reports_an_empty_precomputed_mask(monkeypatch, tmp_path, caplog):
    reader = _pyramid(np.zeros((256, 256), dtype=np.uint8))
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)
    reporter = RecordingReporter()
    caplog.set_level("WARNING", logger="hs2p.tiling.mask")

    with progress.activate_progress_reporter(reporter):
        artifacts = tile_slides(
            [whole_slide],
            tiling=_tiling(),
            output_dir=tmp_path / "output",
            num_workers=1,
        )

    assert [artifact.num_tiles for artifact in artifacts] == [0]
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and record.name == "hs2p.tiling.mask"
    ]
    assert len(warnings) == 1
    assert "Empty precomputed tissue mask" in warnings[0]
    assert "sample_id=slide-1" in warnings[0]
    assert f"mask_path={whole_slide.mask_path}" in warnings[0]
    assert "backend=openslide" in warnings[0]
    assert "mask_level=1" in warnings[0]
    assert "tissue_value=1" in warnings[0]
    finished = next(e for e in reporter.events if e.kind == "tissue.finished")
    assert finished.payload["empty_masks"] == 1
    assert reader.close_count == 1
