"""The annotation sampling workflow through its configuration-driven entrypoints (#192).

A real flat PNG slide is sampled against an in-memory annotation mask pyramid served by the
``openslide`` backend slot, so ``tile_slide`` / ``tile_slides`` run end to end without a
native backend.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import hs2p.wsi.reader as reader_mod
from hs2p.api import (
    BatchPartialFailureWarning,
    SlideSpec,
    TilingConfig,
    tile_slide,
    tile_slides,
)
from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy, SamplingSpec

# "tumor" owns two mask values; every value the mask may hold is declared.
PIXEL_MAPPING = {"background": 0, "tumor": [1, 2], "stroma": 3}

TUMOR_TILES = [(x, y) for x in (0, 64, 128, 192) for y in (0, 64)]
STROMA_TILES = [(x, y) for x in (128, 192) for y in (128, 192)]


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


def _annotations(*, top_right: int = 2, bottom_right: int = 3) -> np.ndarray:
    """Tumor value 1 top-left, ``top_right`` and ``bottom_right`` in their quadrants."""
    labels = np.zeros((256, 256), dtype=np.uint8)
    labels[:128, :128] = 1
    labels[:128, 128:] = top_right
    labels[128:, 128:] = bottom_right
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


def _slide_spec(tmp_path: Path, *, mask_name: str = "mask.tif") -> SlideSpec:
    """A 256 px flat slide at 0.5 um/px with a (never decoded) mask path."""
    image_path = tmp_path / "slide.png"
    Image.fromarray(np.full((256, 256, 3), 200, dtype=np.uint8)).save(image_path)
    mask_path = tmp_path / mask_name
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


def _sampling() -> SamplingSpec:
    return SamplingSpec(
        pixel_mapping=PIXEL_MAPPING,
        color_mapping=None,
        tissue_percentage={"background": None, "tumor": 0.5, "stroma": 0.5},
        active_annotations=("tumor", "stroma"),
    )


def _tiles(result) -> list[tuple[int, int]]:
    return sorted(zip(result.x.tolist(), result.y.tolist()))


@pytest.mark.parametrize(
    "strategy",
    [
        CoordinateSelectionStrategy.JOINT_SAMPLING,
        CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
    ],
)
def test_tile_slide_samples_each_annotation_of_a_multi_value_mask(
    monkeypatch, tmp_path, strategy
):
    reader = _pyramid(_annotations())
    whole_slide = _slide_spec(tmp_path)
    opened = _serve_mask(monkeypatch, reader)

    results = tile_slide(
        whole_slide,
        tiling=_tiling(),
        sampling=_sampling(),
        selection_strategy=strategy,
    )

    assert set(results) == {"tumor", "stroma"}
    # "tumor" is the union of mask values 1 and 2: the whole top half.
    assert _tiles(results["tumor"]) == sorted(TUMOR_TILES)
    assert _tiles(results["stroma"]) == sorted(STROMA_TILES)
    assert [results[name].annotation for name in ("tumor", "stroma")] == [
        "tumor",
        "stroma",
    ]
    # One read of the 0.5 um/px level, not the finer level 0, serves every annotation.
    assert reader.read_levels == [1]
    for result in results.values():
        assert (result.seg_level, result.seg_spacing_um) == (0, 0.5)
        assert (result.mask_level, result.mask_spacing_um) == (1, 0.5)
        assert result.requested_mask_backend == "openslide"
        assert result.mask_backend == "openslide"
    assert opened == [str(whole_slide.mask_path)]
    assert reader.close_count == 1


def test_tile_slide_merges_annotation_coordinates_into_one_result(monkeypatch, tmp_path):
    reader = _pyramid(_annotations())
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)

    results = tile_slide(
        whole_slide,
        tiling=_tiling(),
        sampling=_sampling(),
        output_mode=CoordinateOutputMode.MERGED,
    )

    assert set(results) == {None}
    assert _tiles(results[None]) == sorted(TUMOR_TILES + STROMA_TILES)
    assert results[None].annotation is None


def test_tile_slide_accepts_a_mask_holding_a_subset_of_the_declared_ids(
    monkeypatch, tmp_path
):
    # Neither tumor's second value (2) nor stroma (3) occurs in this mask.
    reader = _pyramid(_annotations(top_right=0, bottom_right=0))
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)

    results = tile_slide(whole_slide, tiling=_tiling(), sampling=_sampling())

    assert _tiles(results["tumor"]) == [(0, 0), (0, 64), (64, 0), (64, 64)]
    assert _tiles(results["stroma"]) == []


def test_tile_slide_closes_the_annotation_mask_when_its_decode_fails(
    monkeypatch, tmp_path
):
    reader = _pyramid(_annotations(), decode_error=RuntimeError("codec unavailable"))
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)

    with pytest.raises(RuntimeError, match="codec unavailable") as excinfo:
        tile_slide(whole_slide, tiling=_tiling(), sampling=_sampling())

    assert str(whole_slide.mask_path) in str(excinfo.value)
    assert "backend=openslide" in str(excinfo.value)
    assert reader.close_count == 1


def test_tile_slide_opens_the_annotation_mask_with_the_preflight_backend(
    monkeypatch, tmp_path
):
    reader = _pyramid(_annotations())
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)
    probed: list[str] = []

    def _only_openslide_opens(*, backend, **kwargs):
        probed.append(backend)
        return backend == "openslide"

    monkeypatch.setattr(reader_mod, "_backend_can_open_source", _only_openslide_opens)

    results = tile_slide(
        whole_slide, tiling=_tiling(mask_backend="auto"), sampling=_sampling()
    )

    # One preflight probe sequence; opening the mask does not resolve ``auto`` again.
    assert probed == ["cucim", "vips", "openslide"]
    assert results["tumor"].requested_mask_backend == "auto"
    assert results["tumor"].mask_backend == "openslide"


def test_tile_slides_records_annotation_mask_provenance(monkeypatch, tmp_path):
    reader = _pyramid(_annotations())
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)

    artifacts = tile_slides(
        [whole_slide],
        tiling=_tiling(),
        output_dir=tmp_path / "output",
        num_workers=1,
        sampling=_sampling(),
    )

    assert {artifact.annotation: artifact.num_tiles for artifact in artifacts} == {
        "tumor": 8,
        "stroma": 4,
    }
    for artifact in artifacts:
        assert artifact.requested_mask_backend == "openslide"
        assert artifact.mask_backend == "openslide"
    rows = pd.read_csv(tmp_path / "output" / "process_list.csv")
    assert rows["tiling_status"].tolist() == ["success", "success"]
    assert rows["requested_mask_backend"].tolist() == ["openslide", "openslide"]
    assert rows["mask_backend"].tolist() == ["openslide", "openslide"]
    assert reader.close_count == 1


def test_tile_slides_records_a_failed_annotation_mask_read(monkeypatch, tmp_path):
    reader = _pyramid(_annotations(), decode_error=RuntimeError("codec unavailable"))
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)

    with pytest.warns(BatchPartialFailureWarning):
        tile_slides(
            [whole_slide],
            tiling=_tiling(),
            output_dir=tmp_path / "output",
            num_workers=1,
            sampling=_sampling(),
        )

    rows = pd.read_csv(tmp_path / "output" / "process_list.csv")
    assert rows["tiling_status"].tolist() == ["failed"]
    assert "codec unavailable" in rows["error"].iloc[0]
    assert rows["requested_mask_backend"].tolist() == ["openslide"]
    assert rows["mask_backend"].tolist() == ["openslide"]
    assert reader.close_count == 1


def test_tile_slide_rejects_an_annotation_id_outside_the_declared_mapping(
    monkeypatch, tmp_path
):
    # 4 is owned by no label of the pixel mapping.
    reader = _pyramid(_annotations(bottom_right=4))
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)

    with pytest.raises(ValueError, match=r"undeclared label IDs \[4\]"):
        tile_slide(whole_slide, tiling=_tiling(), sampling=_sampling())

    assert reader.close_count == 1


def test_tile_slide_rejects_an_annotation_mask_that_does_not_cover_the_slide(
    monkeypatch, tmp_path
):
    # A 256x128 mask cannot span the 256x256 slide at one scale.
    reader = _MaskPyramid([_annotations()[:128]], spacing=0.5)
    whole_slide = _slide_spec(tmp_path)
    _serve_mask(monkeypatch, reader)

    with pytest.raises(ValueError, match="Mask alignment failed"):
        tile_slide(whole_slide, tiling=_tiling(), sampling=_sampling())

    assert reader.read_levels == []
    assert reader.close_count == 1


def test_tile_slide_samples_a_flat_png_annotation_mask_without_spacing(tmp_path):
    whole_slide = _slide_spec(tmp_path, mask_name="mask.png")
    Image.fromarray(_annotations()).save(whole_slide.mask_path)

    results = tile_slide(
        whole_slide, tiling=_tiling(mask_backend="auto"), sampling=_sampling()
    )

    assert _tiles(results["tumor"]) == sorted(TUMOR_TILES)
    assert _tiles(results["stroma"]) == sorted(STROMA_TILES)
    # The spacing is the slide's 0.5 um/px scaled by the 256 px / 256 px dimension ratio.
    assert (results["tumor"].mask_level, results["tumor"].mask_spacing_um) == (0, 0.5)
    assert results["tumor"].mask_backend == "pil"
