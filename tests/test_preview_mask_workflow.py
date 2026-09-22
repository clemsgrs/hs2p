"""Source-backed previews through the first-class mask contract (#194).

Overlay and coordinate previews consume a :class:`~hs2p.mask.Mask` (or an aligned view of
one) instead of opening their own mask readers; tiling-preview workers build and close their
own mask from plain values.
"""

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import hs2p.mask as maskmod
import hs2p.tiling.orchestration as orchmod
import hs2p.wsi.visualization as visualization_mod
from hs2p.api import PreviewConfig, SlideSpec, TilingConfig, tile_slides
from hs2p.mask import AlignedMask, AnnotationLabels, Mask
from hs2p.wsi.preview import build_palette, draw_grid_from_coordinates
from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy, SamplingSpec

PIXEL_MAPPING = {"background": 0, "tumor": 1, "stroma": 2}
COLOR_MAPPING = {"background": None, "tumor": [255, 0, 0], "stroma": [0, 0, 255]}
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
# Red at alpha 0.5 over the gray 200 slide: round(200 * 127 / 255 + 255 * 128 / 255).
HALF_RED = (228, 100, 100)


class _FakeWSI:
    """A single-level square slide at 0.5 um/px, uniformly gray 200."""

    size = 6

    def __init__(self, path, backend="asap", **kwargs):
        del path, backend, kwargs
        self.spacings = [0.5]
        self.level_dimensions = [(self.size, self.size)]
        self.level_downsamples = [(1.0, 1.0)]

    def get_best_level_for_downsample_custom(self, downsample):
        del downsample
        return 0

    def get_level_spacing(self, level):
        return self.spacings[level]

    def get_slide(self, level):
        del level
        return np.full((self.size, self.size, 3), 200, dtype=np.uint8)


def _png_mask(tmp_path: Path, labels: np.ndarray) -> Mask:
    """A flat PNG annotation mask (no spacing metadata) declaring ``PIXEL_MAPPING``."""
    path = tmp_path / "mask.png"
    Image.fromarray(labels, mode="L").save(path)
    return Mask(path=path, labels=AnnotationLabels(pixel_mapping=PIXEL_MAPPING), backend="pil")


class _FakeMaskReader:
    """A single-level mask source served to ``Mask`` through a patched ``open_slide``."""

    def __init__(self, labels=None, *, dimensions=None, decode_error=None):
        self._labels = labels
        self._decode_error = decode_error
        self.native_spacing = None
        self.level_dimensions = [
            dimensions if dimensions is not None else (labels.shape[1], labels.shape[0])
        ]
        self.level_downsamples = [(1.0, 1.0)]
        self.read_count = 0
        self.close_count = 0

    def read_region(self, location, level, size):
        del location, level, size
        self.read_count += 1
        if self._decode_error is not None:
            raise self._decode_error
        return self._labels

    def close(self):
        self.close_count += 1


def _served_mask(monkeypatch, reader: _FakeMaskReader) -> Mask:
    monkeypatch.setattr(maskmod, "open_slide", lambda path, backend, **kwargs: reader)
    return Mask(
        path="/masks/served.tif",
        labels=AnnotationLabels(pixel_mapping=PIXEL_MAPPING),
        backend="openslide",
    )


def _overlay(mask: Mask) -> np.ndarray:
    overlay = visualization_mod.overlay_mask_on_slide(
        wsi_path=Path("slide.tif"),
        mask=mask,
        downsample=1,
        backend="asap",
        pixel_mapping=PIXEL_MAPPING,
        color_mapping=COLOR_MAPPING,
        alpha=1.0,
    )
    return np.array(overlay.convert("RGB"))


def _coordinate_preview(mask: Mask, save_dir: Path, *, tile_size_lv0: int = 4) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    visualization_mod.write_coordinate_preview(
        wsi_path=Path("slide.tif"),
        coordinates=[(0, 0)],
        tile_size_lv0=tile_size_lv0,
        save_dir=save_dir,
        backend="asap",
        sample_id="slide",
        downsample=1,
        mask=mask,
        palette=build_palette(pixel_mapping=PIXEL_MAPPING, color_mapping=COLOR_MAPPING),
        pixel_mapping=PIXEL_MAPPING,
        color_mapping=COLOR_MAPPING,
    )


# --- rendering ------------------------------------------------------------------------


def test_overlay_paints_a_coarser_source_mask_aligned_by_dimension_ratio(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(visualization_mod, "WSI", _FakeWSI)
    # 3x3 over the 6x6 slide: each mask pixel covers a 2x2 block, nearest-neighbor.
    labels = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=np.uint8)
    expected = np.full((6, 6, 3), 200, dtype=np.uint8)
    expected[0:2, 0:2] = RED
    expected[4:6, 4:6] = BLUE

    with _png_mask(tmp_path, labels) as mask:
        overlay = _overlay(mask)

    np.testing.assert_array_equal(overlay, expected)


def test_grid_preview_renders_the_aligned_mask_under_the_grid_with_a_restricted_palette(
    tmp_path,
):
    # The mask holds stroma (2) too: the aligned view validates the full vocabulary while
    # the palette only colors tumor.
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[0:4, 0:4] = 1
    labels[0:4, 4:6] = 2
    wsi = SimpleNamespace(
        level_downsamples=[(1.0, 1.0)],
        level_dimensions=[(6, 6)],
        get_level_spacing=lambda level: 0.5,
    )
    pixel_mapping = {"tumor": 1}
    color_mapping = {"tumor": [255, 0, 0]}
    expected = np.full((6, 6, 3), 200, dtype=np.uint8)
    expected[1:4, 1:4] = HALF_RED
    expected[0, 0:5] = expected[4, 0:5] = BLACK
    expected[0:5, 0] = expected[0:5, 4] = BLACK

    with _png_mask(tmp_path, labels) as mask:
        aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(6, 6))
        image = draw_grid_from_coordinates(
            np.full((6, 6, 3), 200, dtype=np.uint8),
            wsi,
            coords=[(0, 0)],
            tile_size_at_0=(4, 4),
            vis_level=0,
            thickness=1,
            mask=aligned,
            palette=build_palette(pixel_mapping=pixel_mapping, color_mapping=color_mapping),
            pixel_mapping=pixel_mapping,
            color_mapping=color_mapping,
        )

    np.testing.assert_array_equal(np.array(image), expected)


def test_coordinate_preview_writes_the_grid_over_the_source_mask(monkeypatch, tmp_path):
    # A 64 px slide: JPEG chroma subsampling would wash a 6 px preview out.
    monkeypatch.setattr(visualization_mod, "WSI", type("_LargeWSI", (_FakeWSI,), {"size": 64}))
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[0:32, 0:32] = 1

    with _png_mask(tmp_path, labels) as mask:
        _coordinate_preview(mask, tmp_path / "preview", tile_size_lv0=32)

    with Image.open(tmp_path / "preview" / "slide.jpg") as saved:
        arr = np.array(saved.convert("RGB")).astype(int)
    # JPEG-tolerant: the tile interior is red-dominant, its corner is the black grid line.
    assert arr[16, 16, 0] > arr[16, 16, 1] + 80 and arr[16, 16, 0] > arr[16, 16, 2] + 80
    assert arr[0, 0].max() <= 40


# --- validation shared with preprocessing --------------------------------------------


def test_overlay_rejects_an_undeclared_mask_value(monkeypatch, tmp_path):
    monkeypatch.setattr(visualization_mod, "WSI", _FakeWSI)
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[5, 5] = 3

    with _png_mask(tmp_path, labels) as mask, pytest.raises(
        ValueError, match=r"undeclared label IDs \[3\]"
    ):
        _overlay(mask)


def test_coordinate_preview_rejects_an_undeclared_mask_value(monkeypatch, tmp_path):
    monkeypatch.setattr(visualization_mod, "WSI", _FakeWSI)
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[5, 5] = 3

    with _png_mask(tmp_path, labels) as mask, pytest.raises(
        ValueError, match=r"undeclared label IDs \[3\]"
    ):
        _coordinate_preview(mask, tmp_path / "preview")


def test_overlay_refuses_a_native_read_above_the_pixel_cap_before_decoding(monkeypatch):
    monkeypatch.setattr(visualization_mod, "WSI", _FakeWSI)
    reader = _FakeMaskReader(dimensions=(20_000, 20_000))

    with _served_mask(monkeypatch, reader) as mask, pytest.raises(
        ValueError, match="exceeding the 256 Mpx read cap"
    ):
        _overlay(mask)

    assert reader.read_count == 0


def test_coordinate_preview_refuses_a_native_read_above_the_pixel_cap_before_decoding(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(visualization_mod, "WSI", _FakeWSI)
    reader = _FakeMaskReader(dimensions=(20_000, 20_000))

    with _served_mask(monkeypatch, reader) as mask, pytest.raises(
        ValueError, match="exceeding the 256 Mpx read cap"
    ):
        _coordinate_preview(mask, tmp_path / "preview")

    assert reader.read_count == 0


def test_overlay_decode_failure_names_the_mask_path_and_backend(monkeypatch):
    monkeypatch.setattr(visualization_mod, "WSI", _FakeWSI)
    reader = _FakeMaskReader(
        np.zeros((6, 6), dtype=np.uint8), decode_error=RuntimeError("codec unavailable")
    )

    with _served_mask(monkeypatch, reader) as mask, pytest.raises(
        RuntimeError, match="codec unavailable"
    ) as excinfo:
        _overlay(mask)

    assert "path=/masks/served.tif with backend=openslide" in str(excinfo.value)


# --- tiling-preview worker -----------------------------------------------------------


def _artifact(annotation: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        coordinates_npz_path=Path("slide.npz"),
        coordinates_meta_path=Path("slide.json"),
        annotation=annotation,
    )


def _patch_preview_writer(monkeypatch, *, error: Exception | None = None) -> list[dict]:
    """Capture ``write_annotation_tiling_preview`` calls (or fail them with ``error``)."""
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return Path("preview.jpg")

    monkeypatch.setattr(orchmod, "load_tiling_result", lambda npz, meta: "result")
    monkeypatch.setattr(orchmod, "write_annotation_tiling_preview", _fake)
    return calls


def _run_worker(*, annotation: str | None, mask_path=Path("/masks/served.tif")):
    return orchmod._write_annotation_tiling_preview_from_artifacts(
        artifact=_artifact(annotation),
        output_dir=Path("out"),
        downsample=1,
        mask_path=mask_path,
        mask_backend="openslide",
        pixel_mapping=PIXEL_MAPPING,
        color_mapping=COLOR_MAPPING,
    )


def test_worker_builds_a_full_vocabulary_mask_and_restricts_only_the_rendering_mappings(
    monkeypatch,
):
    reader = _FakeMaskReader(np.zeros((6, 6), dtype=np.uint8))
    monkeypatch.setattr(maskmod, "open_slide", lambda path, backend, **kwargs: reader)
    calls = _patch_preview_writer(monkeypatch)

    assert _run_worker(annotation="tumor") == Path("preview.jpg")

    (call,) = calls
    mask = call["mask"]
    assert isinstance(mask, Mask)
    assert (mask.path, mask.backend) == (Path("/masks/served.tif"), "openslide")
    assert mask.labels == AnnotationLabels(pixel_mapping=PIXEL_MAPPING)
    assert call["pixel_mapping"] == {"tumor": 1}
    assert call["color_mapping"] == {"tumor": [255, 0, 0]}
    assert reader.close_count == 1


def test_worker_closes_its_mask_when_the_preview_fails(monkeypatch):
    reader = _FakeMaskReader(np.zeros((6, 6), dtype=np.uint8))
    monkeypatch.setattr(maskmod, "open_slide", lambda path, backend, **kwargs: reader)
    _patch_preview_writer(monkeypatch, error=RuntimeError("render failed"))

    with pytest.raises(RuntimeError, match="render failed"):
        _run_worker(annotation="tumor")

    assert reader.close_count == 1


def test_worker_renders_without_a_mask_when_the_slide_has_none(monkeypatch):
    monkeypatch.setattr(
        maskmod,
        "open_slide",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no mask to open")),
    )
    calls = _patch_preview_writer(monkeypatch)

    _run_worker(annotation="tumor", mask_path=None)

    assert calls[0]["mask"] is None


# --- orchestration ---------------------------------------------------------------------


def _write_fixture_slide(tmp_path: Path) -> SlideSpec:
    """A 256 px openslide-readable RGB slide at 0.5 um/px with a 4-label PNG mask: tumor
    (1) top-left, stroma (2) top-right, necrosis (3) bottom-right."""
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("openslide")
    image_path = tmp_path / "slide.tif"
    pixels_per_cm = 1e4 / 0.5
    tifffile.imwrite(
        image_path,
        np.full((256, 256, 3), 200, dtype=np.uint8),
        tile=(64, 64),
        photometric="rgb",
        resolution=(pixels_per_cm, pixels_per_cm),
        resolutionunit="CENTIMETER",
    )
    labels = np.zeros((256, 256), dtype=np.uint8)
    labels[:128, :128] = 1
    labels[:128, 128:] = 2
    labels[128:, 128:] = 3
    mask_path = tmp_path / "mask.png"
    Image.fromarray(labels, mode="L").save(mask_path)
    return SlideSpec(sample_id="slide-1", image_path=image_path, mask_path=mask_path)


def _sampling() -> SamplingSpec:
    return SamplingSpec(
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2, "necrosis": 3},
        color_mapping={
            "background": None,
            "tumor": [255, 0, 0],
            "stroma": [0, 0, 255],
            "necrosis": [0, 255, 0],
        },
        tissue_percentage={"background": None, "tumor": 0.5, "stroma": 0.5, "necrosis": 0.5},
        active_annotations=("tumor", "stroma", "necrosis"),
    )


@pytest.mark.parametrize("num_workers", [1, 2], ids=["inline", "process-pool"])
def test_tile_slides_renders_per_annotation_previews_of_a_multi_label_mask(
    tmp_path, num_workers
):
    whole_slide = _write_fixture_slide(tmp_path)
    output_dir = tmp_path / "out"

    artifacts = tile_slides(
        [whole_slide],
        tiling=TilingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=64,
            tolerance=0.05,
            overlap=0.0,
            min_coverage={"tissue": 0.5},
            backend="openslide",
            mask_backend="pil",
        ),
        preview=PreviewConfig(save_tiling_preview=True, downsample=1),
        output_dir=output_dir,
        num_workers=num_workers,
        sampling=_sampling(),
        selection_strategy=CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        output_mode=CoordinateOutputMode.PER_ANNOTATION,
    )

    previews = {artifact.annotation: artifact.tiling_preview_path for artifact in artifacts}
    assert previews == {
        "tumor": output_dir / "preview" / "tiling" / "tumor" / "slide-1.jpg",
        "stroma": output_dir / "preview" / "tiling" / "stroma" / "slide-1.jpg",
        "necrosis": output_dir / "preview" / "tiling" / "necrosis" / "slide-1.jpg",
    }
    with Image.open(previews["tumor"]) as saved:
        arr = np.array(saved.convert("RGB")).astype(int)
    # Only tumor is painted on its preview: its tile is red-dominant, stroma stays gray.
    assert arr[32, 32, 0] > arr[32, 32, 1] + 80
    assert np.abs(arr[32, 160] - 200).max() <= 8


def test_tile_slides_submits_only_plain_values_to_the_preview_executor(
    monkeypatch, tmp_path
):
    whole_slide = _write_fixture_slide(tmp_path)
    submitted: list[dict] = []

    class _RecordingExecutor(orchmod._InlineExecutor):
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def submit(self, fn, **kwargs):
            submitted.append(kwargs)
            return super().submit(fn, **kwargs)

    monkeypatch.setattr(orchmod, "ProcessPoolExecutor", _RecordingExecutor)

    tile_slides(
        [whole_slide],
        tiling=TilingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=64,
            tolerance=0.05,
            overlap=0.0,
            min_coverage={"tissue": 0.5},
            backend="openslide",
            mask_backend="auto",
        ),
        preview=PreviewConfig(save_tiling_preview=True, downsample=1),
        output_dir=tmp_path / "out",
        num_workers=2,
        sampling=_sampling(),
        output_mode=CoordinateOutputMode.PER_ANNOTATION,
    )

    assert len(submitted) == 3
    for kwargs in submitted:
        assert kwargs["mask_path"] == whole_slide.mask_path
        # The resolved concrete backend and the full vocabulary, not the restricted one.
        assert kwargs["mask_backend"] == "pil"
        assert kwargs["pixel_mapping"] == _sampling().pixel_mapping
        assert not any(isinstance(v, (Mask, AlignedMask)) for v in kwargs.values())
        pickle.dumps(kwargs)
