from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")
wsi_mod = pytest.importorskip("hs2p.wsi")
visualization_mod = pytest.importorskip("hs2p.wsi.visualization")


def _contour(points: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))


def _build_palette(mapping: dict[int, tuple[int, int, int]]) -> np.ndarray:
    palette = np.zeros(shape=768, dtype=int)
    for label, color in mapping.items():
        palette[label * 3 : label * 3 + 3] = np.array(color, dtype=int)
    return palette


def test_overlay_mask_on_slide_renders_outer_and_hole_contours(monkeypatch):
    slide_arr = np.full((120, 120, 3), 240, dtype=np.uint8)
    mask_arr = np.zeros((120, 120), dtype=np.uint8)
    mask_arr[40:80, 40:80] = 1
    mask_arr[52:68, 52:68] = 0
    contours = SimpleNamespace(
        contours=[_contour([(40, 40), (40, 79), (79, 79), (79, 40)])],
        holes=[[_contour([(52, 52), (52, 67), (67, 67), (67, 52)])]],
    )

    class FakeReader:
        spacings = [0.5]
        level_downsamples = [(1.0, 1.0)]

        def read_level(self, level):
            del level
            return mask_arr

    class FakeWSI:
        def __init__(self, path, backend="asap"):
            del backend
            self.path = Path(path)
            self.spacings = [0.5]
            self.level_dimensions = [(20, 20)]
            self.level_downsamples = [(1.0, 1.0)]
            self.reader = FakeReader()

        def get_best_level_for_downsample_custom(self, downsample):
            del downsample
            return 0

        def get_level_spacing(self, level):
            del level
            return 0.5

        def get_slide(self, level):
            del level
            return slide_arr

    monkeypatch.setattr(visualization_mod, "WSI", FakeWSI)

    overlay = wsi_mod.overlay_mask_on_slide(
        wsi_path=Path("fake-wsi.tif"),
        annotation_mask_path=None,
        downsample=1,
        backend="openslide",
        mask_arr=mask_arr,
        contours=contours,
    )
    overlay_arr = np.array(overlay.convert("RGB"))

    assert np.array_equal(overlay_arr[0, 119], slide_arr[0, 119])
    assert np.any(np.all(overlay_arr == np.array([37, 94, 59], dtype=np.uint8), axis=-1))
    assert np.any(np.all(overlay_arr == np.array([242, 107, 58], dtype=np.uint8), axis=-1))


def test_overlay_mask_on_slide_scales_level_zero_contours_to_vis_level(monkeypatch):
    slide_arr = np.full((5, 5, 3), 240, dtype=np.uint8)
    contours = SimpleNamespace(
        contours=[_contour([(4, 4), (4, 8), (8, 8), (8, 4)])],
        holes=[[]],
    )

    class FakeWSI:
        def __init__(self, path, backend="asap"):
            del backend
            self.path = Path(path)
            self.spacings = [0.5]
            self.level_dimensions = [(10, 10), (5, 5)]
            self.level_downsamples = [(1.0, 1.0), (2.0, 2.0)]

        def get_best_level_for_downsample_custom(self, downsample):
            del downsample
            return 1

        def get_slide(self, level):
            assert level == 1
            return slide_arr

    monkeypatch.setattr(visualization_mod, "WSI", FakeWSI)

    overlay = wsi_mod.overlay_mask_on_slide(
        wsi_path=Path("fake-wsi.tif"),
        annotation_mask_path=None,
        downsample=1,
        backend="openslide",
        mask_arr=np.zeros((5, 5), dtype=np.uint8),
        contours=contours,
        stroke_thickness=1,
    )
    overlay_arr = np.array(overlay.convert("RGB"))

    assert np.array_equal(overlay_arr[2, 2], np.array([37, 94, 59], dtype=np.uint8))


def test_resolve_stroke_thickness_scales_with_requested_downsample():
    assert visualization_mod._resolve_stroke_thickness(
        level_downsample=16,
        stroke_thickness=None,
    ) == 4
    assert visualization_mod._resolve_stroke_thickness(
        level_downsample=16,
        stroke_thickness=None,
    ) == 4
    assert visualization_mod._resolve_stroke_thickness(
        level_downsample=32,
        stroke_thickness=None,
    ) == 2
    assert visualization_mod._resolve_stroke_thickness(
        level_downsample=16,
        stroke_thickness=5,
    ) == 5


def test_overlay_mask_on_slide_accepts_in_memory_mask_array(monkeypatch):
    slide_arr = np.full((120, 120, 3), 120, dtype=np.uint8)
    mask_arr = np.zeros((120, 120), dtype=np.uint8)
    mask_arr[40:80, 40:80] = 1

    class FakeWSI:
        def __init__(self, path, backend="asap"):
            del backend
            self.path = Path(path)
            self.spacings = [0.5]
            self.level_dimensions = [(120, 120)]
            self.level_downsamples = [(1.0, 1.0)]

        def get_best_level_for_downsample_custom(self, downsample):
            del downsample
            return 0

        def get_slide(self, level):
            del level
            return slide_arr

    monkeypatch.setattr(visualization_mod, "WSI", FakeWSI)

    overlay = wsi_mod.overlay_mask_on_slide(
        wsi_path=Path("fake-wsi.tif"),
        annotation_mask_path=None,
        mask_arr=mask_arr,
        downsample=1,
        backend="openslide",
    )
    overlay_arr = np.array(overlay.convert("RGB"))

    assert np.array_equal(overlay_arr[0, 119], slide_arr[0, 119])
    assert np.any(np.all(overlay_arr == np.array([37, 94, 59], dtype=np.uint8), axis=-1))


def test_overlay_mask_on_slide_defaults_to_tissue_overlay_style(monkeypatch):
    slide_arr = np.full((120, 120, 3), 120, dtype=np.uint8)
    mask_arr = np.zeros((120, 120), dtype=np.uint8)
    mask_arr[40:80, 40:80] = 1

    class FakeWSI:
        def __init__(self, path, backend="asap"):
            del backend
            self.path = Path(path)
            self.spacings = [0.5]
            self.level_dimensions = [(120, 120)]
            self.level_downsamples = [(1.0, 1.0)]

        def get_best_level_for_downsample_custom(self, downsample):
            del downsample
            return 0

        def get_slide(self, level):
            del level
            return slide_arr

    monkeypatch.setattr(visualization_mod, "WSI", FakeWSI)

    overlay = wsi_mod.overlay_mask_on_slide(
        wsi_path=Path("fake-wsi.tif"),
        annotation_mask_path=None,
        mask_arr=mask_arr,
        downsample=1,
        backend="openslide",
    )
    overlay_arr = np.array(overlay.convert("RGB"))

    assert np.array_equal(overlay_arr[0, 119], slide_arr[0, 119])
    assert np.any(np.all(overlay_arr == np.array([37, 94, 59], dtype=np.uint8), axis=-1))


def test_save_overlay_preview_writes_rgba_overlay_to_jpeg(monkeypatch, tmp_path: Path):
    overlay = Image.fromarray(
        np.array(
            [
                [[10, 20, 30, 0], [40, 50, 60, 255]],
                [[70, 80, 90, 128], [100, 110, 120, 255]],
            ],
            dtype=np.uint8,
        ),
        mode="RGBA",
    )

    monkeypatch.setattr(
        visualization_mod,
        "overlay_mask_on_slide",
        lambda **kwargs: overlay,
    )

    preview_path = tmp_path / "mask-preview.jpg"
    visualization_mod.save_overlay_preview(
        wsi_path=Path("fake-wsi.tif"),
        backend="openslide",
        mask_arr=np.zeros((2, 2), dtype=np.uint8),
        mask_preview_path=preview_path,
        downsample=1,
    )

    assert preview_path.is_file()
    with Image.open(preview_path) as saved:
        assert saved.mode == "RGB"


def test_draw_grid_from_coordinates_crops_loaded_canvas_instead_of_fetching_tiles():
    class FakeWSI:
        level_downsamples = [(1.0, 1.0)]
        level_dimensions = [(2, 2)]
        spacings = [1.0]

        def get_level_spacing(self, level):
            assert level == 0
            return 1.0

        def get_tile(self, *args, **kwargs):
            raise AssertionError(
                "draw_grid_from_coordinates should crop from the loaded canvas"
            )

    canvas = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )

    image = wsi_mod.draw_grid_from_coordinates(
        canvas.copy(),
        FakeWSI(),
        coords=[(0, 0)],
        tile_size_at_0=(1, 1),
        vis_level=0,
        thickness=0,
        indices=None,
        mask=None,
    )

    rendered = np.array(image)
    assert rendered.shape == canvas.shape
