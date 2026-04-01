from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")
wsi_mod = pytest.importorskip("hs2p.wsi")
coordinate_api_mod = pytest.importorskip("hs2p.wsi.api")
visualization_mod = pytest.importorskip("hs2p.wsi.visualization")


def _build_palette(mapping: dict[int, tuple[int, int, int]]) -> np.ndarray:
    palette = np.zeros(shape=768, dtype=int)
    for label, color in mapping.items():
        palette[label * 3 : label * 3 + 3] = np.array(color, dtype=int)
    return palette


def test_overlay_mask_on_tile_only_colored_labels_are_blended():
    tile_arr = np.full((2, 2, 3), 120, dtype=np.uint8)
    tile = Image.fromarray(tile_arr)
    mask_arr = np.array([[0, 3], [4, 3]], dtype=np.uint8)
    mask = Image.fromarray(mask_arr)

    pixel_mapping = {"background": 0, "gleason3": 3, "gleason4": 4}
    color_mapping = {
        "background": None,
        "gleason3": [255, 0, 0],
        "gleason4": None,
    }
    palette = _build_palette({3: (255, 0, 0)})

    overlay = wsi_mod.overlay_mask_on_tile(
        tile=tile,
        mask=mask,
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        alpha=0.5,
    )
    overlay_arr = np.array(overlay)

    assert np.array_equal(overlay_arr[0, 0], tile_arr[0, 0])  # background untouched
    assert np.array_equal(
        overlay_arr[1, 0], tile_arr[1, 0]
    )  # uncolored label untouched
    assert not np.array_equal(
        overlay_arr[0, 1], tile_arr[0, 1]
    )  # colored label blended


def test_overlay_mask_on_slide_matches_tile_semantics(monkeypatch):
    slide_arr = np.full((2, 2, 3), 120, dtype=np.uint8)
    mask_labels = np.array([[0, 3], [4, 3]], dtype=np.uint8)
    mask_arr = np.stack([mask_labels, mask_labels, mask_labels], axis=-1)

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
            self.level_dimensions = [(2, 2)]
            self.level_downsamples = [(1.0, 1.0)]
            self.reader = FakeReader()

        def get_best_level_for_downsample_custom(self, downsample):
            return 0

        def get_level_spacing(self, level):
            del level
            return 0.5

        def get_slide(self, level):
            del level
            return slide_arr

    monkeypatch.setattr(coordinate_api_mod, "WSI", FakeWSI)
    monkeypatch.setattr(visualization_mod, "WSI", FakeWSI)

    pixel_mapping = {"background": 0, "gleason3": 3, "gleason4": 4}
    color_mapping = {
        "background": None,
        "gleason3": [255, 0, 0],
        "gleason4": None,
    }
    palette = _build_palette({3: (255, 0, 0)})

    overlay = wsi_mod.overlay_mask_on_slide(
        wsi_path=Path("fake-wsi.tif"),
        annotation_mask_path=Path("fake-mask.tif"),
        downsample=1,
        backend="openslide",
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        alpha=0.5,
    )
    overlay_arr = np.array(overlay.convert("RGB"))

    assert np.array_equal(overlay_arr[0, 0], slide_arr[0, 0])  # background untouched
    assert np.array_equal(
        overlay_arr[1, 0], slide_arr[1, 0]
    )  # uncolored label untouched
    assert not np.array_equal(
        overlay_arr[0, 1], slide_arr[0, 1]
    )  # colored label blended


def test_overlay_mask_on_slide_accepts_in_memory_mask_array(monkeypatch):
    slide_arr = np.full((2, 2, 3), 120, dtype=np.uint8)
    mask_arr = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    class FakeWSI:
        def __init__(self, path, backend="asap"):
            del backend
            self.path = Path(path)
            self.spacings = [0.5]
            self.level_dimensions = [(2, 2)]
            self.level_downsamples = [(1.0, 1.0)]

        def get_best_level_for_downsample_custom(self, downsample):
            return 0

        def get_slide(self, level):
            del level
            return slide_arr

    monkeypatch.setattr(coordinate_api_mod, "WSI", FakeWSI)
    monkeypatch.setattr(visualization_mod, "WSI", FakeWSI)

    pixel_mapping = {"background": 0, "tissue": 1}
    color_mapping = {"background": None, "tissue": [157, 219, 129]}
    palette = _build_palette({1: (157, 219, 129)})

    overlay = wsi_mod.overlay_mask_on_slide(
        wsi_path=Path("fake-wsi.tif"),
        annotation_mask_path=None,
        mask_arr=mask_arr,
        downsample=1,
        backend="openslide",
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        alpha=0.5,
    )
    overlay_arr = np.array(overlay.convert("RGB"))

    assert np.array_equal(overlay_arr[0, 0], slide_arr[0, 0])
    assert not np.array_equal(overlay_arr[0, 1], slide_arr[0, 1])


def test_overlay_mask_on_slide_defaults_to_tissue_overlay_style(monkeypatch):
    slide_arr = np.full((2, 2, 3), 120, dtype=np.uint8)
    mask_arr = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    class FakeWSI:
        def __init__(self, path, backend="asap"):
            del backend
            self.path = Path(path)
            self.spacings = [0.5]
            self.level_dimensions = [(2, 2)]
            self.level_downsamples = [(1.0, 1.0)]

        def get_best_level_for_downsample_custom(self, downsample):
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

    assert np.array_equal(overlay_arr[0, 0], slide_arr[0, 0])
    assert not np.array_equal(overlay_arr[0, 1], slide_arr[0, 1])

def test_extract_coordinates_uses_overlay_mask_preview_instead_of_line_rendering(
    monkeypatch, tmp_path: Path
):
    preview_calls = []

    class FakeWSI:
        def __init__(
            self,
            path,
            backend="asap",
            mask_path=None,
            spacing_at_level_0=None,
            segment=False,
            segment_params=None,
            sampling_spec=None,
            pixel_mapping=None,
        ):
            del (
                backend,
                mask_path,
                spacing_at_level_0,
                segment,
                segment_params,
                sampling_spec,
                pixel_mapping,
            )
            self.path = Path(path)
            self.spacings = [0.5]
            self.level_dimensions = [(2, 2)]
            self.level_downsamples = [(1.0, 1.0)]
            self.annotation_mask = {
                "tissue": np.array([[0, 255], [255, 0]], dtype=np.uint8)
            }

        def get_tile_coordinates(
            self,
            tiling_params,
            filter_params,
            annotation=None,
            disable_tqdm=False,
            num_workers=1,
        ):
            return (
                [(0, 0)],
                [1.0],
                [0],
                0,
                1.0,
                224,
            )

        def get_level_spacing(self, level):
            return self.spacings[level]

        def visualize_mask(self, *args, **kwargs):
            raise AssertionError("line-based visualize_mask should not be used")

    def _fake_overlay_mask_on_slide(**kwargs):
        preview_calls.append(kwargs)
        return Image.fromarray(np.full((2, 2, 3), 200, dtype=np.uint8))

    monkeypatch.setattr(coordinate_api_mod, "WSI", FakeWSI)
    monkeypatch.setattr(visualization_mod, "WSI", FakeWSI)
    monkeypatch.setattr(visualization_mod, "overlay_mask_on_slide", _fake_overlay_mask_on_slide)

    preview_path = tmp_path / "mask-preview.jpg"
    result = wsi_mod.extract_coordinates(
        wsi_path=Path("fake-wsi.tif"),
        tissue_mask_path=None,
        backend="openslide",
        segment_params=SimpleNamespace(
            downsample=64,
            sthresh=8,
            sthresh_up=255,
            mthresh=7,
            close=4,
            use_otsu=False,
            use_hsv=True,
        ),
        tiling_params=SimpleNamespace(
            target_spacing_um=0.5,
            tolerance=0.05,
            target_tile_size_px=224,
            overlap=0.0,
            tissue_threshold=0.1,
            use_padding=True,
        ),
        filter_params=SimpleNamespace(
            ref_tile_size=16,
            a_t=4,
            a_h=2,
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
        mask_preview_path=preview_path,
        disable_tqdm=True,
        num_workers=1,
    )

    assert result.coordinates == [(0, 0)]
    assert preview_path.is_file()
    assert len(preview_calls) == 1
    np.testing.assert_array_equal(
        preview_calls[0]["mask_arr"], np.array([[0, 1], [1, 0]], dtype=np.uint8)
    )
    assert preview_calls[0]["annotation_mask_path"] is None


def test_extract_coordinates_preview_uses_in_memory_annotation_labels_when_style_is_provided(
    monkeypatch, tmp_path: Path
):
    preview_calls = []

    class FakeWSI:
        def __init__(
            self,
            path,
            backend="asap",
            mask_path=None,
            spacing_at_level_0=None,
            segment=False,
            segment_params=None,
            sampling_spec=None,
            pixel_mapping=None,
        ):
            del (
                backend,
                mask_path,
                spacing_at_level_0,
                segment,
                segment_params,
                sampling_spec,
                pixel_mapping,
            )
            self.path = Path(path)
            self.spacings = [0.5]
            self.level_dimensions = [(2, 2)]
            self.level_downsamples = [(1.0, 1.0)]
            self.annotation_mask = {
                "tissue": np.array([[0, 255], [255, 0]], dtype=np.uint8),
                "tumor": np.array([[0, 255], [255, 0]], dtype=np.uint8),
            }

        def get_tile_coordinates(
            self,
            tiling_params,
            filter_params,
            annotation=None,
            disable_tqdm=False,
            num_workers=1,
        ):
            return (
                [(0, 0)],
                [1.0],
                [0],
                0,
                1.0,
                224,
            )

        def get_level_spacing(self, level):
            return self.spacings[level]

    def _fake_overlay_mask_on_slide(**kwargs):
        preview_calls.append(kwargs)
        return Image.fromarray(np.full((2, 2, 3), 200, dtype=np.uint8))

    monkeypatch.setattr(coordinate_api_mod, "WSI", FakeWSI)
    monkeypatch.setattr(visualization_mod, "WSI", FakeWSI)
    monkeypatch.setattr(visualization_mod, "overlay_mask_on_slide", _fake_overlay_mask_on_slide)

    preview_path = tmp_path / "mask-preview.jpg"
    palette = _build_palette({1: (255, 0, 0)})
    result = wsi_mod.extract_coordinates(
        wsi_path=Path("fake-wsi.tif"),
        tissue_mask_path=Path("fake-mask.tif"),
        backend="openslide",
        segment_params=SimpleNamespace(
            downsample=64,
            sthresh=8,
            sthresh_up=255,
            mthresh=7,
            close=4,
            use_otsu=False,
            use_hsv=True,
        ),
        tiling_params=SimpleNamespace(
            target_spacing_um=0.5,
            tolerance=0.05,
            target_tile_size_px=224,
            overlap=0.0,
            tissue_threshold=0.1,
            use_padding=True,
        ),
        filter_params=SimpleNamespace(
            ref_tile_size=16,
            a_t=4,
            a_h=2,
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
        sampling_spec=SimpleNamespace(
            pixel_mapping={"background": 0, "tumor": 1},
            color_mapping={"background": None, "tumor": [255, 0, 0]},
            tissue_percentage={"background": None, "tumor": 0.1},
            active_annotations=("tumor",),
        ),
        mask_preview_path=preview_path,
        preview_downsample=8,
        preview_palette=palette,
        preview_pixel_mapping={"background": 0, "tumor": 1},
        preview_color_mapping={"background": None, "tumor": [255, 0, 0]},
        disable_tqdm=True,
        num_workers=1,
    )

    assert result.coordinates == [(0, 0)]
    assert preview_path.is_file()
    assert len(preview_calls) == 1
    np.testing.assert_array_equal(
        preview_calls[0]["mask_arr"], np.array([[0, 1], [1, 0]], dtype=np.uint8)
    )
    assert preview_calls[0]["annotation_mask_path"] is None
    assert preview_calls[0]["downsample"] == 8


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
