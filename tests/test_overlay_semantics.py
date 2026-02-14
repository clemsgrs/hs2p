from pathlib import Path

import numpy as np
import pytest
from PIL import Image


cv2 = pytest.importorskip("cv2")
wsi_mod = pytest.importorskip("hs2p.wsi")


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
    assert np.array_equal(overlay_arr[1, 0], tile_arr[1, 0])  # uncolored label untouched
    assert not np.array_equal(overlay_arr[0, 1], tile_arr[0, 1])  # colored label blended


def test_overlay_mask_on_slide_matches_tile_semantics(monkeypatch):
    slide_arr = np.full((2, 2, 3), 120, dtype=np.uint8)
    mask_labels = np.array([[0, 3], [4, 3]], dtype=np.uint8)
    mask_arr = np.stack([mask_labels, mask_labels, mask_labels], axis=-1)

    class FakeWSI:
        def __init__(self, path, backend="asap"):
            self.path = Path(path)
            self.spacings = [0.5]
            self.level_dimensions = [(2, 2)]
            self.level_downsamples = [(1.0, 1.0)]

        def get_best_level_for_downsample_custom(self, downsample):
            return 0

        def get_slide(self, spacing):
            if "mask" in self.path.name:
                return mask_arr
            return slide_arr

    monkeypatch.setattr(wsi_mod, "WholeSlideImage", FakeWSI)

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
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        alpha=0.5,
    )
    overlay_arr = np.array(overlay.convert("RGB"))

    assert np.array_equal(overlay_arr[0, 0], slide_arr[0, 0])  # background untouched
    assert np.array_equal(overlay_arr[1, 0], slide_arr[1, 0])  # uncolored label untouched
    assert not np.array_equal(overlay_arr[0, 1], slide_arr[0, 1])  # colored label blended
