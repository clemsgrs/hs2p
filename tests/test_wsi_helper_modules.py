import numpy as np
from PIL import Image

import hs2p.wsi as wsi_pkg
from hs2p.wsi import masks as masks_mod
from hs2p.wsi import preview as preview_mod
from hs2p.wsi.masks import (
    compose_overlay_mask_from_annotations,
    extract_padded_crop,
    normalize_tissue_mask,
    pad_array_to_shape,
)
from hs2p.wsi.preview import build_overlay_alpha, build_palette


def test_wsi_package_reexports_helper_functions_directly():
    assert wsi_pkg._normalize_tissue_mask is masks_mod.normalize_tissue_mask
    assert wsi_pkg._read_aligned_mask is masks_mod.read_aligned_mask
    assert wsi_pkg._compose_overlay_mask_from_annotations is masks_mod.compose_overlay_mask_from_annotations
    assert wsi_pkg.overlay_mask_on_tile is preview_mod.overlay_mask_on_tile
    assert wsi_pkg.draw_grid_from_coordinates is preview_mod.draw_grid_from_coordinates
    assert wsi_pkg.pad_to_patch_size is preview_mod.pad_to_patch_size


def test_masks_normalize_and_compose_overlay_mask():
    tissue = np.array(
        [
            [[0], [255]],
            [[255], [255]],
        ],
        dtype=np.uint8,
    )
    tumor = np.array(
        [
            [0, 0],
            [255, 0],
        ],
        dtype=np.uint8,
    )

    normalized = normalize_tissue_mask(tissue)
    overlay = compose_overlay_mask_from_annotations(
        annotation_mask={"tissue": tissue, "tumor": tumor},
        pixel_mapping={"background": 0, "tissue": 1, "tumor": 2},
    )

    np.testing.assert_array_equal(
        normalized,
        np.array(
            [
                [0, 1],
                [1, 1],
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_array_equal(
        overlay,
        np.array(
            [
                [0, 1],
                [2, 1],
            ],
            dtype=np.uint8,
        ),
    )


def test_masks_pad_and_extract_padded_crop_preserve_existing_pixels():
    arr = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=np.uint8,
    )

    padded = pad_array_to_shape(arr, target_width=4, target_height=3)
    crop = extract_padded_crop(arr, x=1, y=1, width=3, height=2)

    np.testing.assert_array_equal(
        padded,
        np.array(
            [
                [1, 2, 0, 0],
                [3, 4, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_array_equal(
        crop,
        np.array(
            [
                [4, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_preview_build_overlay_alpha_only_uses_colored_labels():
    palette = build_palette(
        pixel_mapping={"background": 0, "tissue": 1, "tumor": 2},
        color_mapping={"background": None, "tissue": [10, 20, 30], "tumor": None},
    )
    mask = np.array(
        [
            [0, 1],
            [2, 1],
        ],
        dtype=np.uint8,
    )

    alpha = build_overlay_alpha(
        mask_arr=mask,
        alpha=0.5,
        pixel_mapping={"background": 0, "tissue": 1, "tumor": 2},
        color_mapping={"background": None, "tissue": [10, 20, 30], "tumor": None},
    )

    assert palette[3:6].tolist() == [10, 20, 30]
    assert palette[6:9].tolist() == [0, 0, 0]
    assert isinstance(alpha, Image.Image)
    np.testing.assert_array_equal(
        np.array(alpha),
        np.array(
            [
                [255, 127],
                [255, 127],
            ],
            dtype=np.uint8,
        ),
    )
