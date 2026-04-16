import numpy as np
from PIL import Image

import hs2p.wsi as wsi_pkg
from hs2p.wsi import api as wsi_api_mod
from hs2p.wsi import masks as masks_mod
from hs2p.wsi import preview as preview_mod
from hs2p.wsi import types as wsi_types_mod
from hs2p.wsi import visualization as visualization_mod
from hs2p.wsi.masks import (
    compose_overlay_mask_from_annotations,
    extract_padded_crop,
    normalize_tissue_mask,
    pad_array_to_shape,
)
from hs2p.wsi.preview import build_overlay_alpha, build_palette


def test_wsi_package_reexports_helper_functions_directly():
    assert wsi_pkg.normalize_tissue_mask is masks_mod.normalize_tissue_mask
    assert wsi_pkg.read_aligned_mask is masks_mod.read_aligned_mask
    assert (
        wsi_pkg.compose_overlay_mask_from_annotations
        is masks_mod.compose_overlay_mask_from_annotations
    )
    assert wsi_pkg.build_overlay_alpha is preview_mod.build_overlay_alpha
    assert wsi_pkg.build_palette is preview_mod.build_palette
    assert wsi_pkg.overlay_mask_on_tile is preview_mod.overlay_mask_on_tile
    assert wsi_pkg.draw_grid_from_coordinates is preview_mod.draw_grid_from_coordinates
    assert wsi_pkg.pad_to_patch_size is preview_mod.pad_to_patch_size
    assert wsi_pkg.overlay_mask_on_slide is visualization_mod.overlay_mask_on_slide
    assert wsi_pkg.write_coordinate_preview is visualization_mod.write_coordinate_preview


def test_wsi_package_reexports_coordinate_api_from_owner_module():
    assert wsi_pkg.CoordinateExtractionResult is wsi_api_mod.CoordinateExtractionResult
    assert wsi_pkg.UnifiedCoordinateRequest is wsi_api_mod.UnifiedCoordinateRequest
    assert wsi_pkg.UnifiedCoordinateResponse is wsi_api_mod.UnifiedCoordinateResponse
    assert wsi_pkg.CoordinateSelectionStrategy is wsi_types_mod.CoordinateSelectionStrategy
    assert wsi_pkg.CoordinateOutputMode is wsi_types_mod.CoordinateOutputMode
    assert wsi_pkg.SamplingSpec is wsi_types_mod.SamplingSpec
    assert wsi_pkg.extract_coordinates is wsi_api_mod.extract_coordinates
    assert wsi_pkg.sample_coordinates is wsi_api_mod.sample_coordinates


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


def test_preview_pad_to_patch_size_handles_numpy_masks_and_pil_images():
    image = Image.fromarray(np.array([[1, 2, 3]], dtype=np.uint8))
    array = np.array([[1, 2, 3]], dtype=np.uint8)

    padded_image = preview_mod.pad_to_patch_size(image, (2, 2))
    padded_array = preview_mod.pad_to_patch_size(array, (2, 2), fill=0)

    assert padded_image.size == (4, 2)
    np.testing.assert_array_equal(
        padded_array,
        np.array(
            [
                [1, 2, 3, 0],
                [0, 0, 0, 0],
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
