from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps

from hs2p.wsi.masks import extract_padded_crop, read_aligned_mask


def build_palette(
    *,
    pixel_mapping: dict[str, int],
    color_mapping: dict[str, list[int] | None],
) -> np.ndarray:
    palette = np.zeros(shape=768, dtype=np.uint8)
    for annotation, label_value in pixel_mapping.items():
        color = color_mapping.get(annotation)
        if color is None:
            continue
        palette[label_value * 3 : label_value * 3 + 3] = np.asarray(
            color,
            dtype=np.uint8,
        )
    return palette


def build_overlay_alpha(
    *,
    mask_arr: np.ndarray,
    alpha: float,
    pixel_mapping: dict[str, int],
    color_mapping: dict[str, list[int] | None],
) -> Image.Image:
    alpha_int = int(round(255 * alpha))
    active_labels = set()
    for annotation, label_value in pixel_mapping.items():
        if color_mapping.get(annotation) is not None:
            active_labels.add(label_value)

    overlay_mask = np.isin(mask_arr, list(active_labels)).astype("uint8")
    alpha_content = np.less(overlay_mask, 1).astype("uint8") * alpha_int + (
        255 - alpha_int
    )
    return Image.fromarray(alpha_content)


def overlay_mask_on_tile(
    tile: Image.Image,
    mask: Image.Image,
    palette: np.ndarray,
    pixel_mapping: dict[str, int],
    color_mapping: dict[str, list[int] | None],
    alpha: float = 0.5,
):
    mask_arr = np.array(mask)
    alpha_content = build_overlay_alpha(
        mask_arr=mask_arr,
        alpha=alpha,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )
    mask.putpalette(data=palette.tolist())
    mask_rgb = mask.convert(mode="RGB")
    return Image.composite(image1=tile, image2=mask_rgb, mask=alpha_content)


def draw_grid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(
        img,
        tuple(np.maximum([0, 0], coord - thickness // 2)),
        tuple(coord - thickness // 2 + np.array(shape)),
        color,
        thickness=thickness,
    )
    return img


def draw_grid_from_coordinates(
    canvas,
    wsi,
    coords,
    tile_size_at_0,
    vis_level: int,
    thickness: int = 2,
    indices: list[int] | None = None,
    mask=None,
    palette: np.ndarray | None = None,
    pixel_mapping: dict[str, int] | None = None,
    color_mapping: dict[str, list[int] | None] | None = None,
):
    downsamples = wsi.level_downsamples[vis_level]
    source_canvas = canvas[:, :, :3].copy()
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)

    tile_size = tuple(
        np.ceil((np.array(tile_size_at_0) / np.array(downsamples))).astype(np.int32)
    )
    wsi_width_at_0, wsi_height_at_0 = wsi.level_dimensions[0]

    vis_spacing = wsi.get_level_spacing(vis_level)
    aligned_mask = None
    if mask is not None:
        aligned_mask = read_aligned_mask(
            mask_obj=mask,
            slide_spacing=vis_spacing,
            slide_dimensions=wsi.level_dimensions[vis_level],
        )
        if aligned_mask.ndim == 3 and aligned_mask.shape[-1] == 1:
            aligned_mask = np.squeeze(aligned_mask, axis=-1)

    for idx in range(total):
        tile_id = indices[idx]
        coord = coords[tile_id]
        x, y = coord

        width, height = tile_size
        tile = np.ones((height, width, 3), dtype=np.uint8) * 255

        if x + tile_size_at_0[0] > wsi_width_at_0:
            valid_width_at_0 = max(0, wsi_width_at_0 - x)
            valid_width = int(valid_width_at_0 / downsamples[0])
        else:
            valid_width = width

        if y + tile_size_at_0[1] > wsi_height_at_0:
            valid_height_at_0 = max(0, wsi_height_at_0 - y)
            valid_height = int(valid_height_at_0 / downsamples[1])
        else:
            valid_height = height

        coord = np.ceil(
            tuple(coord[i] / downsamples[i] for i in range(len(coord)))
        ).astype(np.int32)

        if valid_width > 0 and valid_height > 0:
            valid_tile = extract_padded_crop(
                source_canvas,
                x=int(coord[0]),
                y=int(coord[1]),
                width=valid_width,
                height=valid_height,
            )
            valid_tile = Image.fromarray(valid_tile).convert("RGB")

            if aligned_mask is not None:
                if palette is None or pixel_mapping is None or color_mapping is None:
                    raise ValueError(
                        "palette, pixel_mapping, and color_mapping are required when mask overlay is enabled"
                    )
                masked_tile = extract_padded_crop(
                    aligned_mask,
                    x=int(coord[0]),
                    y=int(coord[1]),
                    width=valid_width,
                    height=valid_height,
                )
                if masked_tile.ndim == 3 and masked_tile.shape[-1] == 1:
                    masked_tile = np.squeeze(masked_tile, axis=-1)
                masked_tile = Image.fromarray(masked_tile)
                masked_tile = masked_tile.split()[0]
                masked_tile = masked_tile.resize((valid_width, valid_height), Image.NEAREST)
                overlayed_tile = overlay_mask_on_tile(
                    valid_tile,
                    masked_tile,
                    palette,
                    pixel_mapping,
                    color_mapping,
                )
                tile[:valid_height, :valid_width, :] = overlayed_tile
            else:
                tile[:valid_height, :valid_width, :] = np.array(valid_tile)

        canvas_crop_shape = canvas[
            coord[1] : coord[1] + tile_size[1],
            coord[0] : coord[0] + tile_size[0],
            :3,
        ].shape[:2]
        canvas[
            coord[1] : coord[1] + tile_size[1],
            coord[0] : coord[0] + tile_size[0],
            :3,
        ] = tile[: canvas_crop_shape[0], : canvas_crop_shape[1], :]
        draw_grid(canvas, coord, tile_size, thickness=thickness)

    return Image.fromarray(canvas)


def pad_to_patch_size(canvas: Image.Image, patch_size: tuple[int, int]) -> Image.Image:
    width, height = canvas.size
    pad_width = (patch_size[0] - (width % patch_size[0])) % patch_size[0]
    pad_height = (patch_size[1] - (height % patch_size[1])) % patch_size[1]
    return ImageOps.expand(
        canvas,
        (0, 0, pad_width, pad_height),
        fill=(255, 255, 255),
    )
