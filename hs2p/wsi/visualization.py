from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .masks import pad_array_to_shape
from .preview import (
    build_overlay_alpha,
    build_palette,
    draw_grid_from_coordinates,
    pad_to_patch_size,
)

DEFAULT_TISSUE_PIXEL_MAPPING = {"background": 0, "tissue": 1}
DEFAULT_TISSUE_COLOR_MAPPING = {
    "background": None,
    "tissue": [157, 219, 129],
}


def _wsi_package():
    import hs2p.wsi as wsi_pkg

    return wsi_pkg


def _resolve_overlay_style(
    *,
    palette: np.ndarray | None,
    pixel_mapping: dict[str, int] | None,
    color_mapping: dict[str, list[int] | None] | None,
) -> tuple[np.ndarray, dict[str, int], dict[str, list[int] | None]]:
    if palette is None and pixel_mapping is None and color_mapping is None:
        pixel_mapping = DEFAULT_TISSUE_PIXEL_MAPPING
        color_mapping = DEFAULT_TISSUE_COLOR_MAPPING
        palette = build_palette(
            pixel_mapping=pixel_mapping,
            color_mapping=color_mapping,
        )
        return palette, pixel_mapping, color_mapping
    if palette is None or pixel_mapping is None or color_mapping is None:
        raise ValueError(
            "Provide either all of palette, pixel_mapping, and color_mapping, or none of them"
        )
    return palette, pixel_mapping, color_mapping


def save_overlay_preview(
    *,
    wsi_path: Path,
    backend: str,
    mask_arr: np.ndarray,
    mask_preview_path: Path,
    downsample: int = 32,
    palette: np.ndarray | None = None,
    pixel_mapping: dict[str, int] | None = None,
    color_mapping: dict[str, list[int] | None] | None = None,
    tile_size_lv0: int | None = None,
) -> None:
    mask_preview_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = _wsi_package().overlay_mask_on_slide(
        wsi_path=wsi_path,
        annotation_mask_path=None,
        mask_arr=mask_arr,
        downsample=downsample,
        backend=backend,
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        tile_size_lv0=tile_size_lv0,
    )
    overlay.save(mask_preview_path)


def overlay_mask_on_slide(
    wsi_path: Path,
    annotation_mask_path: Path | None,
    downsample: int,
    backend: str,
    palette: np.ndarray | None = None,
    pixel_mapping: dict[str, int] | None = None,
    color_mapping: dict[str, list[int] | None] | None = None,
    alpha: float = 0.5,
    mask_arr: np.ndarray | None = None,
    tile_size_lv0: int | None = None,
):
    """
    Show a mask overlayed on a slide
    """

    palette, pixel_mapping, color_mapping = _resolve_overlay_style(
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )

    wsi_object = _wsi_package().WholeSlideImage(path=wsi_path, backend=backend)

    vis_level = wsi_object.get_best_level_for_downsample_custom(downsample)
    wsi_arr = wsi_object.get_slide(vis_level)
    height, width, _ = wsi_arr.shape
    base_width, base_height = width, height

    wsi = Image.fromarray(wsi_arr).convert("RGBA")
    if tile_size_lv0 is not None:
        tile_size_at_vis_level = tuple(
            (
                np.array((tile_size_lv0, tile_size_lv0))
                / np.array(wsi_object.level_downsamples[vis_level])
            ).astype(np.int32)
        )
        wsi = pad_to_patch_size(wsi.convert("RGB"), tile_size_at_vis_level).convert(
            "RGBA"
        )
        width, height = wsi.size
    if annotation_mask_path is not None:
        mask_object = _wsi_package().WholeSlideImage(
            path=annotation_mask_path,
            backend=backend,
        )
        mask_width_at_level_0, _ = mask_object.level_dimensions[0]
        mask_downsample = mask_width_at_level_0 / width
        mask_level = int(
            np.argmin(
                [abs(x - mask_downsample) for x, _ in mask_object.level_downsamples]
            )
        )
        mask_spacing = mask_object.spacings[mask_level]
        mask_width, _ = mask_object.level_dimensions[mask_level]

        scale = mask_width / width
        while scale < 1 and mask_level > 0:
            mask_level -= 1
            mask_spacing = mask_object.spacings[mask_level]
            mask_width, _ = mask_object.level_dimensions[mask_level]
            scale = mask_width / width

        mask_arr = mask_object.get_slide(spacing=mask_spacing)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask_arr = cv2.resize(
            mask_arr.astype(np.uint8),
            (base_width, base_height),
            interpolation=cv2.INTER_NEAREST,
        )
    elif mask_arr is not None:
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask_arr = cv2.resize(
            mask_arr.astype(np.uint8),
            (base_width, base_height),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        raise ValueError(
            "Provide annotation_mask_path or mask_arr to overlay_mask_on_slide()"
        )

    if tile_size_lv0 is not None and (width != base_width or height != base_height):
        mask_arr = pad_array_to_shape(
            mask_arr,
            target_width=width,
            target_height=height,
        )

    mask = Image.fromarray(mask_arr)

    alpha_content = build_overlay_alpha(
        mask_arr=mask_arr,
        alpha=alpha,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )

    mask.putpalette(data=palette.tolist())
    mask_rgb = mask.convert(mode="RGB")

    overlayed_image = Image.composite(image1=wsi, image2=mask_rgb, mask=alpha_content)
    return overlayed_image


def write_coordinate_preview(
    *,
    wsi_path: Path,
    coordinates: list[tuple[int, int]],
    tile_size_lv0: int,
    save_dir: Path,
    backend: str,
    sample_id: str | None = None,
    downsample: int = 64,
    grid_thickness: int = 1,
    mask_path: Path | None = None,
    annotation: str | None = None,
    palette: dict[str, int] | None = None,
    pixel_mapping: dict[str, int] | None = None,
    color_mapping: dict[str, list[int] | None] | None = None,
):
    wsi = _wsi_package().WholeSlideImage(wsi_path, backend=backend)
    vis_level = wsi.get_best_level_for_downsample_custom(downsample)
    if mask_path is not None:
        mask = _wsi_package().WholeSlideImage(mask_path, backend=backend)
    else:
        mask = None

    canvas = wsi.get_slide(vis_level)
    canvas = Image.fromarray(canvas).convert("RGB")
    if len(coordinates) == 0:
        return canvas

    w, h = wsi.level_dimensions[vis_level]
    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError(
            f"Visualization downsample ({downsample}) is too large"
        )

    tile_size_at_0 = (tile_size_lv0, tile_size_lv0)
    tile_size_at_vis_level = tuple(
        (np.array(tile_size_at_0) / np.array(wsi.level_downsamples[vis_level])).astype(
            np.int32
        )
    )

    canvas = pad_to_patch_size(canvas, tile_size_at_vis_level)
    canvas = np.array(canvas)
    canvas = draw_grid_from_coordinates(
        canvas,
        wsi,
        coordinates,
        tile_size_at_0,
        vis_level,
        indices=None,
        thickness=grid_thickness,
        mask=mask,
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )
    wsi_name = sample_id if sample_id is not None else wsi_path.stem.replace(" ", "_")
    if annotation is not None:
        save_dir = Path(save_dir, annotation)
        save_dir.mkdir(parents=True, exist_ok=True)
    preview_path = Path(save_dir, f"{wsi_name}.jpg")
    canvas.save(preview_path)
