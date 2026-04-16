from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .masks import read_aligned_mask
from .preview import (
    build_overlay_alpha,
    build_palette,
    draw_grid_from_coordinates,
    pad_to_patch_size,
)
from .wsi import WSI

DEFAULT_TISSUE_PIXEL_MAPPING = {"background": 0, "tissue": 1}
DEFAULT_TISSUE_COLOR_MAPPING = {
    "background": None,
    "tissue": [157, 219, 129],
}
DEFAULT_TISSUE_BORDER_COLOR = (0x25, 0x5E, 0x3B)
DEFAULT_TISSUE_HOLE_COLOR = (0xF2, 0x6B, 0x3A)
DEFAULT_TISSUE_BORDER_THICKNESS = 4


def _iter_contour_groups(contours) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    if contours is None:
        return []
    contour_groups = getattr(contours, "contours", None)
    hole_groups = getattr(contours, "holes", None)
    if contour_groups is None or hole_groups is None:
        raise ValueError("contours must provide contours and holes attributes")
    if len(contour_groups) != len(hole_groups):
        raise ValueError("contours and holes must have the same length")
    return list(zip(contour_groups, hole_groups))


def _find_contour_groups_from_mask(mask_arr: np.ndarray) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_binary = (mask_arr > 0).astype(np.uint8) * 255
    raw_contours, hierarchy = cv2.findContours(
        mask_binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    if hierarchy is None or len(raw_contours) == 0:
        return []
    hierarchy = hierarchy[0]
    contour_groups: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for fg_idx, h in enumerate(hierarchy):
        if h[3] != -1:
            continue
        hole_indices = np.flatnonzero(hierarchy[:, 3] == fg_idx)
        contour_groups.append(
            (
                raw_contours[fg_idx],
                [raw_contours[idx] for idx in hole_indices.tolist()],
            )
        )
    return contour_groups


def _scale_contour(contour: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    contour_lv = contour.copy().astype(np.float64)
    contour_lv[:, 0, 0] *= scale_x
    contour_lv[:, 0, 1] *= scale_y
    return np.rint(contour_lv).astype(np.int32)


def _draw_contour_groups(
    *,
    canvas: np.ndarray,
    contour_groups: list[tuple[np.ndarray, list[np.ndarray]]],
    scale_x: float,
    scale_y: float,
    outer_color: tuple[int, int, int],
    hole_color: tuple[int, int, int],
    stroke_thickness: int,
) -> None:
    outer_rgba = (*outer_color, 255)
    hole_rgba = (*hole_color, 255)
    for outer_contour, hole_contours in contour_groups:
        scaled_outer = _scale_contour(outer_contour, scale_x, scale_y)
        cv2.drawContours(
            canvas,
            [scaled_outer],
            contourIdx=-1,
            color=outer_rgba,
            thickness=stroke_thickness,
            lineType=cv2.LINE_8,
        )
        for hole_contour in hole_contours:
            scaled_hole = _scale_contour(hole_contour, scale_x, scale_y)
            cv2.drawContours(
                canvas,
                [scaled_hole],
                contourIdx=-1,
                color=hole_rgba,
                thickness=stroke_thickness,
                lineType=cv2.LINE_8,
            )


def _resolve_stroke_thickness(
    *,
    level_downsample: float | tuple[float, float],
    stroke_thickness: int | None,
) -> int:
    if stroke_thickness is not None:
        return max(1, int(stroke_thickness))
    if isinstance(level_downsample, tuple):
        effective_downsample = max(float(level_downsample[0]), float(level_downsample[1]))
    else:
        effective_downsample = float(level_downsample)
    if effective_downsample <= 0:
        return DEFAULT_TISSUE_BORDER_THICKNESS
    resolved = round(
        DEFAULT_TISSUE_BORDER_THICKNESS
        * 16.0
        / effective_downsample
    )
    return max(1, int(resolved))


def _resolve_fill_overlay_style(
    *,
    palette: np.ndarray | None,
    pixel_mapping: dict[str, int] | None,
    color_mapping: dict[str, list[int] | None] | None,
) -> tuple[np.ndarray, dict[str, int], dict[str, list[int] | None]] | None:
    if palette is None and pixel_mapping is None and color_mapping is None:
        return None
    if pixel_mapping is None or color_mapping is None:
        raise ValueError(
            "Provide both pixel_mapping and color_mapping when using filled mask previews"
        )
    if palette is None:
        palette = build_palette(
            pixel_mapping=pixel_mapping,
            color_mapping=color_mapping,
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
    alpha: float = 0.5,
    tile_size_lv0: int | None = None,
    contours=None,
    outer_border_color: tuple[int, int, int] = DEFAULT_TISSUE_BORDER_COLOR,
    hole_border_color: tuple[int, int, int] = DEFAULT_TISSUE_HOLE_COLOR,
    stroke_thickness: int | None = None,
) -> None:
    mask_preview_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_mask_on_slide(
        wsi_path=wsi_path,
        annotation_mask_path=None,
        mask_arr=mask_arr,
        downsample=downsample,
        backend=backend,
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        alpha=alpha,
        tile_size_lv0=tile_size_lv0,
        contours=contours,
        outer_border_color=outer_border_color,
        hole_border_color=hole_border_color,
        stroke_thickness=stroke_thickness,
    )
    if mask_preview_path.suffix.lower() in {".jpg", ".jpeg"}:
        overlay = overlay.convert("RGB")
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
    contours=None,
    outer_border_color: tuple[int, int, int] = DEFAULT_TISSUE_BORDER_COLOR,
    hole_border_color: tuple[int, int, int] = DEFAULT_TISSUE_HOLE_COLOR,
    stroke_thickness: int | None = None,
):
    wsi_object = WSI(path=wsi_path, backend=backend)

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
        mask_object = WSI(
            path=annotation_mask_path,
            backend=backend,
        )
        mask_arr = read_aligned_mask(
            mask_obj=mask_object.reader,
            slide_spacing=wsi_object.get_level_spacing(vis_level),
            slide_dimensions=(base_width, base_height),
        )
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
    elif mask_arr is not None:
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask_arr = cv2.resize(
            mask_arr.astype(np.uint8),
            (base_width, base_height),
            interpolation=cv2.INTER_NEAREST,
        )
    elif contours is None:
        raise ValueError(
            "Provide annotation_mask_path, mask_arr, or contours to overlay_mask_on_slide()"
        )
    else:
        mask_arr = np.zeros((base_height, base_width), dtype=np.uint8)

    if tile_size_lv0 is not None and (width != base_width or height != base_height):
        mask_arr = pad_to_patch_size(mask_arr, tile_size_at_vis_level, fill=0)

    fill_style = _resolve_fill_overlay_style(
        palette=palette,
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
    )
    if fill_style is not None:
        palette, pixel_mapping, color_mapping = fill_style
        mask = Image.fromarray(mask_arr)
        alpha_content = build_overlay_alpha(
            mask_arr=mask_arr,
            alpha=alpha,
            pixel_mapping=pixel_mapping,
            color_mapping=color_mapping,
        )
        mask.putpalette(data=palette.tolist())
        mask_rgb = mask.convert(mode="RGB")
        return Image.composite(image1=wsi, image2=mask_rgb, mask=alpha_content)

    contour_groups = _iter_contour_groups(contours)
    if not contour_groups:
        contour_groups = _find_contour_groups_from_mask(mask_arr)

    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    if contours is not None:
        level_downsample = wsi_object.level_downsamples[vis_level]
        if isinstance(level_downsample, tuple):
            scale_x = 1.0 / float(level_downsample[0])
            scale_y = 1.0 / float(level_downsample[1])
        else:
            scale_x = scale_y = 1.0 / float(level_downsample)
    else:
        scale_x = scale_y = 1.0
    resolved_stroke_thickness = _resolve_stroke_thickness(
        level_downsample=wsi_object.level_downsamples[vis_level],
        stroke_thickness=stroke_thickness,
    )
    _draw_contour_groups(
        canvas=overlay,
        contour_groups=contour_groups,
        scale_x=scale_x,
        scale_y=scale_y,
        outer_color=outer_border_color,
        hole_color=hole_border_color,
        stroke_thickness=resolved_stroke_thickness,
    )

    overlay_image = Image.fromarray(overlay, mode="RGBA")
    return Image.alpha_composite(wsi, overlay_image)


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
    wsi = WSI(wsi_path, backend=backend)
    vis_level = wsi.get_best_level_for_downsample_custom(downsample)
    if mask_path is not None:
        mask = WSI(mask_path, backend=backend)
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
