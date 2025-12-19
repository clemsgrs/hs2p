from pathlib import Path

import cv2
import tqdm
import numpy as np
from PIL import Image, ImageOps
from collections import defaultdict

from .wsi import (
    FilterParameters,
    SegmentationParameters,
    TilingParameters,
    SamplingParameters,
    WholeSlideImage,
)


def sort_coordinates_with_tissue(coordinates, tissue_percentages):
    """
    Deduplicate coordinates, then sort deterministically by mocked filename.
    """
    seen = set()
    dedup_coordinates = []
    dedup_tissue_percentages = []
    # deduplicate
    for (x, y), tissue in zip(coordinates, tissue_percentages):
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        dedup_coordinates.append((x, y))
        dedup_tissue_percentages.append(tissue)
    # mock filenames
    mocked_filenames = [f"{x}_{y}.jpg" for x, y in dedup_coordinates]
    # sort combined list by mocked filenames
    combined = list(zip(mocked_filenames, dedup_coordinates, dedup_tissue_percentages))
    sorted_combined = sorted(combined, key=lambda x: x[0])
    # extract sorted coordinates and tissue percentages
    sorted_coordinates = [coord for _, coord, _ in sorted_combined]
    sorted_tissue_percentages = [tissue for _, _, tissue in sorted_combined]
    return sorted_coordinates, sorted_tissue_percentages


def get_mask_coverage(mask: np.ndarray, val: int):
    """
    Determine the percentage of a mask that is equal to value `val`.
    input:
        mask: mask as a numpy array
        val: given value we want to check for
    output:
        percentage of the numpy array that is equal to val
    """
    binary_mask = mask == val
    mask_percentage = np.sum(binary_mask) / np.size(mask)
    return mask_percentage


def extract_coordinates(
    *,
    wsi_path: Path,
    mask_path: Path,
    backend: str,
    segment_params: SegmentationParameters,
    tiling_params: TilingParameters,
    filter_params: FilterParameters,
    sampling_params: SamplingParameters | None = None,
    mask_visu_path: Path | None = None,
    disable_tqdm: bool = False,
    num_workers: int = 1,
):
    wsi = WholeSlideImage(
        path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        segment=True,
        segment_params=segment_params,
        sampling_params=sampling_params,
    )
    tolerance = tiling_params.tolerance
    starting_spacing = wsi.spacings[0]
    desired_spacing = tiling_params.spacing
    if desired_spacing < starting_spacing:
        relative_diff = abs(starting_spacing - desired_spacing) / desired_spacing
        if relative_diff > tolerance:
            raise ValueError(
                f"Desired spacing ({desired_spacing}) is smaller than the whole-slide image starting spacing ({starting_spacing}) and does not fall within tolerance ({tolerance})"
            )
    (
        contours,
        holes,
        coordinates,
        tissue_percentages,
        tile_level,
        resize_factor,
        tile_size_lv0,
    ) = wsi.get_tile_coordinates(
        tiling_params,
        filter_params,
        disable_tqdm=disable_tqdm,
        num_workers=num_workers,
    )
    sorted_coordinates, _ = sort_coordinates_with_tissue(
        coordinates, tissue_percentages
    )
    if mask_visu_path is not None:
        wsi.visualize_mask(contours, holes).save(mask_visu_path)
    return (
        sorted_coordinates,
        tile_level,
        resize_factor,
        tile_size_lv0,
    )


def sample_coordinates(
    *,
    wsi_path: Path,
    mask_path: Path,
    backend: str,
    segment_params: SegmentationParameters,
    tiling_params: TilingParameters,
    filter_params: FilterParameters,
    sampling_params: SamplingParameters,
    annotation: str,
    mask_visu_path: Path | None = None,
    disable_tqdm: bool = False,
    num_workers: int = 1,
):
    wsi = WholeSlideImage(
        path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        segment=True,
        segment_params=segment_params,
        sampling_params=sampling_params,
    )
    tolerance = tiling_params.tolerance
    starting_spacing = wsi.spacings[0]
    desired_spacing = tiling_params.spacing
    if desired_spacing < starting_spacing:
        relative_diff = abs(starting_spacing - desired_spacing) / desired_spacing
        if relative_diff > tolerance:
            raise ValueError(
                f"Desired spacing ({desired_spacing}) is smaller than the whole-slide image starting spacing ({starting_spacing}) and does not fall within tolerance ({tolerance})"
            )
    (
        contours,
        holes,
        coordinates,
        tissue_percentages,
        tile_level,
        resize_factor,
        tile_size_lv0,
    ) = wsi.get_tile_coordinates(
        tiling_params,
        filter_params,
        annotation=annotation,
        disable_tqdm=disable_tqdm,
        num_workers=num_workers,
    )
    sorted_coordinates, _ = sort_coordinates_with_tissue(
        coordinates, tissue_percentages
    )
    if mask_visu_path is not None:
        wsi.visualize_mask(contours, holes).save(mask_visu_path)
    return (
        sorted_coordinates,
        tile_level,
        resize_factor,
        tile_size_lv0,
    )


def filter_coordinates(
    *,
    wsi_path: Path,
    mask_path: Path,
    backend: str,
    coordinates: np.ndarray,
    tile_level: int,
    segment_params: SegmentationParameters,
    tiling_params: TilingParameters,
    sampling_params: SamplingParameters,
    disable_tqdm: bool = False,
):
    wsi = WholeSlideImage(
        path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        segment=True,
        segment_params=segment_params,
        sampling_params=sampling_params,
    )
    tile_spacing = wsi.get_level_spacing(tile_level)
    resize_factor = tiling_params.spacing / tile_spacing
    tile_size_resized = int(round(tiling_params.tile_size * resize_factor, 0))

    mask = WholeSlideImage(
        path=mask_path,
        backend=backend,
    )
    mask_spacing_at_level_0 = mask.spacings[0]
    mask_downsample = tile_spacing / mask_spacing_at_level_0
    mask_level = int(
        np.argmin([abs(x - mask_downsample) for x, _ in mask.level_downsamples])
    )
    mask_spacing = mask.spacings[mask_level]
    scale = tile_spacing / mask_spacing
    while scale < 1 and mask_level > 0:
        mask_level -= 1
        mask_spacing = mask.spacings[mask_level]
        scale = tile_spacing / mask_spacing

    filtered_coordinates = defaultdict(list)
    for annotation, pct in sampling_params.tissue_percentage.items():
        if pct is None:
            continue
        if annotation not in sampling_params.pixel_mapping:
            continue
        with tqdm.tqdm(
            coordinates,
            desc=f"Filtering coordinates for annotation '{annotation}' (min coverage {pct}%) on slide {wsi_path.stem}",
            unit=" tile",
            total=len(coordinates),
            disable=disable_tqdm,
        ) as t:
            for coord in t:
                x, y = coord
                # need to scale (x, y) defined w.r.t. slide level 0
                # to mask level 0
                downsample = wsi.spacings[0] / mask.spacings[0]
                x_downsampled = int(round(x * downsample, 0))
                y_downsampled = int(round(y * downsample, 0))
                tile_size_at_mask_spacing = int(round(tile_size_resized * scale, 0))
                masked_tile = mask.get_tile(
                    x_downsampled,
                    y_downsampled,
                    tile_size_at_mask_spacing,
                    tile_size_at_mask_spacing,
                    spacing=mask_spacing,
                )
                if masked_tile.shape[-1] == 1:
                    masked_tile = np.squeeze(masked_tile, axis=-1)
                mask_pct = get_mask_coverage(masked_tile, sampling_params.pixel_mapping[annotation])
                if mask_pct >= pct:
                    filtered_coordinates[annotation].append(coord)
    return filtered_coordinates     


def save_coordinates(
    *,
    coordinates: list[tuple[int, int]],
    target_spacing: float,
    tile_level: int,
    tile_size: int,
    resize_factor: float,
    tile_size_lv0: int,
    save_path: Path,
):
    x = [x for x, _ in coordinates]  # defined w.r.t level 0
    y = [y for _, y in coordinates]  # defined w.r.t level 0
    ntile = len(x)
    tile_size_resized = int(round(tile_size * resize_factor, 0))
    dtype = [
        ("x", int),
        ("y", int),
        ("tile_size_resized", int),
        ("tile_level", int),
        ("resize_factor", float),
        ("tile_size_lv0", int),
        ("target_spacing", float),
    ]
    data = np.zeros(ntile, dtype=dtype)
    for i in range(ntile):
        data[i] = (
            x[i],
            y[i],
            tile_size_resized,
            tile_level,
            resize_factor,
            tile_size_lv0,
            target_spacing,
        )
    data_arr = np.array(data)
    np.save(save_path, data_arr)
    return save_path


def overlay_mask_on_tile(
    tile: Image.Image,
    mask: Image.Image,
    palette: dict[str, int],
    alpha=0.5,
):

    # create alpha mask
    mask_arr = np.array(mask)
    alpha_int = int(round(255 * alpha))
    alpha_content = np.less_equal(mask_arr, 0).astype("uint8") * alpha_int + (
        255 - alpha_int
    )
    alpha_content = Image.fromarray(alpha_content)

    mask.putpalette(data=palette.tolist())
    mask_rgb = mask.convert(mode="RGB")

    overlayed_image = Image.composite(image1=tile, image2=mask_rgb, mask=alpha_content)
    return overlayed_image


def overlay_mask_on_slide(
    wsi_path: Path,
    annotation_mask_path: Path,
    downsample: int,
    palette: dict[str, int],
    pixel_mapping: dict[str, int],
    color_mapping: dict[str, list[int]] | None = None,
    alpha: float = 0.5,
):
    """
    Show a mask overlayed on a slide
    """

    wsi_object = WholeSlideImage(path=wsi_path, backend="asap")
    mask_object = WholeSlideImage(path=annotation_mask_path, backend="asap")

    vis_level = wsi_object.get_best_level_for_downsample_custom(downsample)
    vis_spacing = wsi_object.spacings[vis_level]
    wsi_arr = wsi_object.get_slide(spacing=vis_spacing)
    _, width, _ = wsi_arr.shape

    wsi = Image.fromarray(wsi_arr).convert("RGBA")
    mask_width_at_level_0, _ = mask_object.level_dimensions[0]
    mask_downsample = mask_width_at_level_0 / width
    mask_level = int(
        np.argmin([abs(x - mask_downsample) for x, _ in mask_object.level_downsamples])
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
    mask_arr = mask_arr[:, :, 0]
    mask_height, mask_width = mask_arr.shape

    # resize the mask to the size of the slide at seg_spacing
    mask_arr = cv2.resize(
        mask_arr.astype(np.uint8),
        (int(round(mask_width / scale, 0)), int(round(mask_height / scale, 0))),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = Image.fromarray(mask_arr)

    # create alpha mask
    alpha_int = int(round(255 * alpha))
    if color_mapping is not None:
        alpha_content = np.zeros_like(mask_arr)
        for k, v in pixel_mapping.items():
            if color_mapping[k] is not None:
                alpha_content += mask_arr == v
        alpha_content = np.less(alpha_content, 1).astype("uint8") * alpha_int + (
            255 - alpha_int
        )
    else:
        alpha_content = np.less_equal(mask_arr, 0).astype("uint8") * alpha_int + (
            255 - alpha_int
        )
    alpha_content = Image.fromarray(alpha_content)

    mask.putpalette(data=palette.tolist())
    mask_rgb = mask.convert(mode="RGB")

    overlayed_image = Image.composite(image1=wsi, image2=mask_rgb, mask=alpha_content)
    return overlayed_image


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
    mask = None,
    palette: dict[str, int] | None = None,
):
    downsamples = wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)

    tile_size = tuple(
        np.ceil((np.array(tile_size_at_0) / np.array(downsamples))).astype(np.int32)
    )  # defined w.r.t vis_level

    wsi_width_at_0, wsi_height_at_0 = wsi.level_dimensions[
        0
    ]  # retrieve slide dimension at level 0

    vis_spacing = wsi.get_level_spacing(vis_level)
    if mask is not None:
        mask_spacing_at_level_0 = mask.spacings[0]
        mask_downsample = vis_spacing / mask_spacing_at_level_0
        mask_level = int(
            np.argmin([abs(x - mask_downsample) for x, _ in mask.level_downsamples])
        )
        mask_spacing = mask.spacings[mask_level]
        scale = vis_spacing / mask_spacing
        while scale < 1 and mask_level > 0:
            mask_level -= 1
            mask_spacing = mask.spacings[mask_level]
            scale = vis_spacing / mask_spacing

    for idx in range(total):
        tile_id = indices[idx]
        coord = coords[tile_id]
        x, y = coord

        width, height = tile_size
        tile = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        # compute valid tile area
        if x + tile_size_at_0[0] > wsi_width_at_0:
            valid_width_at_0 = max(
                0, wsi_width_at_0 - x
            )  # how much of the tile width is inside the wsi
            valid_width = int(valid_width_at_0 / downsamples[0])
        else:
            valid_width = width

        if y + tile_size_at_0[1] > wsi_height_at_0:
            valid_height_at_0 = max(
                0, wsi_height_at_0 - y
            )  # how much of the tile height is inside the wsi
            valid_height = int(valid_height_at_0 / downsamples[1])
        else:
            valid_height = height

        # extract only the valid portion of the tile
        if valid_width > 0 and valid_height > 0:
            valid_tile = wsi.get_tile(
                x, y, valid_width, valid_height, spacing=vis_spacing
            )
            valid_tile = Image.fromarray(valid_tile).convert("RGB")

            if mask is not None:
                # need to scale (x, y) defined w.r.t. slide level 0
                # to mask level 0
                downsample = wsi.spacings[0] / mask.spacings[0]
                x_downsampled = int(round(x * downsample, 0))
                y_downsampled = int(round(y * downsample, 0))
                valid_width_at_mask_spacing = int(round(valid_width * scale, 0))
                valid_height_at_mask_spacing = int(round(valid_height * scale, 0))
                masked_tile = mask.get_tile(
                    x_downsampled,
                    y_downsampled,
                    valid_width_at_mask_spacing,
                    valid_height_at_mask_spacing,
                    spacing=mask_spacing,
                )
                if masked_tile.shape[-1] == 1:
                    masked_tile = np.squeeze(masked_tile, axis=-1)
                masked_tile = Image.fromarray(masked_tile)
                masked_tile = masked_tile.split()[0]
                masked_tile = masked_tile.resize(
                    (valid_width, valid_height),
                    Image.NEAREST,
                )
                # masked_tile = cv2.resize(
                #     masked_tile.astype(np.uint8),
                #     (valid_width, valid_height),
                #     interpolation=cv2.INTER_NEAREST,
                # )
                overlayed_tile = overlay_mask_on_tile(
                    valid_tile,
                    masked_tile,
                    palette,
                )

                # paste the valid part into the white tile
                tile[:valid_height, :valid_width, :] = overlayed_tile
            
            else:
                valid_tile = np.array(valid_tile)
                # paste the valid part into the white tile
                tile[:valid_height, :valid_width, :] = valid_tile

        coord = np.ceil(
            tuple(coord[i] / downsamples[i] for i in range(len(coord)))
        ).astype(np.int32)
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
    # compute amount of padding required for width and height
    pad_width = (patch_size[0] - (width % patch_size[0])) % patch_size[0]
    pad_height = (patch_size[1] - (height % patch_size[1])) % patch_size[1]
    # apply the padding to canvas
    padded_canvas = ImageOps.expand(
        canvas, (0, 0, pad_width, pad_height), fill=(255, 255, 255)
    )  # white padding
    return padded_canvas


def visualize_coordinates(
    *,
    wsi_path: Path,
    coordinates: list[tuple[int, int]],
    tile_size_lv0: int,
    save_dir: Path,
    downsample: int = 64,
    backend: str = "asap",
    grid_thickness: int = 1,
    mask_path: Path | None = None,
    annotation: str | None = None,
    palette: dict[str, int] | None = None,
):
    wsi = WholeSlideImage(wsi_path, backend=backend)
    vis_level = wsi.get_best_level_for_downsample_custom(downsample)
    vis_spacing = wsi.spacings[vis_level]

    if mask_path is not None:
        mask = WholeSlideImage(mask_path, backend=backend)
    else:
        mask = None

    canvas = wsi.get_slide(spacing=vis_spacing)
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
    )  # defined w.r.t vis_level

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
    )
    wsi_name = wsi_path.stem.replace(" ", "_")
    if annotation is not None:
        save_dir = Path(save_dir, annotation)
        save_dir.mkdir(parents=True, exist_ok=True)
    visu_path = Path(save_dir, f"{wsi_name}.jpg")
    canvas.save(visu_path)
