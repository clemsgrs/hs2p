from __future__ import annotations

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tqdm

from hs2p.wsi.geometry import ResolvedTileGeometry
from hs2p.wsi.utils import HasEnoughTissue


def filter_black_and_white_tiles(
    *,
    reader,
    level_dimensions,
    level_downsamples,
    keep_flags,
    coord_candidates,
    tile_size,
    tile_level,
    filter_params,
):
    if not (filter_params.filter_white or filter_params.filter_black):
        return keep_flags

    img_w, img_h = level_dimensions[tile_level]
    downsample_x, downsample_y = level_downsamples[tile_level]
    keep_array = np.asarray(keep_flags, dtype=np.uint8).copy()
    active_indices = np.flatnonzero(keep_array)
    if active_indices.size == 0:
        return keep_array.tolist()

    level_coords = np.empty((coord_candidates.shape[0], 2), dtype=np.int64)
    level_coords[:, 0] = np.floor(coord_candidates[:, 0] / downsample_x).astype(np.int64)
    level_coords[:, 1] = np.floor(coord_candidates[:, 1] / downsample_y).astype(np.int64)

    supertile_span = int(tile_size) * 8
    batched_indices: dict[tuple[int, int], list[int]] = {}
    for idx in active_indices.tolist():
        x_level, y_level = level_coords[idx]
        batch_key = (int(x_level // supertile_span), int(y_level // supertile_span))
        batched_indices.setdefault(batch_key, []).append(idx)

    error_count = 0
    error_samples = []
    for (batch_x, batch_y), batch_member_indices in batched_indices.items():
        x0_level = batch_x * supertile_span
        y0_level = batch_y * supertile_span
        read_width = int(min(supertile_span + tile_size, max(0, img_w - x0_level)))
        read_height = int(min(supertile_span + tile_size, max(0, img_h - y0_level)))
        x0 = int(round(x0_level * downsample_x))
        y0 = int(round(y0_level * downsample_y))
        try:
            window = reader.read_region(
                (x0, y0),
                tile_level,
                (read_width, read_height),
                pad_missing=True,
            )
        except Exception as exc:
            error_count += len(batch_member_indices)
            if len(error_samples) < 3:
                error_samples.append(str(exc))
            continue

        tiles = np.zeros((len(batch_member_indices), tile_size, tile_size, 3), dtype=window.dtype)
        valid_mask = np.zeros((len(batch_member_indices), tile_size, tile_size), dtype=bool)
        for local_idx, coord_idx in enumerate(batch_member_indices):
            x_level, y_level = level_coords[coord_idx]
            offset_x = int(x_level - x0_level)
            offset_y = int(y_level - y0_level)
            tile_view = window[offset_y : offset_y + tile_size, offset_x : offset_x + tile_size, :3]
            height = min(tile_size, tile_view.shape[0])
            width = min(tile_size, tile_view.shape[1])
            if height > 0 and width > 0:
                tiles[local_idx, :height, :width, :] = tile_view[:height, :width]

            valid_w = int(min(tile_size, max(0, img_w - x_level)))
            valid_h = int(min(tile_size, max(0, img_h - y_level)))
            if valid_h > 0 and valid_w > 0:
                valid_mask[local_idx, :valid_h, :valid_w] = True

        valid_area = np.maximum(valid_mask.sum(axis=(1, 2)), 1)
        keep_batch = np.ones(len(batch_member_indices), dtype=np.uint8)
        if filter_params.filter_white:
            white_pixels = np.all(tiles > filter_params.white_threshold, axis=-1) & valid_mask
            white_fraction = white_pixels.sum(axis=(1, 2)) / valid_area
            keep_batch = (white_fraction <= filter_params.fraction_threshold).astype(np.uint8)
        if filter_params.filter_black:
            black_pixels = np.all(tiles < filter_params.black_threshold, axis=-1) & valid_mask
            black_fraction = black_pixels.sum(axis=(1, 2)) / valid_area
            keep_batch = keep_batch & (
                black_fraction <= filter_params.fraction_threshold
            ).astype(np.uint8)
        for coord_idx, keep in zip(batch_member_indices, keep_batch.tolist()):
            keep_array[coord_idx] = keep

    if error_count > 0:
        warnings.warn(
            f"Encountered {error_count} tile filtering error(s). "
            f"Keeping affected tile(s). Sample error(s): {'; '.join(error_samples)}",
            UserWarning,
        )
    return keep_array.tolist()


def process_contour(
    *,
    reader,
    annotation_mask,
    annotation_pct,
    level_spacings,
    level_dimensions,
    level_downsamples,
    seg_level,
    contour,
    contour_holes,
    tiling_params,
    filter_params,
    annotation: str | None = None,
):
    target_tile_size = tiling_params.target_tile_size_px
    overlap = tiling_params.overlap
    use_padding = tiling_params.use_padding

    from hs2p.wsi.geometry import select_level
    from hs2p.wsi.segmentation import scale_contour_dim

    selection = select_level(
        requested_spacing_um=tiling_params.target_spacing_um,
        level0_spacing_um=level_spacings[0],
        level_downsamples=level_downsamples,
        tolerance=tiling_params.tolerance,
    )
    tile_level = selection.level
    tile_spacing = level_spacings[tile_level]
    resize_factor = tiling_params.target_spacing_um / tile_spacing
    if selection.is_within_tolerance:
        resize_factor = 1.0

    tile_size_resized = int(round(target_tile_size * resize_factor, 0))
    step_size = int(tile_size_resized * (1.0 - overlap))

    import cv2

    if contour is not None:
        start_x, start_y, w, h = cv2.boundingRect(contour)
    else:
        start_x, start_y, w, h = (0, 0, level_dimensions[tile_level][0], level_dimensions[tile_level][1])

    tile_downsample = (
        int(level_downsamples[tile_level][0]),
        int(level_downsamples[tile_level][1]),
    )
    tile_size_at_level_0 = (
        tile_size_resized * tile_downsample[0],
        tile_size_resized * tile_downsample[1],
    )

    img_w, img_h = level_dimensions[0]
    if use_padding:
        stop_y = int(start_y + h)
        stop_x = int(start_x + w)
    else:
        stop_y = min(start_y + h, img_h - tile_size_at_level_0[1] + 1)
        stop_x = min(start_x + w, img_w - tile_size_at_level_0[0] + 1)

    scale = level_downsamples[seg_level]
    cont = scale_contour_dim([contour], (1.0 / scale[0], 1.0 / scale[1]))[0]
    mask = annotation_mask["tissue"] if annotation is None else annotation_mask[annotation]
    if annotation is None and annotation_pct is None:
        pct = tiling_params.tissue_threshold
    else:
        pct = annotation_pct["tissue"] if annotation is None else annotation_pct[annotation]
    tissue_checker = HasEnoughTissue(
        contour=cont,
        contour_holes=contour_holes,
        tissue_mask=mask,
        geometry=ResolvedTileGeometry(
            target_tile_size_px=target_tile_size,
            read_spacing_um=tile_spacing,
            resize_factor=resize_factor,
            seg_spacing_um=level_spacings[seg_level],
            level0_spacing_um=level_spacings[0],
        ),
        pct=pct,
    )

    ref_step_size_x = int(round(step_size * tile_downsample[0], 0))
    ref_step_size_y = int(round(step_size * tile_downsample[1], 0))

    x_range = np.arange(start_x, stop_x, step=ref_step_size_x)
    y_range = np.arange(start_y, stop_y, step=ref_step_size_y)
    x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
    coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

    keep_flags, tissue_pcts = tissue_checker.check_coordinates(coord_candidates)
    keep_flags = filter_black_and_white_tiles(
        reader=reader,
        level_dimensions=level_dimensions,
        level_downsamples=level_downsamples,
        keep_flags=keep_flags,
        coord_candidates=coord_candidates,
        tile_size=tile_size_resized,
        tile_level=tile_level,
        filter_params=filter_params,
    )

    filtered_coordinates = coord_candidates[np.array(keep_flags) == 1]
    filtered_tissue_percentages = np.array(tissue_pcts)[np.array(keep_flags) == 1]
    if len(filtered_coordinates) == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            tile_level,
            resize_factor,
        )

    x_coords = filtered_coordinates[:, 0].astype(np.int64, copy=False)
    y_coords = filtered_coordinates[:, 1].astype(np.int64, copy=False)
    return (
        x_coords,
        y_coords,
        filtered_tissue_percentages.astype(np.float32, copy=False),
        tile_level,
        resize_factor,
    )


def process_contours(
    *,
    reader,
    annotation_mask,
    annotation_pct,
    level_spacings,
    level_dimensions,
    level_downsamples,
    seg_level,
    contours,
    holes,
    tiling_params,
    filter_params,
    annotation: str | None = None,
    disable_tqdm: bool = False,
    num_workers: int = 1,
):
    x_coord_chunks: list[np.ndarray] = []
    y_coord_chunks: list[np.ndarray] = []
    tissue_pct_chunks: list[np.ndarray] = []
    contour_index_chunks: list[np.ndarray] = []

    def process_single_contour(index: int):
        return process_contour(
            reader=reader,
            annotation_mask=annotation_mask,
            annotation_pct=annotation_pct,
            level_spacings=level_spacings,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            seg_level=seg_level,
            contour=contours[index],
            contour_holes=holes[index],
            tiling_params=tiling_params,
            filter_params=filter_params,
            annotation=annotation,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(process_single_contour, range(len(contours))),
                desc="Extracting tissue tiles",
                unit=" tissue blob",
                total=len(contours),
                leave=True,
                disable=disable_tqdm,
                file=sys.stdout,
            )
        )

    tile_level = None
    resize_factor = None
    for index, (x_coords, y_coords, tissue_pct, cont_tile_level, cont_resize_factor) in enumerate(results):
        tile_level = cont_tile_level
        resize_factor = cont_resize_factor
        if x_coords.shape[0] > 0:
            x_coord_chunks.append(x_coords.astype(np.int64, copy=False))
            y_coord_chunks.append(y_coords.astype(np.int64, copy=False))
            tissue_pct_chunks.append(np.asarray(tissue_pct, dtype=np.float32))
            contour_index_chunks.append(np.full(x_coords.shape[0], index, dtype=np.int32))

    if x_coord_chunks:
        running_x_coords = np.concatenate(x_coord_chunks)
        running_y_coords = np.concatenate(y_coord_chunks)
        running_tissue_pct = np.concatenate(tissue_pct_chunks)
        running_contour_indices = np.concatenate(contour_index_chunks)
    else:
        running_x_coords = np.array([], dtype=np.int64)
        running_y_coords = np.array([], dtype=np.int64)
        running_tissue_pct = np.array([], dtype=np.float32)
        running_contour_indices = np.array([], dtype=np.int32)

    return (
        running_x_coords,
        running_y_coords,
        running_tissue_pct,
        running_contour_indices,
        tile_level if tile_level is not None else 0,
        resize_factor if resize_factor is not None else 1.0,
    )
