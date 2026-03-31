from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np

from hs2p.wsi.geometry import project_discrete_grid_origins, select_level


@dataclass(frozen=True)
class QCReadGeometry:
    level: int
    read_spacing_um: float
    requested_spacing_um: float
    read_tile_size_px: int
    qc_tile_size_px: int


def _masked_fraction(mask: np.ndarray, valid_mask: np.ndarray | None) -> float:
    mask = np.asarray(mask, dtype=bool)
    if valid_mask is None:
        return float(mask.mean())

    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != mask.shape:
        raise ValueError("valid_mask must match the tile height and width")
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0.0
    return float(mask[valid].mean())


def filter_whitespace(
    tile: np.ndarray,
    *,
    threshold: int = 220,
    max_fraction: float = 0.9,
    valid_mask: np.ndarray | None = None,
) -> bool:
    white_pixels = np.all(np.asarray(tile) > int(threshold), axis=-1)
    return bool(_masked_fraction(white_pixels, valid_mask) <= max_fraction)


def filter_blackspace(
    tile: np.ndarray,
    *,
    threshold: int = 25,
    max_fraction: float = 0.9,
    valid_mask: np.ndarray | None = None,
) -> bool:
    black_pixels = np.all(np.asarray(tile) < int(threshold), axis=-1)
    return bool(_masked_fraction(black_pixels, valid_mask) <= max_fraction)


def filter_grayspace(
    tile: np.ndarray,
    *,
    saturation_threshold: float = 0.05,
    max_fraction: float = 0.6,
    valid_mask: np.ndarray | None = None,
) -> bool:
    hsv = cv2.cvtColor(np.asarray(tile), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    gray_pixels = saturation < float(saturation_threshold)
    return bool(_masked_fraction(gray_pixels, valid_mask) <= max_fraction)


def compute_blur_score(
    tile: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> float:
    tile = np.asarray(tile)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != tile.shape[:2]:
            raise ValueError("valid_mask must match the tile height and width")
        valid_y, valid_x = np.nonzero(valid)
        if valid_y.size == 0:
            return float("inf")
        y0 = int(valid_y.min())
        y1 = int(valid_y.max()) + 1
        x0 = int(valid_x.min())
        x1 = int(valid_x.max()) + 1
        tile = tile[y0:y1, x0:x1]
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def apply_tile_qc(
    tile: np.ndarray,
    *,
    valid_mask: np.ndarray | None,
    filter_params,
) -> bool:
    if getattr(filter_params, "filter_white", False) and not filter_whitespace(
        tile,
        threshold=int(filter_params.white_threshold),
        max_fraction=float(filter_params.fraction_threshold),
        valid_mask=valid_mask,
    ):
        return False
    if getattr(filter_params, "filter_black", False) and not filter_blackspace(
        tile,
        threshold=int(filter_params.black_threshold),
        max_fraction=float(filter_params.fraction_threshold),
        valid_mask=valid_mask,
    ):
        return False
    if getattr(filter_params, "filter_grayspace", False) and not filter_grayspace(
        tile,
        saturation_threshold=float(filter_params.grayspace_saturation_threshold),
        max_fraction=float(filter_params.grayspace_fraction_threshold),
        valid_mask=valid_mask,
    ):
        return False
    if getattr(filter_params, "filter_blur", False):
        blur_score = compute_blur_score(tile, valid_mask=valid_mask)
        if blur_score < float(filter_params.blur_threshold):
            return False
    return True


def needs_pixel_qc(filter_params) -> bool:
    return bool(
        getattr(filter_params, "filter_white", False)
        or getattr(filter_params, "filter_black", False)
        or getattr(filter_params, "filter_grayspace", False)
        or getattr(filter_params, "filter_blur", False)
    )


def resolve_qc_read_geometry(
    *,
    target_tile_size_px: int,
    target_spacing_um: float,
    qc_spacing_um: float,
    base_spacing_um: float,
    level_downsamples: list[float] | list[tuple[float, float]],
    tolerance: float,
) -> QCReadGeometry:
    normalized_downsamples = [
        (float(value[0]), float(value[1]))
        if isinstance(value, tuple)
        else (float(value), float(value))
        for value in level_downsamples
    ]
    selection = select_level(
        requested_spacing_um=float(qc_spacing_um),
        level0_spacing_um=float(base_spacing_um),
        level_downsamples=normalized_downsamples,
        tolerance=float(tolerance),
    )
    physical_tile_um = float(target_tile_size_px) * float(target_spacing_um)
    qc_tile_size_px = max(1, int(round(physical_tile_um / float(qc_spacing_um))))
    read_tile_size_px = max(
        1,
        int(round(physical_tile_um / float(selection.effective_spacing_um))),
    )
    return QCReadGeometry(
        level=int(selection.level),
        read_spacing_um=float(selection.effective_spacing_um),
        requested_spacing_um=float(qc_spacing_um),
        read_tile_size_px=read_tile_size_px,
        qc_tile_size_px=qc_tile_size_px,
    )


def _normalize_window(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window)
    if window.ndim == 2:
        return np.repeat(window[:, :, None], 3, axis=2)
    if window.shape[2] > 3:
        return window[:, :, :3]
    return window


def _resize_tile_for_qc(
    tile: np.ndarray,
    *,
    target_size: int,
) -> np.ndarray:
    if tile.shape[0] == target_size and tile.shape[1] == target_size:
        return tile
    interpolation = cv2.INTER_AREA
    if target_size > tile.shape[0] or target_size > tile.shape[1]:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(tile, (target_size, target_size), interpolation=interpolation)


def _resize_valid_mask(valid_mask: np.ndarray, *, target_size: int) -> np.ndarray:
    if valid_mask.shape[0] == target_size and valid_mask.shape[1] == target_size:
        return valid_mask
    resized = cv2.resize(
        valid_mask.astype(np.uint8),
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def filter_coordinate_tiles(
    *,
    coord_candidates: np.ndarray,
    keep_flags,
    level_dimensions: list[tuple[int, int]],
    level_downsamples: list[float] | list[tuple[float, float]],
    target_tile_size_px: int,
    target_spacing_um: float,
    base_spacing_um: float,
    tolerance: float,
    filter_params,
    read_window: Callable[[int, int, int, int, int], np.ndarray],
    batch_read_windows: Callable[
        [list[tuple[int, int]], tuple[int, int], int, int], Iterable[np.ndarray]
    ]
    | None = None,
    num_workers: int = 1,
    source_label: str = "<unknown-slide>",
) -> np.ndarray:
    keep_array = np.asarray(keep_flags, dtype=np.uint8).copy()
    if not needs_pixel_qc(filter_params):
        return keep_array

    active_indices = np.flatnonzero(keep_array)
    if active_indices.size == 0:
        return keep_array

    geometry = resolve_qc_read_geometry(
        target_tile_size_px=int(target_tile_size_px),
        target_spacing_um=float(target_spacing_um),
        qc_spacing_um=float(filter_params.qc_spacing_um),
        base_spacing_um=float(base_spacing_um),
        level_downsamples=level_downsamples,
        tolerance=float(tolerance),
    )
    img_w, img_h = level_dimensions[geometry.level]
    downsample = level_downsamples[geometry.level]
    if isinstance(downsample, tuple):
        downsample_x = float(downsample[0])
        downsample_y = float(downsample[1])
    else:
        downsample_x = float(downsample)
        downsample_y = float(downsample)

    level_coords = project_discrete_grid_origins(
        np.asarray(coord_candidates, dtype=np.int64),
        scale_x=1.0 / float(downsample_x),
        scale_y=1.0 / float(downsample_y),
    )

    supertile_span = int(geometry.read_tile_size_px) * 8
    batched_indices: dict[tuple[int, int], list[int]] = {}
    for idx in active_indices.tolist():
        x_level, y_level = level_coords[idx]
        batch_key = (
            int(x_level // supertile_span),
            int(y_level // supertile_span),
        )
        batched_indices.setdefault(batch_key, []).append(idx)

    error_count = 0
    error_samples: list[str] = []
    batch_requests: list[tuple[tuple[int, int], tuple[int, int], tuple[int, ...], int, int]] = []
    for (batch_x, batch_y), batch_member_indices in batched_indices.items():
        x0_level = int(batch_x * supertile_span)
        y0_level = int(batch_y * supertile_span)
        read_width = int(
            min(supertile_span + geometry.read_tile_size_px, max(0, img_w - x0_level))
        )
        read_height = int(
            min(supertile_span + geometry.read_tile_size_px, max(0, img_h - y0_level))
        )
        batch_requests.append(
            (
                (int(round(x0_level * float(downsample_x))), int(round(y0_level * float(downsample_y)))),
                (read_width, read_height),
                tuple(batch_member_indices),
                x0_level,
                y0_level,
            )
        )

    def _consume_window(
        window: np.ndarray,
        *,
        batch_member_indices: tuple[int, ...],
        x0_level: int,
        y0_level: int,
    ) -> None:
        window = _normalize_window(window)
        tile_dtype = window.dtype
        for coord_idx in batch_member_indices:
            x_level, y_level = level_coords[coord_idx]
            offset_x = int(x_level - x0_level)
            offset_y = int(y_level - y0_level)
            tile_arr = np.zeros(
                (geometry.read_tile_size_px, geometry.read_tile_size_px, 3),
                dtype=tile_dtype,
            )
            tile_view = window[
                offset_y : offset_y + geometry.read_tile_size_px,
                offset_x : offset_x + geometry.read_tile_size_px,
                :3,
            ]
            height = min(geometry.read_tile_size_px, tile_view.shape[0])
            width = min(geometry.read_tile_size_px, tile_view.shape[1])
            if height > 0 and width > 0:
                tile_arr[:height, :width, :] = tile_view[:height, :width]

            valid_mask = np.zeros(
                (geometry.read_tile_size_px, geometry.read_tile_size_px),
                dtype=bool,
            )
            valid_w = int(min(geometry.read_tile_size_px, max(0, img_w - x_level)))
            valid_h = int(min(geometry.read_tile_size_px, max(0, img_h - y_level)))
            if valid_h > 0 and valid_w > 0:
                valid_mask[:valid_h, :valid_w] = True

            if geometry.read_tile_size_px != geometry.qc_tile_size_px:
                tile_arr = _resize_tile_for_qc(
                    tile_arr,
                    target_size=geometry.qc_tile_size_px,
                )
                valid_mask = _resize_valid_mask(
                    valid_mask,
                    target_size=geometry.qc_tile_size_px,
                )

            keep_array[coord_idx] = np.uint8(
                apply_tile_qc(
                    tile_arr,
                    valid_mask=valid_mask,
                    filter_params=filter_params,
                )
            )

    processed_request_ids: set[int] = set()
    if batch_read_windows is not None:
        requests_by_size: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int], tuple[int, ...], int, int]]] = {}
        for request in batch_requests:
            requests_by_size.setdefault(request[1], []).append(request)
        for size, requests in requests_by_size.items():
            try:
                regions = batch_read_windows(
                    [request[0] for request in requests],
                    size,
                    int(geometry.level),
                    int(max(1, num_workers)),
                )
                for request, region in zip(requests, regions):
                    processed_request_ids.add(id(request))
                    _consume_window(
                        region,
                        batch_member_indices=request[2],
                        x0_level=int(request[3]),
                        y0_level=int(request[4]),
                    )
            except Exception as exc:
                if len(error_samples) < 3:
                    error_samples.append(str(exc))
    for request in batch_requests:
        if id(request) in processed_request_ids:
            continue
        try:
            window = read_window(
                request[0][0],
                request[0][1],
                request[1][0],
                request[1][1],
                int(geometry.level),
            )
        except Exception as exc:
            error_count += len(request[2])
            if len(error_samples) < 3:
                error_samples.append(str(exc))
            continue
        _consume_window(
            window,
            batch_member_indices=request[2],
            x0_level=int(request[3]),
            y0_level=int(request[4]),
        )

    if error_count > 0:
        import warnings

        sample_msg = "; ".join(error_samples)
        warnings.warn(
            f"Encountered {error_count} tile filtering error(s) on slide {source_label}. "
            f"Keeping affected tile(s). Sample error(s): {sample_msg}",
            UserWarning,
        )
    return keep_array
