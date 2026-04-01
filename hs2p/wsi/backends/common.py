
from dataclasses import dataclass

import numpy as np

WHITE_RGB = 255


def make_white_canvas(width: int, height: int) -> np.ndarray:
    return np.full((int(height), int(width), 3), WHITE_RGB, dtype=np.uint8)


@dataclass(frozen=True)
class PaddedReadBounds:
    canvas: np.ndarray
    read_location: tuple[int, int]
    read_size: tuple[int, int]
    paste_offset: tuple[int, int]


def resolve_padded_read_bounds(
    *,
    location: tuple[int, int],
    size: tuple[int, int],
    level_dimensions: tuple[int, int],
    downsample: float,
) -> PaddedReadBounds:
    width, height = int(size[0]), int(size[1])
    canvas = make_white_canvas(width, height)
    if width <= 0 or height <= 0:
        return PaddedReadBounds(canvas, (0, 0), (0, 0), (0, 0))

    x_level = int(np.floor(location[0] / downsample))
    y_level = int(np.floor(location[1] / downsample))
    x1 = max(x_level, 0)
    y1 = max(y_level, 0)
    x2 = min(x_level + width, int(level_dimensions[0]))
    y2 = min(y_level + height, int(level_dimensions[1]))
    if x2 <= x1 or y2 <= y1:
        return PaddedReadBounds(canvas, (0, 0), (0, 0), (0, 0))

    read_location = (int(round(x1 * downsample)), int(round(y1 * downsample)))
    read_size = (x2 - x1, y2 - y1)
    paste_offset = (x1 - x_level, y1 - y_level)
    return PaddedReadBounds(canvas, read_location, read_size, paste_offset)


def paste_region(
    canvas: np.ndarray,
    region: np.ndarray,
    *,
    paste_offset: tuple[int, int],
) -> np.ndarray:
    read_height, read_width = region.shape[:2]
    if read_width <= 0 or read_height <= 0:
        return canvas
    paste_x, paste_y = paste_offset
    canvas[
        int(paste_y) : int(paste_y) + int(read_height),
        int(paste_x) : int(paste_x) + int(read_width),
    ] = region
    return canvas
