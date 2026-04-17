"""Integral-image tile coverage computation."""

import cv2
import numpy as np

from hs2p.wsi.geometry import project_discrete_grid_origins


def compute_tile_coverage(
    candidates: np.ndarray,
    binary_mask: np.ndarray,
    tile_size_lv0: int,
    slide_dimensions: tuple[int, int],
) -> np.ndarray:
    """Compute the tissue-covered fraction for each candidate tile.

    Uses an integral image for O(1) per-tile summation.

    Args:
        candidates: (N, 2) int64 array of (x, y) tile origins in level-0 pixel space.
        binary_mask: 2-D uint8 mask in mask/segmentation space. Any non-zero pixel
            counts as tissue.
        tile_size_lv0: tile side length in level-0 pixels (square tiles assumed).
        slide_dimensions: (width, height) of the slide in level-0 pixels.

    Returns:
        (N,) float32 array with values in [0, 1].
    """
    mask_h, mask_w = binary_mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = mask_w / slide_w
    scale_y = mask_h / slide_h

    binary = (binary_mask > 0).astype(np.float64)
    integral = cv2.integral(binary)

    tile_w_mask = max(1, round(tile_size_lv0 * scale_x))
    tile_h_mask = max(1, round(tile_size_lv0 * scale_y))
    tile_area = tile_w_mask * tile_h_mask

    mask_origins = project_discrete_grid_origins(
        candidates,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    x1 = np.clip(mask_origins[:, 0], 0, mask_w)
    y1 = np.clip(mask_origins[:, 1], 0, mask_h)
    x2 = np.clip(mask_origins[:, 0] + tile_w_mask, 0, mask_w)
    y2 = np.clip(mask_origins[:, 1] + tile_h_mask, 0, mask_h)

    tissue_sum = (
        integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    )
    return np.clip((tissue_sum / tile_area).astype(np.float32), 0.0, 1.0)


__all__ = ["compute_tile_coverage"]
