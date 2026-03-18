from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ResolvedTileGeometry:
    target_tile_size_px: int
    read_spacing_um: float
    resize_factor: float
    seg_spacing_um: float
    level0_spacing_um: float

    @property
    def target_spacing_um(self) -> float:
        return self.read_spacing_um * self.resize_factor

    @property
    def downsampled_tile_size_px(self) -> int:
        scale = self.seg_spacing_um / self.target_spacing_um
        return int(round(self.target_tile_size_px / scale, 0))

    @property
    def resized_tile_size_px(self) -> int:
        return int(round(self.target_tile_size_px * self.resize_factor, 0))

    @property
    def level0_to_seg_scale(self) -> float:
        return self.seg_spacing_um / self.level0_spacing_um


class HasEnoughTissue(object):
    def __init__(
        self,
        contour,
        contour_holes,
        tissue_mask,
        geometry: ResolvedTileGeometry,
        pct=0.01,
    ):
        self.cont = contour
        self.holes = contour_holes
        self.mask = tissue_mask // 255
        self.geometry = geometry
        self.pct = pct

        self.downsampled_tile_size = self.geometry.downsampled_tile_size_px
        assert (
            self.downsampled_tile_size > 0
        ), "downsampled tile_size is equal to zero, aborting; please consider using a smaller seg_params.downsample parameter"

        self.tile_size_resized = self.geometry.resized_tile_size_px

        # precompute the combined tissue mask
        self.precomputed_mask = self._precompute_tissue_mask()

    def _precompute_tissue_mask(self):
        """
        Precompute a binary mask for the entire region, combining the contour and holes.

        Returns:
            np.ndarray: A binary mask where tissue regions are 1 and non-tissue regions are 0.
        """
        contour_mask = np.zeros_like(self.mask, dtype=np.uint8)

        # Draw white filled contour on black background
        cv2.drawContours(contour_mask, [self.cont], 0, 255, -1)

        # Draw black filled holes on white filled contour
        cv2.drawContours(contour_mask, self.holes, -1, 0, -1)

        # Combine with the tissue mask
        return cv2.bitwise_and(self.mask, contour_mask)

    def _extract_sub_mask(self, x_tile, y_tile):
        """
        Extract the sub-mask for a tile at the given downsampled coordinates.
        """
        return self.precomputed_mask[
            y_tile : y_tile + self.downsampled_tile_size,
            x_tile : x_tile + self.downsampled_tile_size,
        ]

    def check_coordinates(self, coords):
        """
        Check multiple tile coordinates for tissue coverage.

        Args:
            coords (np.ndarray): An array of shape (N, 2), where each row is (x, y).

        Returns:
            tuple: (keep_flags, tissue_pcts), where:
                - keep_flags is a list of 1s and 0s indicating whether each tile has enough tissue.
                - tissue_pcts is a list of tissue percentages for each tile.
        """
        # downsample coordinates from level 0 to seg_level
        scale = self.geometry.level0_to_seg_scale
        downsampled_coords = coords * 1 / scale
        downsampled_coords = downsampled_coords.astype(int)

        tile_area = float(self.downsampled_tile_size**2)
        height, width = self.precomputed_mask.shape
        x1 = np.clip(downsampled_coords[:, 0], 0, width)
        y1 = np.clip(downsampled_coords[:, 1], 0, height)
        x2 = np.clip(x1 + self.downsampled_tile_size, 0, width)
        y2 = np.clip(y1 + self.downsampled_tile_size, 0, height)

        integral = cv2.integral(
            self.precomputed_mask.astype(np.uint8), sdepth=cv2.CV_32S
        )
        tissue_area = (
            integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        ).astype(np.float32)
        tissue_pcts = np.round(tissue_area / tile_area, 3).astype(np.float32)
        keep_flags = (tissue_pcts >= self.pct).astype(np.uint8)

        return keep_flags, tissue_pcts

    def get_tile_mask(self, x, y):
        """
        Get the binary mask for a single tile at the given coordinates.

        Args:
            x (int): The x-coordinate of the top-left corner of the tile.
            y (int): The y-coordinate of the top-left corner of the tile.

        Returns:
            np.ndarray: The binary mask for the tile (0 or 1).
        """
        # downsample coordinates from level 0 to seg_level
        scale = self.geometry.level0_to_seg_scale
        x_tile = int(x / scale)
        y_tile = int(y / scale)

        # extract the sub-mask for the tile
        sub_mask = self._extract_sub_mask(x_tile, y_tile)

        # handle edge cases where sub_mask is smaller than expected
        if (
            sub_mask.shape[0] != self.downsampled_tile_size
            or sub_mask.shape[1] != self.downsampled_tile_size
        ):
            padded_mask = np.zeros(
                (self.downsampled_tile_size, self.downsampled_tile_size),
                dtype=sub_mask.dtype,
            )
            padded_mask[: sub_mask.shape[0], : sub_mask.shape[1]] = sub_mask
            sub_mask = padded_mask

        # upsample the mask to the original tile size
        mask = cv2.resize(
            sub_mask,
            (self.tile_size_resized, self.tile_size_resized),
            interpolation=cv2.INTER_NEAREST,
        )
        return mask
