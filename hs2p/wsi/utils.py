import cv2
import numpy as np


class HasEnoughTissue(object):
    def __init__(self, contour, contour_holes, tissue_mask, tile_size, scale, pct=0.01):
        self.cont = contour
        self.holes = contour_holes
        self.mask = tissue_mask // 255
        self.tile_size = tile_size
        self.scale = scale
        self.pct = pct

        # Precompute the combined tissue mask
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
        cv2.drawContours(contour_mask, self.holes, 0, 0, -1)

        # Combine with the tissue mask
        return cv2.bitwise_and(self.mask, contour_mask)

    def __call__(self, pt):
        """
        Check if a single tile at the given point has enough tissue.

        Args:
            pt (tuple): The (x, y) coordinates of the top-left corner of the tile.

        Returns:
            tuple: (keep_flag, tissue_pct), where:
                - keep_flag is 1 if the tile has enough tissue, otherwise 0.
                - tissue_pct is the percentage of tissue in the tile.
        """
        downsampled_tile_size = int(round(self.tile_size * 1 / self.scale[0], 0))
        assert (
            downsampled_tile_size > 0
        ), "downsampled tile_size is equal to zero, aborting; please consider using a smaller seg_params.downsample parameter"
        downsampled_pt = pt * 1 / self.scale[0]
        x_tile, y_tile = map(int, downsampled_pt)

        # Extract the sub-mask for the tile
        sub_mask = self.precomputed_mask[
            y_tile : y_tile + downsampled_tile_size,
            x_tile : x_tile + downsampled_tile_size,
        ]

        tile_area = downsampled_tile_size**2
        tissue_area = np.sum(sub_mask)
        tissue_pct = round(tissue_area / tile_area, 3)

        if tissue_pct >= self.pct:
            return 1, tissue_pct
        else:
            return 0, tissue_pct

    def check_coordinates(self, coords):
        """
        Check multiple tile coordinates for tissue coverage in a vectorized manner.

        Args:
            coords (np.ndarray): An array of shape (N, 2), where each row is (x, y).

        Returns:
            tuple: (keep_flags, tissue_pcts), where:
                - keep_flags is a list of 1s and 0s indicating whether each tile has enough tissue.
                - tissue_pcts is a list of tissue percentages for each tile.
        """
        downsampled_tile_size = int(round(self.tile_size * 1 / self.scale[0], 0))
        assert (
            downsampled_tile_size > 0
        ), "downsampled tile_size is equal to zero, aborting; please consider using a smaller seg_params.downsample parameter"

        # Downsample coordinates
        downsampled_coords = coords * 1 / self.scale[0]
        downsampled_coords = downsampled_coords.astype(int)

        keep_flags = []
        tissue_pcts = []

        for x_tile, y_tile in downsampled_coords:
            # Extract the sub-mask for the tile
            sub_mask = self.precomputed_mask[
                y_tile : y_tile + downsampled_tile_size,
                x_tile : x_tile + downsampled_tile_size,
            ]

            tile_area = downsampled_tile_size**2
            tissue_area = np.sum(sub_mask)
            tissue_pct = round(tissue_area / tile_area, 3)

            if tissue_pct >= self.pct:
                keep_flags.append(1)
                tissue_pcts.append(tissue_pct)
            else:
                keep_flags.append(0)
                tissue_pcts.append(tissue_pct)

        return keep_flags, tissue_pcts
