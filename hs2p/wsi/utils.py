import cv2
import numpy as np


class HasEnoughTissue(object):
    def __init__(self, contour, contour_holes, tissue_mask, tile_size, tile_spacing, resize_factor, seg_spacing, spacing_at_level_0, pct=0.01):
        self.cont = contour
        self.holes = contour_holes
        self.mask = tissue_mask // 255
        self.tile_size = tile_size
        self.tile_spacing = tile_spacing
        self.resize_factor = resize_factor
        self.seg_spacing = seg_spacing
        self.spacing_at_level_0 = spacing_at_level_0
        self.pct = pct

        # downsample tile size from target_spacing to seg_spacing
        # where contour and tissue masks are defined
        target_spacing = self.tile_spacing * self.resize_factor
        scale = self.seg_spacing / target_spacing
        self.downsampled_tile_size = int(round(self.tile_size * 1 / scale, 0))
        assert (
            self.downsampled_tile_size > 0
        ), "downsampled tile_size is equal to zero, aborting; please consider using a smaller seg_params.downsample parameter"

        self.tile_size_resized = int(round(tile_size * resize_factor,0))

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
        cv2.drawContours(contour_mask, self.holes, 0, 0, -1)

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
        Check multiple tile coordinates for tissue coverage in a vectorized manner.

        Args:
            coords (np.ndarray): An array of shape (N, 2), where each row is (x, y).

        Returns:
            tuple: (keep_flags, tissue_pcts), where:
                - keep_flags is a list of 1s and 0s indicating whether each tile has enough tissue.
                - tissue_pcts is a list of tissue percentages for each tile.
        """
        # downsample coordinates from level 0 to seg_level
        scale = self.seg_spacing / self.spacing_at_level_0
        downsampled_coords = coords * 1 / scale
        downsampled_coords = downsampled_coords.astype(int)

        keep_flags = []
        tissue_pcts = []

        for x_tile, y_tile in downsampled_coords:
            # extract the sub-mask for the tile
            sub_mask = self._extract_sub_mask(x_tile, y_tile)

            tile_area = self.downsampled_tile_size**2
            tissue_area = np.sum(sub_mask)
            tissue_pct = round(tissue_area / tile_area, 3)

            if tissue_pct >= self.pct:
                keep_flags.append(1)
                tissue_pcts.append(tissue_pct)
            else:
                keep_flags.append(0)
                tissue_pcts.append(tissue_pct)

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
        scale = self.seg_spacing / self.spacing_at_level_0
        x_tile = int(x / scale)
        y_tile = int(y / scale)

        # extract the sub-mask for the tile
        sub_mask = self._extract_sub_mask(x_tile, y_tile)

        # handle edge cases where sub_mask is smaller than expected
        if sub_mask.shape[0] != self.downsampled_tile_size or sub_mask.shape[1] != self.downsampled_tile_size:
            padded_mask = np.zeros((self.downsampled_tile_size, self.downsampled_tile_size), dtype=sub_mask.dtype)
            padded_mask[:sub_mask.shape[0], :sub_mask.shape[1]] = sub_mask
            sub_mask = padded_mask

        # upsample the mask to the original tile size
        mask = cv2.resize(sub_mask, (self.tile_size_resized, self.tile_size_resized), interpolation=cv2.INTER_NEAREST)
        return mask
