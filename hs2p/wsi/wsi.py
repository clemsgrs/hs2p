from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hs2p.wsi.backend import open_slide, resolve_backend
from hs2p.wsi.geometry import plan_spacing_read, select_level

Image.MAX_IMAGE_PIXELS = 933120000

# String aliases for cv2 interpolation flags. ``nearest`` is mandatory for label
# masks (any averaging interpolation invents class indices); ``area`` is the
# quality default for image downscaling.
INTERPOLATION_FLAGS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def resize_array(
    arr: np.ndarray, target_size: tuple[int, int], *, interpolation: str
) -> np.ndarray:
    """Resize ``arr`` to ``target_size`` (width, height) using a named interpolation.

    ``target_size`` matching the current shape is a no-op (returns ``arr`` unchanged),
    so an exact spacing/level match is lossless.
    """
    if interpolation not in INTERPOLATION_FLAGS:
        raise ValueError(
            f"unknown interpolation {interpolation!r}; expected one of "
            f"{sorted(INTERPOLATION_FLAGS)}"
        )
    target_width, target_height = int(target_size[0]), int(target_size[1])
    if arr.shape[1] == target_width and arr.shape[0] == target_height:
        return arr
    return cv2.resize(
        arr, (target_width, target_height), interpolation=INTERPOLATION_FLAGS[interpolation]
    )


class WSI(object):
    """
    A class for handling Whole Slide Images (wsi).
    Attributes:
        path (Path): Full path to the wsi.
        name (str): Name of the wsi (stem of the path).
        fmt (str): File format of the wsi.
        reader: slide reader object.
        spacing_at_level_0 (float): Manually set spacing at level 0.
        spacings (list[float]): List of spacings for each level.
        level_dimensions (list[tuple[int, int]]): Dimensions at each level.
        level_downsamples (list[tuple[float, float]]): Downsample factors for each level.
        backend (str): Backend used for opening the wsi (default: "asap").
        mask_path (Path, optional): Path to the segmentation mask.
        mask_reader: Segmentation mask reader object.
    """

    def __init__(
        self,
        path: Path,
        backend: str,
        mask_path: Path | None = None,
        spacing_at_level_0: float | None = None,
    ):
        self.path = path
        self.name = path.stem.replace(" ", "_")
        self.fmt = path.suffix
        self.requested_backend = backend
        selection = resolve_backend(backend, wsi_path=path, mask_path=mask_path)
        self.backend = selection.backend
        self.reader = open_slide(
            path,
            backend=self.backend,
            spacing_override=spacing_at_level_0,
        )

        self._level_spacing_cache = {}

        self.spacing_at_level_0 = spacing_at_level_0
        self.raw_spacings = list(self.reader.spacings)
        self.spacings = self.get_spacings()
        self.level_dimensions = list(self.reader.level_dimensions)
        self.level_downsamples = list(self.reader.level_downsamples)

        self.mask_path = mask_path
        self.mask_reader = None
        if mask_path is not None:
            self.mask_reader = open_slide(
                mask_path,
                backend=self.backend,
            )

    def get_slide(self, level: int) -> np.ndarray:
        """Return the full slide image at the given pyramid level."""
        return np.asarray(self.reader.read_level(level))

    def get_tile(self, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
        """
        Extracts a tile from a whole slide image at the specified coordinates and pyramid level.

        Args:
            x (int): The x-coordinate of the top-left corner of the tile (level-0 pixel space).
            y (int): The y-coordinate of the top-left corner of the tile (level-0 pixel space).
            width (int): Tile width in pixels at the target level.
            height (int): Tile height in pixels at the target level.
            level (int): Pyramid level to read from.

        Returns:
            numpy.ndarray: The extracted tile as a numpy array.
        """
        return np.asarray(
            self.reader.read_region(
                (int(x), int(y)),
                int(level),
                (int(width), int(height)),
            )
        )

    def get_spacings(self):
        """
        Retrieve the spacings for the whole slide image.

        If the `spacing_at_level_0` is not set, the method returns the original spacings
        from the wsi. Otherwise, it calculates adjusted spacings based on the provided
        value and the original spacings.
        """
        if self.spacing_at_level_0 is None:
            return self.raw_spacings
        return [
            self.spacing_at_level_0 * s / self.raw_spacings[0]
            for s in self.raw_spacings
        ]

    def get_level_spacing(self, level: int):
        """
        Retrieve the spacing value for a specified level.

        Args:
            level (int): Level for which to retrieve the spacing.

        Returns:
            float: Spacing value corresponding to the specified level.
        """
        if level not in self._level_spacing_cache:
            self._level_spacing_cache[level] = self.spacings[level]
        return self._level_spacing_cache[level]

    def get_best_level_for_spacing(self, requested_spacing_um: float, tolerance: float):
        """
        Determines the best level in a multi-resolution image pyramid for a given target spacing.

        Ensures that the spacing of the returned level is either within the specified tolerance of the target
        spacing or smaller than the target spacing to avoid upsampling.

        Args:
            requested_spacing_um (float): Desired spacing.
            tolerance (float, optional): Tolerance for matching the spacing, deciding how much
                spacing can deviate from those specified in the slide metadata.

        Returns:
            level (int): Index of the best matching level in the image pyramid.
        """
        spacing = self.get_level_spacing(0)
        requested_downsample = requested_spacing_um / spacing
        level = self.get_best_level_for_downsample_custom(requested_downsample)
        level_spacing = self.get_level_spacing(level)

        is_within_tolerance = False
        if abs(level_spacing - requested_spacing_um) / requested_spacing_um <= tolerance:
            is_within_tolerance = True
            return level, is_within_tolerance

        else:
            while level > 0 and level_spacing > requested_spacing_um:
                level -= 1
                level_spacing = self.get_level_spacing(level)
                if abs(level_spacing - requested_spacing_um) / requested_spacing_um <= tolerance:
                    is_within_tolerance = True
                    break

        assert (
            level_spacing <= requested_spacing_um
            or abs(level_spacing - requested_spacing_um) / requested_spacing_um <= tolerance
        ), f"Unable to find a spacing less than or equal to the requested spacing ({requested_spacing_um}) or within {int(tolerance * 100)}% of the requested spacing."
        return level, is_within_tolerance

    def get_best_level_for_downsample_custom(self, downsample: int):
        """
        Determines the best level for a given downsample factor based on the available
        level downsample values.

        Args:
            downsample (float): Target downsample factor.

        Returns:
            int: Index of the best matching level for the given downsample factor.
        """
        level = int(np.argmin([abs(x - downsample) for x, _ in self.level_downsamples]))
        return level

    def read_region(self, location, level, size):
        """Read a region from the slide at the given level."""
        return self.reader.read_region(location, level, size)

    def read_region_at_spacing(
        self,
        location: tuple[int, int],
        requested_spacing_um: float,
        size: tuple[int, int],
        *,
        tolerance: float,
        interpolation: str,
    ) -> np.ndarray:
        """Read a region at an arbitrary spacing (level-select + downscale).

        Picks the finest pyramid level whose spacing is ``<= requested_spacing_um``
        (within ``tolerance``) — never upsampling — reads it natively, then downscales
        to ``size`` (width, height in px at ``requested_spacing_um``) with the named
        ``interpolation``. When a level matches the requested spacing exactly there is
        no resize (lossless).

        Args:
            location: ``(x, y)`` top-left in level-0 pixel space.
            requested_spacing_um: Target spacing (µm/px).
            size: Output ``(width, height)`` in pixels at the requested spacing.
            tolerance: Relative spacing tolerance for the level match (no default —
                the caller must state it).
            interpolation: One of :data:`INTERPOLATION_FLAGS`, required so the caller
                consciously picks it (``"nearest"`` for label masks, ``"area"`` for
                image downscaling).
        """
        plan = plan_spacing_read(
            requested_spacing_um=float(requested_spacing_um),
            level0_spacing_um=float(self.get_level_spacing(0)),
            level_downsamples=self.level_downsamples,
            target_size_px=(int(size[0]), int(size[1])),
            tolerance=float(tolerance),
        )
        region = self.read_region(location, plan.level, plan.read_size_px)
        return resize_array(region, (int(size[0]), int(size[1])), interpolation=interpolation)

    def read_full_at_spacing(
        self,
        requested_spacing_um: float,
        *,
        tolerance: float,
        interpolation: str,
    ) -> np.ndarray:
        """Read the entire image resampled to ``requested_spacing_um``.

        Convenience over :meth:`read_region_at_spacing` for the "read this whole
        (small) image at a target spacing" case (e.g. a pre-cropped ROI tile): full
        native read of the finest level ``<=`` the request, then downscale by
        ``level_spacing / requested_spacing_um``. Exact-level match ⇒ no resize.
        """
        sel = select_level(
            requested_spacing_um=float(requested_spacing_um),
            level0_spacing_um=float(self.get_level_spacing(0)),
            level_downsamples=self.level_downsamples,
            tolerance=float(tolerance),
        )
        arr = self.get_slide(sel.level)
        if sel.is_within_tolerance:
            # Level matches the request (within tolerance) — read native, no resize.
            return arr
        level_height, level_width = arr.shape[0], arr.shape[1]
        scale = float(sel.read_spacing_um) / float(requested_spacing_um)  # < 1 (downscale)
        target_size = (int(round(level_width * scale)), int(round(level_height * scale)))
        return resize_array(arr, target_size, interpolation=interpolation)
