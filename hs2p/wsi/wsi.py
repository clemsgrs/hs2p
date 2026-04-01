import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import tqdm
from PIL import Image

from hs2p.configs import FilterConfig, SegmentationConfig, TilingConfig
from hs2p.wsi.backend import open_slide, resolve_backend
from hs2p.wsi.types import SamplingSpec
from hs2p.wsi.utils import TissueFilter, ResolvedGeometry

Image.MAX_IMAGE_PIXELS = 933120000


class WSI(object):
    """
    A class for handling Whole Slide Images (wsi) and tile extraction.
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
        seg_level (int): Level for segmentation.
        binary_mask (np.ndarray): Binary segmentation mask as a numpy array.
    """

    def __init__(
        self,
        path: Path,
        backend: str,
        mask_path: Path | None = None,
        spacing_at_level_0: float | None = None,
        segment: bool = False,
        segment_params: SegmentationConfig | None = None,
        sampling_spec: SamplingSpec | None = None,
        pixel_mapping: dict | None = None,
    ):
        """
        Initializes a Whole Slide Image object with optional mask and spacing.

        Args:
            path (Path): Path to the wsi.
            mask_path (Path, optional): Path to the tissue mask, if available. Defaults to None.
            spacing_at_level_0 (float, optional): Manually set spacing at level 0, if speficied. Defaults to None.
            backend (str): Backend to use for opening the wsi.
            segment (bool): Whether to segment the slide if tissue mask is not provided. Defaults to False.
            segment_params (SegmentationConfig, optional): Segmentation parameters used for either loading an existing tissue mask or segmenting the slide.
            sampling_spec (SamplingSpec, optional): Normalized sampling specification
                used when loading an existing annotation mask.
        """

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

        self._scaled_contours_cache = {}  # add a cache for scaled contours
        self._scaled_holes_cache = {}  # add a cache for scaled holes
        self._level_spacing_cache = {}  # add a cache for level spacings

        self.spacing_at_level_0 = spacing_at_level_0  # manually set spacing at level 0
        self.raw_spacings = list(self.reader.spacings)  # baseline reader pyramid spacings; metadata-derived when available
        self.spacings = self.get_spacings()  # physical spacings, possibly overridden via spacing_at_level_0
        self.level_dimensions = list(self.reader.level_dimensions)
        self.level_downsamples = list(self.reader.level_downsamples)
        self.pixel_mapping = pixel_mapping

        self.mask_path = mask_path
        self.mask_reader = None
        if mask_path is not None:
            if sampling_spec is None:
                raise ValueError(
                    "sampling_spec is required when loading a mask-backed slide"
                )
            self.mask_reader = open_slide(
                mask_path,
                backend=self.backend,
            )
            self.seg_level = self.load_segmentation(
                segment_params,
                sampling_spec=sampling_spec,
            )
        elif segment:
            self.seg_level = self.segment_tissue(segment_params)

        if sampling_spec is not None:
            self.annotation_pct = sampling_spec.tissue_percentage

    def get_slide(self, level: int) -> np.ndarray:
        """Return the full slide image at the given pyramid level.

        Uses the selected reader level directly so backend selection stays
        encapsulated behind the reader contract.
        """
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

        If the `spacing` attribute is not set, the method returns the original spacings
        from the wsi. Otherwise, it calculates adjusted spacings based on the provided
        `spacing` value and the original spacings.

        Returns:
            list: A list of spacings, either the original or adjusted based on the
            `spacing` attribute.
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

    def get_best_level_for_spacing(self, target_spacing: float, tolerance: float):
        """
        Determines the best level in a multi-resolution image pyramid for a given target spacing.

        Ensures that the spacing of the returned level is either within the specified tolerance of the target
        spacing or smaller than the target spacing to avoid upsampling.

        Args:
            target_spacing (float): Desired spacing.
            tolerance (float, optional): Tolerance for matching the spacing, deciding how much
                spacing can deviate from those specified in the slide metadata.

        Returns:
            level (int): Index of the best matching level in the image pyramid.
        """
        spacing = self.get_level_spacing(0)
        target_downsample = target_spacing / spacing
        level = self.get_best_level_for_downsample_custom(target_downsample)
        level_spacing = self.get_level_spacing(level)

        # check if the level_spacing is within the tolerance of the target_spacing
        is_within_tolerance = False
        if abs(level_spacing - target_spacing) / target_spacing <= tolerance:
            is_within_tolerance = True
            return level, is_within_tolerance

        # otherwise, look for a spacing smaller than or equal to the target_spacing
        else:
            while level > 0 and level_spacing > target_spacing:
                level -= 1
                level_spacing = self.get_level_spacing(level)
                if abs(level_spacing - target_spacing) / target_spacing <= tolerance:
                    is_within_tolerance = True
                    break

        assert (
            level_spacing <= target_spacing
            or abs(level_spacing - target_spacing) / target_spacing <= tolerance
        ), f"Unable to find a spacing less than or equal to the target spacing ({target_spacing}) or within {int(tolerance * 100)}% of the target spacing."
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

    def load_segmentation(
        self,
        segment_params: SegmentationConfig,
        sampling_spec: SamplingSpec,
    ):
        """
        Load and process a segmentation mask for a whole slide image.

        This method determines the best level for the given downsample factor, and
        processes the segmentation mask to create a binary mask.

        Args:
            downsample (int): Downsample factor for finding best level for tissue segmentation.
            sthresh_up (int, optional): Upper threshold value for scaling the binary
                mask. Defaults to 255.
            tissue_pixel_value (int, optional): Pixel value in the segmentation mask that
                represents tissue. Defaults to 1.

        Returns:
            int: Level at which the tissue mask was loaded.
        """
        if self.mask_reader is None:
            raise ValueError("mask_reader is required for load_segmentation()")
        mask_spacings = list(self.mask_reader.spacings)
        mask_level_downsamples = list(self.mask_reader.level_downsamples)
        mask_spacing_at_level_0 = mask_spacings[0]
        seg_level = self.get_best_level_for_downsample_custom(segment_params.downsample)
        seg_spacing = self.get_level_spacing(seg_level)

        mask_downsample = seg_spacing / mask_spacing_at_level_0
        mask_level = int(
            np.argmin([abs(x - mask_downsample) for x, _ in mask_level_downsamples])
        )
        mask_spacing = mask_spacings[mask_level]

        scale = seg_spacing / mask_spacing
        while scale < 1 and mask_level > 0:
            mask_level -= 1
            mask_spacing = mask_spacings[mask_level]
            scale = seg_spacing / mask_spacing

        mask = np.asarray(self.mask_reader.read_level(mask_level))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # WSD/OpenSlide may return non-integer values when reading pyramid levels
        # (bilinear resampling from level 0 instead of reading stored pages).
        # Fall back to a direct PIL read of the correct pyramid page to recover
        # the exact label values defined in sampling_spec.pixel_mapping.
        known_values = set(sampling_spec.pixel_mapping.values())
        if not set(np.unique(mask).tolist()).issubset(known_values):
            with Image.open(self.mask_path) as mask_img:
                mask_img.seek(mask_level)
                mask = np.array(mask_img)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
        height, width = mask.shape

        # resize the mask to the size of the slide at seg_spacing
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(round(width / scale, 0)), int(round(height / scale, 0))),
            interpolation=cv2.INTER_NEAREST,
        )

        background = sampling_spec.pixel_mapping["background"]
        m = (mask != background).astype("uint8") * segment_params.sthresh_up
        self.annotation_mask = {"tissue": m}
        for annotation, val in sampling_spec.pixel_mapping.items():
            if annotation != "background":
                self.annotation_mask[annotation] = (mask == val).astype(
                    "uint8"
                ) * segment_params.sthresh_up

        return seg_level

    def segment_tissue(
        self,
        segment_params: SegmentationConfig,
    ):
        """
        Segment the tissue via HSV -> Median thresholding -> Binary thresholding -> Morphological closing.
        Or via HSV thresholding if use_hsv is True.

        Args:
            downsample (int): Downsample factor for finding best level for tissue segmentation.
            sthresh (int, optional): Lower threshold for binary thresholding. Defaults to 20.
            sthresh_up (int, optional): Upper threshold for binary thresholding. Defaults to 255.
            mthresh (int, optional): Kernel size for median blurring. Defaults to 7.
            close (int, optional): Size of the kernel for morphological closing.
                If 0, no morphological closing is applied. Defaults to 0.
            use_otsu (bool, optional): Whether to use Otsu's method for thresholding. Defaults to False.
            use_hsv (bool, optional): Whether to use HSV thresholding. Defaults to False.

        Returns:
            int: Level at which the tissue mask was created.
        """

        seg_level = self.get_best_level_for_downsample_custom(segment_params.downsample)

        img = np.asarray(self.get_slide(seg_level))
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert to HSV space

        if segment_params.use_hsv:
            # hsv thresholding
            lower = np.array([90, 8, 103])
            upper = np.array([180, 255, 255])
            img_thresh = cv2.inRange(img_hsv, lower, upper)

        else:
            img_med = cv2.medianBlur(
                img_hsv[:, :, 1], segment_params.mthresh
            )  # apply median blurring

            # thresholding
            if segment_params.use_otsu:
                _, img_thresh = cv2.threshold(
                    img_med,
                    0,
                    segment_params.sthresh_up,
                    cv2.THRESH_OTSU + cv2.THRESH_BINARY,
                )
            else:
                _, img_thresh = cv2.threshold(
                    img_med,
                    segment_params.sthresh,
                    segment_params.sthresh_up,
                    cv2.THRESH_BINARY,
                )

        # morphological closing
        if segment_params.close > 0:
            kernel = np.ones((segment_params.close, segment_params.close), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        self.annotation_mask = {"tissue": img_thresh}
        return seg_level

    def get_tile_coordinates(
        self,
        tiling_params: TilingConfig,
        filter_params: FilterConfig,
        annotation: str | None = None,
        disable_tqdm: bool = False,
        num_workers: int = 1,
    ):
        """
        Extract tile coordinates based on the specified target spacing, tile size, overlap,
        and additional tiling and filtering parameters.

        Args:
            tiling_params: Tiling configuration with spacing, tile size, overlap,
                tissue threshold, and edge-inclusive tile extraction.
                - target_spacing (float): Desired spacing of the tiles.
                - tolerance (float): Tolerance for matching the target_spacing, deciding how much
                    target_spacing can deviate from those specified in the slide metadata.
                - target_tile_size (int): Desired size of the tiles at the target spacing.
                - overlap (float, optional): Overlap between adjacent tiles. Defaults to 0.0.
                - "tissue_percentage" (dict[str, float]): Minimum amount pixels covered with tissue required for a tile for a given annotation.
            filter_params (NamedTuple): Parameters for filtering contours, including:
                - "ref_tile_size" (int): Reference tile size for filtering. Defaults to 256.
                - "a_t" (int): Contour area threshold for filtering. Defaults to 4.
                - "a_h" (int): Hole area threshold for filtering. Defaults to 2.
            num_workers (int, optional): Number of workers to use for parallel processing.
                Defaults to 1.

        Returns:
            tuple:
                - tile_coordinates (list[tuple[int, int]]): List of (x, y) coordinates for the extracted tiles.
                - tissue_percentages (list[float]): List of tissue percentages for each tile.
                - contour_indices (list[int]): List of contour indices for each tile.
                - tile_level (int): Level of the wsi used for tile extraction.
                - resize_factor (float): The factor by which the tile size was resized.
                - tile_size_lv0 (int): The tile size at level 0 of the wsi pyramid.
        """
        scale = tiling_params.target_spacing_um / self.get_level_spacing(0)
        tile_size_lv0 = int(round(tiling_params.target_tile_size_px * scale, 0))
        self._active_target_tile_size_px = int(tiling_params.target_tile_size_px)
        self._active_target_spacing_um = float(tiling_params.target_spacing_um)
        self._active_tolerance = float(tiling_params.tolerance)

        contours, holes = self.detect_contours(
            target_spacing=tiling_params.target_spacing_um,
            tolerance=tiling_params.tolerance,
            filter_params=filter_params,
            annotation=annotation,
        )
        (
            running_x_coords,
            running_y_coords,
            tissue_percentages,
            contour_indices,
            tile_level,
            resize_factor,
        ) = self.process_contours(
            contours,
            holes,
            tiling_params=tiling_params,
            filter_params=filter_params,
            annotation=annotation,
            disable_tqdm=disable_tqdm,
            num_workers=num_workers,
        )
        coord_candidates = np.column_stack([running_x_coords, running_y_coords])
        keep_flags = self.filter_black_and_white_tiles(
            np.ones(len(coord_candidates), dtype=np.uint8),
            coord_candidates,
            0,
            0,
            filter_params,
            num_workers=num_workers,
        )
        keep_mask = np.asarray(keep_flags, dtype=bool)
        running_x_coords = running_x_coords[keep_mask]
        running_y_coords = running_y_coords[keep_mask]
        tissue_percentages = tissue_percentages[keep_mask]
        contour_indices = contour_indices[keep_mask]
        tile_coordinates = list(zip(running_x_coords, running_y_coords))
        return (
            tile_coordinates,
            tissue_percentages,
            contour_indices,
            tile_level,
            resize_factor,
            tile_size_lv0,
        )

    def filter_black_and_white_tiles(
        self,
        keep_flags,
        coord_candidates,
        tile_size,
        tile_level,
        filter_params,
        num_workers: int = 1,
    ):
        target_tile_size_px = int(
            getattr(self, "_active_target_tile_size_px", int(tile_size))
        )
        target_spacing_um = float(
            getattr(self, "_active_target_spacing_um", self.get_level_spacing(tile_level))
        )
        tolerance = float(getattr(self, "_active_tolerance", 0.05))
        batch_read_windows = None
        if self.backend == "cucim":
            try:
                from hs2p.wsi.backends.cucim import CuCIMReader

                reader = CuCIMReader(
                    str(self.path),
                    spacing_override=float(self.get_level_spacing(0)),
                    gpu_decode=False,
                )
                batch_read_windows = (
                    lambda locations, size, level, workers: list(reader.read_regions(
                        locations,
                        int(level),
                        size,
                        num_workers=max(1, int(workers)),
                    ))
                )
            except ImportError:
                warnings.warn(
                    "CuCIM is unavailable for backend='cucim'; falling back to sequential tile filtering reads.",
                    UserWarning,
                )

        from hs2p.tile_qc import filter_coordinate_tiles  # local import to avoid circular dependency
        keep_array = filter_coordinate_tiles(
            coord_candidates=np.asarray(coord_candidates, dtype=np.int64),
            keep_flags=keep_flags,
            level_dimensions=self.level_dimensions,
            level_downsamples=self.level_downsamples,
            target_tile_size_px=target_tile_size_px,
            target_spacing_um=target_spacing_um,
            base_spacing_um=float(self.get_level_spacing(0)),
            tolerance=tolerance,
            filter_params=filter_params,
            read_window=self.get_tile,
            batch_read_windows=batch_read_windows,
            num_workers=num_workers,
            source_label=str(getattr(self, "path", "<unknown-slide>")),
        )
        return keep_array.tolist()

    @staticmethod
    def filter_contours(contours, hierarchy, filter_params: FilterConfig):
        """
        Filter contours by area using FilterConfig.
        """
        filtered = []

        # find indices of foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        all_holes = []

        # loop through foreground contour indices
        for cont_idx in hierarchy_1:
            # actual contour
            cont = contours[cont_idx]
            # indices of holes contained in this contour (children of parent contour)
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            # take contour area (includes holes)
            a = cv2.contourArea(cont)
            # calculate the contour area of each hole
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            # actual area of foreground contour region
            a = a - np.array(hole_areas).sum()
            if a == 0:
                continue
            if a > filter_params.a_t:  # Use named tuple instead of dictionary
                filtered.append(cont_idx)
                all_holes.append(holes)

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []
        for hole_ids in all_holes:
            hole_contours.append(
                [contours[idx] for idx in hole_ids if cv2.contourArea(contours[idx]) > filter_params.a_h]
            )

        return foreground_contours, hole_contours

    def detect_contours(
        self,
        target_spacing: float,
        tolerance: float,
        filter_params: FilterConfig,
        annotation: str | None = None,
    ):
        """
        Detect and filter contours from a binary mask based on specified parameters.

        This method identifies contours in a binary mask, filters them based on area
        thresholds, and scales the contours to a specified target resolution.

        Args:
            target_spacing (float): Desired spacing at which tiles should be extracted.
            tolerance (float): Tolerance for matching the target_spacing, deciding how much
                target_spacing can deviate from those specified in the slide metadata.
            filter_params (FilterConfig): Filtering parameters containing:
                - "a_t" (int): Minimum area threshold for foreground contours.
                - "a_h" (int): Minimum area threshold for holes within contours.
                - "ref_tile_size" (int): Reference tile size for computing areas.
            annotation (str, optional): Specific annotation to use for contour detection.

        Returns:
            tuple[list[np.ndarray], list[list[np.ndarray]]]:
                - A list of scaled foreground contours.
                - A list of lists containing scaled hole contours for each foreground contour.
        """

        spacing_level, _ = self.get_best_level_for_spacing(target_spacing, tolerance)
        current_scale = self.level_downsamples[spacing_level]
        target_scale = self.level_downsamples[self.seg_level]
        scale = tuple(a / b for a, b in zip(target_scale, current_scale))
        ref_tile_size = (filter_params.ref_tile_size, filter_params.ref_tile_size)
        ref_tile_size_at_target_scale = tuple(
            a / b for a, b in zip(ref_tile_size, scale)
        )
        scaled_ref_tile_area = int(
            ref_tile_size_at_target_scale[0] * ref_tile_size_at_target_scale[1]
        )

        adjusted_filter_params = replace(
            filter_params,
            a_t=filter_params.a_t * scaled_ref_tile_area,
            a_h=filter_params.a_h * scaled_ref_tile_area,
        )

        # find and filter contours
        mask = (
            self.annotation_mask["tissue"]
            if annotation is None
            else self.annotation_mask[annotation]
        )
        if mask.ndim == 3:  # If the mask has 3 channels
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if hierarchy is None or len(contours) == 0:
            return [], []

        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        foreground_contours, hole_contours = self.filter_contours(
            contours, hierarchy, adjusted_filter_params
        )

        # scale detected contours to level 0
        contours = self.scaleContourDim(foreground_contours, target_scale)
        holes = self.scaleHolesDim(hole_contours, target_scale)
        return contours, holes

    @staticmethod
    def scaleContourDim(contours, scale):
        """
        Scales the dimensions of a list of contours by a given factor.

        Args:
            contours (list of numpy.ndarray): A list of contours, where each contour is
                represented as a numpy array of coordinates.
            scale (float): The scaling factor to apply to the contours.

        Returns:
            list of numpy.ndarray: A list of scaled contours, where each contour's
            coordinates are multiplied by the scaling factor and converted to integers.
        """
        return [np.array(cont * scale, dtype="int32") for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        """
        Scales the dimensions of holes within a set of contours by a given factor.

        Args:
            contours (list of list of numpy.ndarray): A list of contours, where each contour
                is represented as a list of holes, and each hole is a numpy array of coordinates.
            scale (float): The scaling factor to apply to the dimensions of the holes.

        Returns:
            list of list of numpy.ndarray: A new list of contours with the dimensions of
            the holes scaled by the specified factor.
        """
        return [
            [np.array(hole * scale, dtype="int32") for hole in holes]
            for holes in contours
        ]

    def process_contours(
        self,
        contours,
        holes,
        tiling_params: TilingConfig,
        filter_params: FilterConfig,
        annotation: str | None = None,
        disable_tqdm: bool = False,
        num_workers: int = 1,
    ):
        """
        Processes a list of contours and their corresponding holes to generate tile coordinates,
        tissue percentages, and other metadata.

        Args:
            contours (list): List of contours representing tissue blobs in the wsi.
            holes (list): List of tissue holes in each contour.
            tiling_params: Tiling configuration for contour processing.
            filter_params (FilterConfig): Parameters for filtering.
            annotation (str, optional): annotation type to use for tile extraction.
            num_workers (int, optional): Number of workers to use for parallel processing. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - running_x_coords (list): The x-coordinates of the extracted tiles.
                - running_y_coords (list): The y-coordinates of the extracted tiles.
                - running_tissue_pct (list): List of tissue percentages for each extracted tile.
                - running_contour_indices (list): List of contour indices for each extracted tile.
                - tile_level (int): Level of the wsi used for tile extraction.
                - resize_factor (float): The factor by which the tile size was resized.
        """
        x_coord_chunks: list[np.ndarray] = []
        y_coord_chunks: list[np.ndarray] = []
        tissue_pct_chunks: list[np.ndarray] = []
        contour_index_chunks: list[np.ndarray] = []
        tile_level, _, resize_factor = self._resolve_tile_read_metadata(tiling_params)

        def process_single_contour(i):
            return self.process_contour(
                contours[i],
                holes[i],
                tiling_params,
                filter_params,
                annotation,
                num_workers=num_workers,
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

        for i, (
            x_coords,
            y_coords,
            tissue_pct,
            cont_tile_level,
            cont_resize_factor,
        ) in enumerate(results):
            assert (
                cont_tile_level == tile_level
            ), "Tile level should be the same for all contours"
            assert (
                cont_resize_factor == resize_factor
            ), "Resize factor should be the same for all contours"
            if x_coords.shape[0] > 0:
                x_coord_chunks.append(x_coords.astype(np.int64, copy=False))
                y_coord_chunks.append(y_coords.astype(np.int64, copy=False))
                tissue_pct_chunks.append(np.asarray(tissue_pct, dtype=np.float32))
                contour_index_chunks.append(
                    np.full(x_coords.shape[0], i, dtype=np.int32)
                )

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
            tile_level,
            resize_factor,
        )

    def _resolve_tile_read_metadata(self, tiling_params: TilingConfig):
        target_spacing = tiling_params.target_spacing_um
        tolerance = tiling_params.tolerance
        tile_level, is_within_tolerance = self.get_best_level_for_spacing(
            target_spacing, tolerance
        )
        tile_spacing = self.get_level_spacing(tile_level)
        resize_factor = target_spacing / tile_spacing
        if is_within_tolerance:
            resize_factor = 1.0

        assert (
            resize_factor >= 1
        ), f"Resize factor should be greater than or equal to 1. Got {resize_factor}"
        return tile_level, tile_spacing, resize_factor

    def process_contour(
        self,
        contour,
        contour_holes,
        tiling_params: TilingConfig,
        filter_params: FilterConfig,
        annotation: str | None = None,
        num_workers: int = 1,
    ):
        """
        Processes a contour to generate tile coordinates and associated metadata.

        Args:
            contour (numpy.ndarray): Contour to process, defined as a set of points.
            contour_holes (list): List of holes within the contour.
            tiling_params: Tiling configuration for contour processing.
            filter_params (FilterConfig): Parameters for filtering.
            annotation (str, optional): Annotation type to use for tile extraction.

        Returns:
            tuple: A tuple containing:
                - x_coords (list): List of x-coordinates for each tile.
                - y_coords (list): List of y-coordinates for each tile.
                - filtered_tissue_percentages (list): List of tissue percentages for each tile.
                - tile_level (int): Level of the image used for tile extraction.
                - resize_factor (float): The factor by which the tile size was resized.
        """
        target_tile_size = tiling_params.target_tile_size_px
        overlap = tiling_params.overlap

        tile_level, tile_spacing, resize_factor = self._resolve_tile_read_metadata(
            tiling_params
        )
        tile_size_resized = int(round(target_tile_size * resize_factor, 0))
        step_size = int(tile_size_resized * (1.0 - overlap))

        if contour is not None:
            start_x, start_y, w, h = cv2.boundingRect(contour)
        else:
            start_x, start_y, w, h = (
                0,
                0,
                self.level_dimensions[tile_level][0],
                self.level_dimensions[tile_level][1],
            )

        tile_downsample = (
            int(self.level_downsamples[tile_level][0]),
            int(self.level_downsamples[tile_level][1]),
        )
        tile_size_at_level_0 = (
            tile_size_resized * tile_downsample[0],
            tile_size_resized * tile_downsample[1],
        )
        img_w, img_h = self.level_dimensions[0]
        stop_y = int(start_y + h)
        stop_x = int(start_x + w)

        scale = self.level_downsamples[self.seg_level]
        cont = self.scaleContourDim([contour], (1.0 / scale[0], 1.0 / scale[1]))[0]

        mask = (
            self.annotation_mask["tissue"]
            if annotation is None
            else self.annotation_mask[annotation]
        )
        if annotation is None and not hasattr(self, "annotation_pct"):
            pct = tiling_params.tissue_threshold
        else:
            pct = (
                self.annotation_pct["tissue"]
                if annotation is None
                else self.annotation_pct[annotation]
            )
        tissue_checker = TissueFilter(
            contour=cont,
            contour_holes=contour_holes,
            tissue_mask=mask,
            geometry=ResolvedGeometry(
                target_tile_size_px=target_tile_size,
                read_spacing_um=tile_spacing,
                resize_factor=resize_factor,
                seg_spacing_um=self.get_level_spacing(self.seg_level),
                level0_spacing_um=self.get_level_spacing(0),
            ),
            pct=pct,
        )

        ref_step_size_x = int(round(step_size * tile_downsample[0], 0))
        ref_step_size_y = int(round(step_size * tile_downsample[1], 0))

        x_range = np.arange(start_x, stop_x, step=ref_step_size_x)
        y_range = np.arange(start_y, stop_y, step=ref_step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
        coord_candidates = np.array(
            [x_coords.flatten(), y_coords.flatten()]
        ).transpose()

        # filter coordinates based on tissue coverage (reads tissue mask for active contour only)
        keep_flags, tissue_pcts = tissue_checker.check_coordinates(coord_candidates)

        filtered_coordinates = coord_candidates[np.array(keep_flags) == 1]
        filtered_tissue_percentages = np.array(tissue_pcts)[np.array(keep_flags) == 1]

        ntile = len(filtered_coordinates)

        if ntile > 0:
            x_coords = filtered_coordinates[:, 0].astype(np.int64, copy=False)
            y_coords = filtered_coordinates[:, 1].astype(np.int64, copy=False)
            return (
                x_coords,
                y_coords,
                filtered_tissue_percentages.astype(np.float32, copy=False),
                tile_level,
                resize_factor,
            )

        else:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                tile_level,
                resize_factor,
            )
