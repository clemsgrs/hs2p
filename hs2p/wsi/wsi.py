from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from hs2p.configs import FilterConfig, SegmentationConfig, TilingConfig
from hs2p.wsi.geometry import (
    compute_level_spacings,
    select_level,
    select_level_for_downsample,
)
from hs2p.wsi.masks import load_annotation_masks
from hs2p.wsi.reader import open_slide, resolve_backend
from hs2p.wsi.segmentation import (
    detect_contours as detect_contours_impl,
    filter_contours as filter_contours_impl,
    scale_contour_dim,
    scale_holes_dim,
    segment_tissue as segment_tissue_impl,
)
from hs2p.wsi.tiling import (
    filter_black_and_white_tiles as filter_black_and_white_tiles_impl,
    is_in_holes,
    process_contour as process_contour_impl,
    process_contours as process_contours_impl,
)

Image.MAX_IMAGE_PIXELS = 933120000


@dataclass(frozen=True)
class ResolvedSamplingSpec:
    pixel_mapping: dict[str, int]
    color_mapping: dict[str, list[int] | None] | None
    tissue_percentage: dict[str, float | None]
    active_annotations: tuple[str, ...]


class WholeSlideImage(object):
    def __init__(
        self,
        path: Path,
        backend: str,
        mask_path: Path | None = None,
        spacing_at_level_0: float | None = None,
        segment: bool = False,
        segment_params: SegmentationConfig | None = None,
        sampling_spec: ResolvedSamplingSpec | None = None,
        pixel_mapping: dict | None = None,
    ):
        self.path = path
        self.name = path.stem.replace(" ", "_")
        self.fmt = path.suffix
        self.requested_backend = backend
        selection = resolve_backend(backend, wsi_path=path, mask_path=mask_path)
        self.backend = selection.backend
        self.wsi = open_slide(path, backend=self.backend)
        self._scaled_contours_cache = {}
        self._scaled_holes_cache = {}
        self._level_spacing_cache = {}

        self.spacing_at_level_0 = spacing_at_level_0
        self.raw_spacings = list(getattr(self.wsi, "spacings"))
        self.spacings = self.get_spacings()
        self.level_dimensions = list(self.wsi.level_dimensions)
        self.level_downsamples = list(self.wsi.level_downsamples)
        self.pixel_mapping = pixel_mapping

        self.mask_path = mask_path
        self.mask = None
        if mask_path is not None:
            if sampling_spec is None:
                raise ValueError("sampling_spec is required when loading a mask-backed slide")
            self.mask = open_slide(mask_path, backend=self.backend)
            self.seg_level = self.load_segmentation(
                segment_params=segment_params,
                sampling_spec=sampling_spec,
            )
        elif segment:
            self.seg_level = self.segment_tissue(segment_params)

        if sampling_spec is not None:
            self.annotation_pct = sampling_spec.tissue_percentage

    def get_slide(self, level: int) -> np.ndarray:
        return self.wsi.read_level(level)

    def get_tile(self, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
        return self.wsi.read_region(
            (int(x), int(y)),
            int(level),
            (int(width), int(height)),
            pad_missing=True,
        )

    def get_downsamples(self):
        return list(self.level_downsamples)

    def get_spacings(self):
        if self.spacing_at_level_0 is None:
            return self.raw_spacings
        return compute_level_spacings(
            level0_spacing_um=self.spacing_at_level_0,
            level_downsamples=self.level_downsamples,
        )

    def get_level_spacing(self, level: int):
        if level not in self._level_spacing_cache:
            self._level_spacing_cache[level] = self.spacings[level]
        return self._level_spacing_cache[level]

    def get_best_level_for_spacing(self, target_spacing: float, tolerance: float):
        selection = select_level(
            requested_spacing_um=target_spacing,
            level0_spacing_um=self.spacings[0],
            level_downsamples=self.level_downsamples,
            tolerance=tolerance,
        )
        return selection.level, selection.is_within_tolerance

    def get_best_level_for_downsample_custom(self, downsample: int):
        return select_level_for_downsample(
            requested_downsample=float(downsample),
            level_downsamples=self.level_downsamples,
        )

    def load_segmentation(
        self,
        segment_params: SegmentationConfig,
        sampling_spec: ResolvedSamplingSpec,
    ):
        seg_level = self.get_best_level_for_downsample_custom(segment_params.downsample)
        seg_spacing = self.get_level_spacing(seg_level)
        self.annotation_mask = load_annotation_masks(
            mask_reader=self.mask,
            mask_path=self.mask_path,
            segment_params=segment_params,
            sampling_spec=sampling_spec,
            seg_level=seg_level,
            seg_spacing=seg_spacing,
        )
        return seg_level

    def segment_tissue(
        self,
        segment_params: SegmentationConfig,
    ):
        seg_level = self.get_best_level_for_downsample_custom(segment_params.downsample)
        self.annotation_mask = segment_tissue_impl(
            reader=self.wsi,
            segment_params=segment_params,
            seg_level=seg_level,
        )
        return seg_level

    def get_tile_coordinates(
        self,
        tiling_params: TilingConfig,
        filter_params: FilterConfig,
        annotation: str | None = None,
        disable_tqdm: bool = False,
        num_workers: int = 1,
    ):
        scale = tiling_params.target_spacing_um / self.get_level_spacing(0)
        tile_size_lv0 = int(round(tiling_params.target_tile_size_px * scale, 0))
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
    ):
        return filter_black_and_white_tiles_impl(
            reader=self.wsi,
            level_dimensions=self.level_dimensions,
            level_downsamples=self.level_downsamples,
            keep_flags=keep_flags,
            coord_candidates=coord_candidates,
            tile_size=tile_size,
            tile_level=tile_level,
            filter_params=filter_params,
        )

    @staticmethod
    def filter_contours(contours, hierarchy, filter_params: FilterConfig):
        return filter_contours_impl(contours, hierarchy, filter_params)

    def detect_contours(
        self,
        target_spacing: float,
        tolerance: float,
        filter_params: FilterConfig,
        annotation: str | None = None,
    ):
        return detect_contours_impl(
            annotation_mask=self.annotation_mask,
            annotation=annotation,
            target_spacing=target_spacing,
            tolerance=tolerance,
            filter_params=filter_params,
            spacings=self.spacings,
            level_downsamples=self.level_downsamples,
            seg_level=self.seg_level,
        )

    @staticmethod
    def isInHoles(holes, pt, tile_size):
        return is_in_holes(holes, pt, tile_size)

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, drop_holes=True, tile_size=256):
        keep_flag, tissue_pct = cont_check_fn(pt)
        if keep_flag:
            if holes is not None and drop_holes:
                return not WholeSlideImage.isInHoles(holes, pt, tile_size), tissue_pct
            return 1, tissue_pct
        return 0, tissue_pct

    @staticmethod
    def scaleContourDim(contours, scale):
        return scale_contour_dim(contours, scale)

    @staticmethod
    def scaleHolesDim(contours, scale):
        return scale_holes_dim(contours, scale)

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
        return process_contours_impl(
            reader=self.wsi,
            annotation_mask=self.annotation_mask,
            annotation_pct=getattr(self, "annotation_pct", None),
            level_spacings=self.spacings,
            level_dimensions=self.level_dimensions,
            level_downsamples=self.level_downsamples,
            seg_level=self.seg_level,
            contours=contours,
            holes=holes,
            tiling_params=tiling_params,
            filter_params=filter_params,
            annotation=annotation,
            disable_tqdm=disable_tqdm,
            num_workers=num_workers,
        )

    def _resolve_tile_read_metadata(self, tiling_params: TilingConfig):
        selection = select_level(
            requested_spacing_um=tiling_params.target_spacing_um,
            level0_spacing_um=self.spacings[0],
            level_downsamples=self.level_downsamples,
            tolerance=tiling_params.tolerance,
        )
        tile_spacing = self.get_level_spacing(selection.level)
        resize_factor = tiling_params.target_spacing_um / tile_spacing
        if selection.is_within_tolerance:
            resize_factor = 1.0
        assert resize_factor >= 1, f"Resize factor should be >= 1. Got {resize_factor}"
        return selection.level, tile_spacing, resize_factor

    def process_contour(
        self,
        contour,
        contour_holes,
        tiling_params: TilingConfig,
        filter_params: FilterConfig,
        annotation: str | None = None,
    ):
        return process_contour_impl(
            reader=self.wsi,
            annotation_mask=self.annotation_mask,
            annotation_pct=getattr(self, "annotation_pct", None),
            level_spacings=self.spacings,
            level_dimensions=self.level_dimensions,
            level_downsamples=self.level_downsamples,
            seg_level=self.seg_level,
            contour=contour,
            contour_holes=contour_holes,
            tiling_params=tiling_params,
            filter_params=filter_params,
            annotation=annotation,
        )
