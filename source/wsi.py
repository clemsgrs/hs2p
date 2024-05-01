import os
import cv2
import time
import math
import numpy as np
import pandas as pd
import wholeslidedata as wsd
import multiprocessing as mp

from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from source.utils import save_hdf5, save_patch, compute_time, find_common_spacings
from source.util_classes import (
    Contour_Checking_fn,
    isInContourV1,
    isInContourV2,
    isInContourV3_Easy,
    isInContourV3_Hard,
    isInContour_pct,
)

import warnings
# ignore all warnings from wholeslidedata
warnings.filterwarnings("ignore", module="wholeslidedata")

Image.MAX_IMAGE_PIXELS = 933120000


class WholeSlideImage(object):
    def __init__(
        self, path: Path, spacing: Optional[float] = None, backend: str = "asap"
    ):

        """
        Args:
            path (Path): fullpath to WSI file
        """

        self.path = path
        self.name = path.stem
        self.fmt = path.suffix
        self.wsi = wsd.WholeSlideImage(path, backend=backend)
        self.level_dimensions = self.wsi.shapes
        self.level_downsamples = self.get_downsamples()
        self.spacing = spacing
        self.spacings = self.get_spacings()
        self.backend = backend

        self.contours_tissue = None
        self.contours_tumor = None

    def get_downsamples(self):
        level_downsamples = []
        dim_0 = self.level_dimensions[0]
        for dim in self.level_dimensions:
            level_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(level_downsample)
        return level_downsamples

    def get_spacings(self):
        if self.spacing is None:
            spacings = self.wsi.spacings
        else:
            spacings = [
                self.spacing * s / self.wsi.spacings[0] for s in self.wsi.spacings
            ]
        return spacings

    def get_level_spacing(self, level: int = 0):
        return self.spacings[level]

    def get_best_level_for_spacing(
        self, target_spacing: float, ignore_warning: bool = False
    ):
        spacing = self.get_level_spacing(0)
        target_downsample = target_spacing / spacing
        level, tol, above_tol = self.get_best_level_for_downsample_custom(
            target_downsample, return_tol_status=True
        )
        level_spacing = self.get_level_spacing(level)
        resize_factor = int(target_spacing / level_spacing)
        if above_tol and not ignore_warning:
            print(
                f"WARNING! The natural spacing ({round(self.spacings[level],4)}) closest to the target spacing ({round(target_spacing,4)}) was more than {tol*100:.1f}% appart ({self.name})."
            )
        return level, resize_factor

    def get_best_level_for_downsample_custom(
        self, downsample, tol: float = 0.1, return_tol_status: bool = False
    ):
        level = int(np.argmin([abs(x - downsample) for x, _ in self.level_downsamples]))
        delta = abs(self.level_downsamples[level][0] / downsample - 1)
        above_tol = delta > tol
        if return_tol_status:
            return level, tol, above_tol
        else:
            return level

    def initSegmentation(self, mask_fp: Path):
        # load segmentation results from pickle file
        import pickle

        with open(mask_fp, "rb") as f:
            asset_dict = pickle.load(f)
            self.holes_tissue = asset_dict["holes"]
            self.contours_tissue = asset_dict["tissue"]

    def loadSegmentation(
        self,
        mask_fp: Path,
        spacing: float,
        downsample: int,
        filter_params,
        sthresh_up: int = 255,
        tissue_val: int = 1,
        try_previous_level: bool = True,
    ):

        mask = WholeSlideImage(mask_fp, backend=self.backend)

        # ensure mask and slide have at least one common spacing
        common_spacings = find_common_spacings(self.spacings, mask.spacings, tolerance=0.1)
        assert (
            len(common_spacings) >= 1
        ), f"The provided segmentation mask (spacings={mask.spacings}) has no common spacing with the slide (spacings={self.spacings}). A minimum of 1 common spacing is required."

        seg_level = self.get_best_level_for_downsample_custom(downsample)
        seg_spacing = self.get_level_spacing(seg_level)

        # check if this spacing is present in common spacings
        is_in_common_spacings = seg_spacing in [s for s,_ in common_spacings]
        if not is_in_common_spacings:
            # find spacing that is common to slide and mask and that is the closest to seg_spacing
            closest = np.argmin([abs(seg_spacing - s) for s,_ in common_spacings])
            closest_common_spacing = common_spacings[closest][0]
            seg_spacing = closest_common_spacing
            seg_level, _ = self.get_best_level_for_spacing(seg_spacing, ignore_warning=True)

        m = mask.wsi.get_slide(spacing=seg_spacing)
        m = m[..., 0]

        m = (m == tissue_val).astype('uint8')
        if np.max(m) <= 1:
            m = m * sthresh_up

        self.binary_mask = m
        self.detect_contours(m, spacing, seg_level, filter_params)
        return seg_level

    def segmentTissue(
        self,
        spacing: float,
        seg_level: int = 0,
        sthresh: int = 20,
        sthresh_up: int = 255,
        mthresh: int = 7,
        close: int = 0,
        use_otsu: bool = False,
        filter_params: Dict[str, int] = {"ref_patch_size": 512, "a_t": 1, "a_h": 1},
    ):
        """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """

        seg_spacing = self.get_level_spacing(seg_level)
        img = self.wsi.get_slide(spacing=seg_spacing)
        img = np.array(Image.fromarray(img).convert("RGBA"))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring

        # Thresholding
        if use_otsu:
            _, img_thresh = cv2.threshold(
                img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY
            )
        else:
            _, img_thresh = cv2.threshold(
                img_med, sthresh, sthresh_up, cv2.THRESH_BINARY
            )

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        self.binary_mask = img_thresh
        self.detect_contours(img_thresh, spacing, seg_level, filter_params)

    def detect_contours(
        self, img_thresh, spacing: float, seg_level: int, filter_params: Dict[str, int]
    ):
        def _filter_contours(contours, hierarchy, filter_params):
            """
            Filter contours by: area.
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
                if a > filter_params["a_t"]:
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]

            hole_contours = []
            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids]
                unfilered_holes = sorted(
                    unfiltered_holes, key=cv2.contourArea, reverse=True
                )
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[: filter_params["max_n_holes"]]
                filtered_holes = []

                # filter these holes
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params["a_h"]:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        spacing_level, resize_factor = self.get_best_level_for_spacing(spacing, ignore_warning=True)
        current_scale = self.level_downsamples[spacing_level]
        target_scale = self.level_downsamples[seg_level]
        scale = tuple(a / b for a, b in zip(target_scale, current_scale))
        ref_patch_size = filter_params["ref_patch_size"]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))

        filter_params = filter_params.copy()
        filter_params["a_t"] = filter_params["a_t"] * scaled_ref_patch_area
        filter_params["a_h"] = filter_params["a_h"] * scaled_ref_patch_area

        # Find and filter contours
        contours, hierarchy = cv2.findContours(
            img_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )  # Find contours
        # contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Find contours
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params:
            # Necessary for filtering out artifacts
            foreground_contours, hole_contours = _filter_contours(
                contours, hierarchy, filter_params
            )

        # scale detected contours to level 0
        self.contours_tissue = self.scaleContourDim(foreground_contours, target_scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, target_scale)

    def visWSI(
        self,
        vis_level: int = 0,
        color: Tuple[int] = (0, 255, 0),
        hole_color: Tuple[int] = (0, 0, 255),
        annot_color: Tuple[int] = (255, 0, 0),
        line_thickness: int = 250,
        max_size: Optional[int] = None,
        top_left: Optional[Tuple[int]] = None,
        bot_right: Optional[Tuple[int]] = None,
        custom_downsample: int = 1,
        view_slide_only: bool = False,
        number_contours: bool = False,
        seg_display: bool = True,
        annot_display: bool = True,
    ):

        downsample = self.level_downsamples[vis_level]
        scale = [1 / downsample[0], 1 / downsample[1]]

        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple(
                (np.array(bot_right) * scale).astype(int)
                - (np.array(top_left) * scale).astype(int)
            )
            region_size = (w, h)
        else:
            top_left = (0, 0)
            region_size = self.level_dimensions[vis_level]

        x, y = top_left
        s = self.spacings[vis_level]
        width, height = region_size[0], region_size[1]
        img = self.wsi.get_patch(x, y, width, height, spacing=s, center=False)
        if self.backend == "openslide":
            img = np.ascontiguousarray(img)

        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(
                        img,
                        self.scaleContourDim(self.contours_tissue, scale),
                        -1,
                        color,
                        line_thickness,
                        lineType=cv2.LINE_8,
                        offset=offset,
                    )

                else:
                    # add numbering to each contour
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center
                        cv2.drawContours(
                            img,
                            [contour],
                            -1,
                            color,
                            line_thickness,
                            lineType=cv2.LINE_8,
                            offset=offset,
                        )
                        cv2.putText(
                            img,
                            "{}".format(idx),
                            (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255, 0, 0),
                            10,
                        )

                for holes in self.holes_tissue:
                    cv2.drawContours(
                        img,
                        self.scaleContourDim(holes, scale),
                        -1,
                        hole_color,
                        line_thickness,
                        lineType=cv2.LINE_8,
                    )

            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(
                    img,
                    self.scaleContourDim(self.contours_tumor, scale),
                    -1,
                    annot_color,
                    line_thickness,
                    lineType=cv2.LINE_8,
                    offset=offset,
                )

        img = Image.fromarray(img)

        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if (
                cv2.pointPolygonTest(
                    hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False
                )
                > 0
            ):
                return 1

        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, drop_holes=True, patch_size=256):
        keep_flag, tissue_pct = cont_check_fn(pt)
        if keep_flag:
            if holes is not None and drop_holes:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size), tissue_pct
            else:
                return 1, tissue_pct
        return 0, tissue_pct

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype="int32") for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [
            [np.array(hole * scale, dtype="int32") for hole in holes]
            for holes in contours
        ]

    def process_contours(
        self,
        save_dir: Optional[Path] = None,
        seg_level: int = -1,
        spacing: float = 0.5,
        patch_size: int = 256,
        overlap: float = 0.0,
        contour_fn: str = "pct",
        drop_holes: bool = True,
        tissue_thresh: float = 0.01,
        use_padding: bool = True,
        save_patches_to_disk: bool = False,
        save_patches_in_common_dir: bool = False,
        patch_format: str = "png",
        top_left: Optional[List[int]] = None,
        bot_right: Optional[List[int]] = None,
        num_workers: int = 1,
        verbose: bool = False,
    ):
        save_flag = save_dir is not None
        if save_flag:
            if save_patches_in_common_dir:
                save_path_hdf5 = Path(save_dir.parent, "h5", f"{self.name}.h5")
            else:
                save_path_hdf5 = Path(save_dir, f"{self.name}.h5")
        else:
            save_path_hdf5 = None
        start_time = time.time()
        n_contours = len(self.contours_tissue)
        if verbose:
            print(f"Total number of contours to process: {n_contours}")

        dfs = []
        init = True

        for i, cont in enumerate(self.contours_tissue):

            asset_dict, attr_dict, tile_df = self.process_contour(
                cont,
                self.holes_tissue[i],
                seg_level,
                spacing,
                save_dir,
                patch_size,
                overlap,
                contour_fn,
                drop_holes,
                tissue_thresh,
                use_padding,
                top_left,
                bot_right,
                num_workers=num_workers,
                verbose=verbose,
            )

            if tile_df is not None:
                tile_df["contour"] = [i] * len(tile_df)
                if save_patches_to_disk:
                    if save_patches_in_common_dir:
                        tile_df["tile_path"] = tile_df.apply(
                            lambda row: Path(
                                save_dir.parent, "imgs", f"{row.slide_id}_{row.x}_{row.y}.{patch_format}"
                            ).resolve(),
                            axis=1,
                        )
                    else:
                        tile_df["tile_path"] = tile_df.apply(
                            lambda row: Path(
                                save_dir, "imgs", f"{row.x}_{row.y}.{patch_format}"
                            ).resolve(),
                            axis=1,
                        )
                    cols = list(tile_df.columns)
                    cols = [cols[-1]] + cols[:-1]
                    tile_df = tile_df[cols]
                dfs.append(tile_df)

            if len(asset_dict) > 0 and save_flag:
                if init:
                    save_path_hdf5.parent.mkdir(parents=True, exist_ok=True)
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode="w")
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode="a")
                if save_patches_to_disk:
                    if save_patches_in_common_dir:
                        patch_save_dir = Path(save_dir.parent, "imgs")
                    else:
                        patch_save_dir = Path(save_dir, "imgs")
                    patch_save_dir.mkdir(parents=True, exist_ok=True)
                    npatch, mins, secs = save_patch(
                        self.wsi,
                        patch_save_dir,
                        asset_dict,
                        attr_dict,
                        fmt=patch_format,
                        save_patches_in_common_dir=save_patches_in_common_dir,
                    )

        end_time = time.time()
        patch_saving_mins, patch_saving_secs = compute_time(start_time, end_time)
        if len(dfs) > 0:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = None
        return save_path_hdf5, df

    def process_contour(
        self,
        cont,
        contour_holes,
        seg_level: int,
        spacing: float,
        save_dir: Path,
        patch_size: int = 256,
        overlap: float = 0.0,
        contour_fn: str = "pct",
        drop_holes: bool = True,
        tissue_thresh: float = 0.01,
        use_padding: bool = True,
        top_left: Optional[List[int]] = None,
        bot_right: Optional[List[int]] = None,
        num_workers: int = 1,
        verbose: bool = False,
    ):

        patch_level, resize_factor = self.get_best_level_for_spacing(spacing, ignore_warning=True)

        patch_spacing = self.get_level_spacing(patch_level)
        resize_factor = int(spacing / patch_spacing)
        patch_size_resized = patch_size * resize_factor
        step_size = int(patch_size_resized * (1.0 - overlap))

        if cont is not None:
            start_x, start_y, w, h = cv2.boundingRect(cont)
        else:
            start_x, start_y, w, h = (
                0,
                0,
                self.level_dimensions[patch_level][0],
                self.level_dimensions[patch_level][1],
            )

        # 256x256 patches at 1mpp are equivalent to 512x512 patches at 0.5mpp
        # ref_patch_size capture the patch size at level 0
        # assumes self.level_downsamples[0] is always (1, 1)
        patch_downsample = (
            int(self.level_downsamples[patch_level][0]),
            int(self.level_downsamples[patch_level][1]),
        )
        ref_patch_size = (
            patch_size * patch_downsample[0],
            patch_size * patch_downsample[1],
        )

        img_w, img_h = self.level_dimensions[0]
        if use_padding:
            stop_y = int(start_y + h)
            stop_x = int(start_x + w)
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)

        if verbose:
            print(f"Bounding Box: {start_x}, {start_y}, {w}, {h}")
            print(f"Contour Area: {cv2.contourArea(cont)}")

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print(f"Adjusted Bounding Box: {start_x}, {start_y}, {w}, {h}")

        # TODO: work with the ref_patch_size tuple instead of using ref_patch_size[0] to account for difference along x & y axes
        if isinstance(contour_fn, str):
            if contour_fn == "four_pt":
                cont_check_fn = isInContourV3_Easy(
                    contour=cont, patch_size=ref_patch_size[0], center_shift=0.5
                )
            elif contour_fn == "four_pt_hard":
                cont_check_fn = isInContourV3_Hard(
                    contour=cont, patch_size=ref_patch_size[0], center_shift=0.5
                )
            elif contour_fn == "center":
                cont_check_fn = isInContourV2(
                    contour=cont, patch_size=ref_patch_size[0]
                )
            elif contour_fn == "basic":
                cont_check_fn = isInContourV1(contour=cont)
            elif contour_fn == "pct":
                scale = self.level_downsamples[seg_level]
                cont = self.scaleContourDim([cont], (1.0 / scale[0], 1.0 / scale[1]))[0]
                cont_check_fn = isInContour_pct(
                    contour=cont,
                    contour_holes=contour_holes,
                    tissue_mask=self.binary_mask,
                    patch_size=ref_patch_size[0],
                    scale=scale,
                    pct=tissue_thresh,
                )
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        # input step_size is defined w.r.t to input spacing
        # given contours are defined w.r.t level 0, step_size (potentially) needs to be upsampled
        ref_step_size_x = int(step_size * patch_downsample[0])
        ref_step_size_y = int(step_size * patch_downsample[1])

        # x & y values are defined w.r.t level 0
        x_range = np.arange(start_x, stop_x, step=ref_step_size_x)
        y_range = np.arange(start_y, stop_y, step=ref_step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
        coord_candidates = np.array(
            [x_coords.flatten(), y_coords.flatten()]
        ).transpose()

        if num_workers > 1:
            num_workers = min(mp.cpu_count(), num_workers)
            if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
                num_workers = min(num_workers, int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

            pool = mp.Pool(num_workers)

            iterable = [
                (coord, contour_holes, ref_patch_size[0], cont_check_fn, drop_holes)
                for coord in coord_candidates
            ]
            results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
            pool.close()
            filtered_results = np.array(
                [result[0] for result in results if result[0] is not None]
            )
            filtered_tissue_pcts = [
                result[1] for result in results if result[0] is not None
            ]
        else:
            results = []
            tissue_pcts = []
            for coord in coord_candidates:
                c, pct = self.process_coord_candidate(
                    coord, contour_holes, ref_patch_size[0], cont_check_fn, drop_holes
                )
                results.append(c)
                tissue_pcts.append(pct)
            filtered_results = np.array(
                [result for result in results if result is not None]
            )
            filtered_tissue_pcts = [
                tissue_pct
                for i, tissue_pct in enumerate(tissue_pcts)
                if results[i] is not None
            ]

        npatch = len(filtered_results)

        if verbose:
            print(f"Extracted {npatch} patches")

        if npatch > 0:
            asset_dict = {
                "coords": filtered_results,
            }
            attr = {
                "patch_size": patch_size,
                "spacing": spacing,
                "patch_size_resized": patch_size_resized,
                "patch_level": patch_level,
                "patch_spacing": patch_spacing,
                "ref_patch_size": ref_patch_size[0],
                "downsample": self.level_downsamples[patch_level],
                "downsampled_level_dim": tuple(
                    np.array(self.level_dimensions[patch_level])
                ),
                "level_dimension": self.level_dimensions[patch_level],
                "wsi_name": self.name,
                "save_path": str(save_dir),
            }
            attr_dict = {"coords": attr}
            df_data = {
                "slide_id": [self.name] * npatch,
                "spacing": [spacing] * npatch,
                "patch_level": [patch_level] * npatch,
                "patch_spacing": [patch_spacing] * npatch,
                "level_dimension": [self.level_dimensions[patch_level]] * npatch,
                "tile_size": [patch_size] * npatch,
                "x": list(filtered_results[:, 0]),  # defined w.r.t level 0
                "y": list(filtered_results[:, 1]),  # defined w.r.t level 0
                "tissue_pct": filtered_tissue_pcts,
            }
            tile_df = pd.DataFrame.from_dict(df_data)
            return asset_dict, attr_dict, tile_df

        else:
            return {}, {}, None

    @staticmethod
    def process_coord_candidate(
        coord, contour_holes, patch_size, cont_check_fn, drop_holes
    ):
        keep_flag, tissue_pct = WholeSlideImage.isInContours(
            cont_check_fn, coord, contour_holes, drop_holes, patch_size
        )
        if keep_flag:
            return coord, tissue_pct
        else:
            return None, tissue_pct
