import cv2
import time
import tqdm
import h5py
import math
import openslide
import numpy as np
import multiprocessing as mp

from PIL import Image
from pathlib import Path
from typing import List, Optional

from wsi.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, save_hdf5, save_patch, isBlackPatch, isWhitePatch
from wsi.util_classes import Contour_Checking_fn, isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, isInContour_pct

Image.MAX_IMAGE_PIXELS = 933120000


class WholeSlideImage(object):

    def __init__(self, path: Path):

        """
        Args:
            path (Path): fullpath to WSI file
        """

        self.path = path
        self.name = path.stem
        self.wsi = openslide.open_slide(str(path))
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.wsi.level_dimensions

        self.contours_tissue = None
        self.contours_tumor = None
        self.hdf5_file = None

    def getOpenSlide(self):
        return self.wsi

    def initSegmentation(self, mask_file: Path):
        # load segmentation results from pickle file
        import pickle
        with open(mask_file,'rb') as f:
            asset_dict = pickle.load(f)
            self.holes_tissue = asset_dict['holes']
            self.contours_tissue = asset_dict['tissue']

    def saveSegmentation(self, mask_file):
        # save segmentation results using pickle
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        import pickle
        with open(mask_file,'wb') as f:
	        pickle.dump(asset_dict, f)

    def segmentTissue(
        self,
        seg_level=0,
        sthresh=20,
        sthresh_up=255,
        mthresh=7,
        close=0,
        use_otsu=False,
        filter_params={'a_t':100},
        ref_patch_size=512,
        ):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """

        def _filter_contours(contours, hierarchy, filter_params):
            """
                Filter contours by: area.
            """
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
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
                if a == 0: continue
                if tuple((filter_params['a_t'],)) < tuple((a,)):
                    filtered.append(cont_idx)
                    all_holes.append(holes)


            foreground_contours = [contours[cont_idx] for cont_idx in filtered]

            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []

                # filter these holes
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring


        # Thresholding
        if use_otsu:
            _, img_thresh = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_thresh = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        self.binary_mask = img_thresh

        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours
        # contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Find contours
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params:
            # Necessary for filtering out artifacts
            foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)

        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

    def visWSI(self, vis_level=0, color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0),
                    line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False,
                    number_contours=False, seg_display=True, annot_display=True):

        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]]

        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0,0)
            region_size = self.level_dim[vis_level]

        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))

        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale),
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

                else: # add numbering to each contour
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center
                        cv2.drawContours(img,  [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

                for holes in self.holes_tissue:
                    cv2.drawContours(img, self.scaleContourDim(holes, scale),
                                     -1, hole_color, line_thickness, lineType=cv2.LINE_8)

            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale),
                                 -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset)

        img = Image.fromarray(img)

        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))

        return img

    def createPatches_bag_hdf5(self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, **kwargs):
        contours = self.contours_tissue
        contour_holes = self.holes_tissue

        # print(f'Start patching {self.name}...')
        elapsed = time.time()
        for idx, cont in enumerate(contours):
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)

            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)

                # empty contour, continue
                except StopIteration:
                    continue

                file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                self.hdf5_file = file_path

            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch)

        return self.hdf5_file

    def _getPatchGenerator(self, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256, custom_downsample=1,
        white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if custom_downsample > 1:
            assert custom_downsample == 2
            target_patch_size = patch_size
            patch_size = target_patch_size * 2
            step_size = step_size * 2
            print("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(custom_downsample, patch_size, patch_size,
                target_patch_size, target_patch_size))

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1])
            stop_x = min(start_x+w, img_w-ref_patch_size[0])

        count = 0
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):

                if not self.isInContours(cont_check_fn, (x,y), self.holes_tissue[cont_idx], ref_patch_size[0]): #point not inside contour and its associated holes
                    continue

                count+=1
                patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))

                if white_black:
                    if isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or isWhitePatch(np.array(patch_PIL), satThresh=white_thresh):
                        continue

                patch_info = {
                    'x': x // (patch_downsample[0] * custom_downsample),
                    'y': y // (patch_downsample[1] * custom_downsample),
                    'cont_idx': cont_idx,
                    'patch_size': patch_size,
                    'patch_level': patch_level,
                    'downsample': self.level_downsamples[patch_level],
                    'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])//custom_downsample),
                    'level_dim': self.level_dim[patch_level],
                    'patch_PIL': patch_PIL,
                    'wsi_name': self.name,
                    'save_path': save_path,
                }

                yield patch_info


        print("patches extracted: {}".format(count))

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1

        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, drop_holes=True, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None and drop_holes:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]

        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))

        return level_downsamples

    def process_contours(
        self,
        save_dir,
        seg_level: int = -1,
        patch_level: int = 0,
        patch_size: int = 256,
        step_size: int = 256,
        contour_fn: str = 'pct',
        drop_holes: bool = True,
        tissue_thresh: float = 0.01,
        use_padding: bool = True,
        save_patches_to_disk: bool = False,
        patch_format: str = 'png',
        top_left: Optional[List[int]] = None,
        bot_right: Optional[List[int]] = None,
        position: int = -1,
        verbose: bool = False,
    ):
        save_path_hdf5 = Path(save_dir, f'{self.name}.h5')
        # self.hdf5_file = save_path_hdf5
        elapsed = time.time()
        n_contours = len(self.contours_tissue)
        if verbose:
            print(f'Total number of contours to process: {n_contours}')
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True

        with tqdm.tqdm(
            self.contours_tissue,
            desc=(f'Patching'),
            unit=' contour',
            ncols=80,
            position=position,
            leave=True,
        ) as t:

            for i, cont in enumerate(t):

                asset_dict, attr_dict = self.process_contour(
                    cont,
                    self.holes_tissue[i],
                    seg_level,
                    patch_level,
                    save_dir,
                    patch_size,
                    step_size,
                    contour_fn,
                    drop_holes,
                    tissue_thresh,
                    use_padding,
                    verbose=verbose,
                )
                if len(asset_dict) > 0:
                    if init:
                        save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                        init = False
                    else:
                        save_hdf5(save_path_hdf5, asset_dict, mode='a')
                    if save_patches_to_disk:
                        png_save_dir = Path(save_dir, 'imgs')
                        png_save_dir.mkdir(parents=True, exist_ok=True)
                        print('Saving extracted patches to disk...')
                        save_patch(self.wsi, png_save_dir, asset_dict, attr_dict, fmt=patch_format)

        return self.hdf5_file

    def process_contour(
        self,
        cont,
        contour_holes,
        seg_level: int,
        patch_level: int,
        save_dir: Path,
        patch_size: int = 256,
        step_size: int = 256,
        contour_fn: str = 'pct',
        drop_holes: bool = True,
        tissue_thresh: float = 0.01,
        use_padding: bool = True,
        top_left: Optional[List[int]] = None,
        bot_right: Optional[List[int]] = None,
        verbose: bool = False,
    ):

        if cont is not None:
            start_x, start_y, w, h = cv2.boundingRect(cont)
        else:
            start_x, start_y, w, h = (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-patch_size+1)
            stop_x = min(start_x+w, img_w-patch_size+1)

        if verbose:
            print(f'Bounding Box: {start_x}, {start_y}, {w}, {h}')
            print(f'Contour Area: {cv2.contourArea(cont)}')

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print('Contour is not in specified ROI, skip')
                return {}, {}
            else:
                print(f'Adjusted Bounding Box: {start_x}, {start_y}, {w}, {h}')

        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=patch_size, center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=patch_size, center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=patch_size)
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            elif contour_fn == 'pct':
                scale = self.level_downsamples[seg_level]
                cont = self.scaleContourDim([cont],  (1./scale[0], 1./scale[1]))[0]
                cont_check_fn = isInContour_pct(contour=cont, tissue_mask=self.binary_mask, patch_size=patch_size, scale=scale, pct=tissue_thresh)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        x_range = np.arange(start_x, stop_x, step=step_size)
        y_range = np.arange(start_y, stop_y, step=step_size)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        pool = mp.Pool(num_workers)

        iterable = [(coord, contour_holes, patch_size, cont_check_fn, drop_holes) for coord in coord_candidates]
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        results = np.array([result for result in results if result is not None])

        if verbose:
            print(f'Extracted {len(results)} patches')

        if len(results)>1:
            asset_dict = {
                'coords': results,
            }

            attr = {
                'patch_size': patch_size,
                'patch_level': patch_level,
                'downsample': self.level_downsamples[patch_level],
                'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])),
                'level_dim': self.level_dim[patch_level],
                'wsi_name': self.name,
                'save_path': str(save_dir),
            }

            attr_dict = {'coords': attr}
            return asset_dict, attr_dict

        else:
            return {}, {}

    @staticmethod
    def process_coord_candidate(coord, contour_holes, patch_size, cont_check_fn, drop_holes):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, drop_holes, patch_size):
            return coord
        else:
            return None
