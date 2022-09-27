import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from wsi.WholeSlideImage import WholeSlideImage
from wsi.wsi_utils import StitchCoords
from wsi.batch_process_utils import initialize_df


def stitching(file_path, wsi_object, downscale=64, bg_color=(255,255,255), draw_grid=False):
	start = time.time()
	heatmap = StitchCoords(
		file_path,
		wsi_object,
		downscale=downscale,
		bg_color=bg_color,
		alpha=-1,
		draw_grid=draw_grid,
	)
	total_time = time.time() - start
	return heatmap, total_time


def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
	start_time = time.time()
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed


def patching_old(WSI_object, **kwargs):
	start_time = time.time()
	file_path = WSI_object.createPatches_bag_hdf5(**kwargs, save_coord=True)
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed
 
def patching(WSI_object, **kwargs):
    start_time = time.time()
    file_path = WSI_object.process_contours(**kwargs)
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(
	data_dir: Path,
	save_dir: Path,
	patch_save_dir: Path,
	mask_save_dir: Path,
	stitch_save_dir: Path,
	seg_params,
	filter_params,
	vis_params,
	patch_params,
	slide_list: Optional[List[str]] = None,
	patch_size: int = 256,
	step_size: int = 256,
	patch_level: int = 0,
	seg: bool = False,
	stitch: bool = False, 
	patch: bool = False,
	auto_skip: bool = True,
	process_list: str = None,
	supported_fmts: List[str] = ['.tiff', '.tif', '.svs']
	):
	
	if slide_list is not None:
		with open(slide_list, 'r') as f:
			slides = sorted([s.strip() for s in f])
	else:
		slides = sorted([d for d in data_dir.iterdir()])
		slides = [slide.name for slide in slides if slide.is_file() and slide.suffix in supported_fmts]
	
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in range(total):
		df.to_csv(Path(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i+1/total, i+1, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id = Path(slide).stem

		if auto_skip and Path(patch_save_dir, slide_id + '.h5').is_file():
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		full_path = Path(data_dir, slide)
		WSI_object = WholeSlideImage(full_path)

		current_vis_params = {}
		current_filter_params = {}
		current_seg_params = {}
		current_patch_params = {}


		for key in vis_params.keys():
			if legacy_support and key == 'vis_level':
				df.loc[idx, key] = -1
			current_vis_params.update({key: df.loc[idx, key]})

		for key in filter_params.keys():
			if legacy_support and key == 'a_t':
				old_area = df.loc[idx, 'a']
				seg_level = df.loc[idx, 'seg_level']
				scale = WSI_object.level_downsamples[seg_level]
				adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
				current_filter_params.update({key: adjusted_area})
				df.loc[idx, key] = adjusted_area
			current_filter_params.update({key: df.loc[idx, key]})

		for key in seg_params.keys():
			if legacy_support and key == 'seg_level':
				df.loc[idx, key] = -1
			if key in df.columns:
				current_seg_params.update({key: df.loc[idx, key]})

		for key in patch_params.keys():
			if key in df.columns:
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = current_seg_params['keep_ids']
		if keep_ids is not None and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = current_seg_params['exclude_ids']
		if exclude_ids is not None and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if seg_params.save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = Path(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update(
				{
					'patch_level': patch_level,
					'patch_size': patch_size,
					'step_size': step_size,
					'save_path': str(patch_save_dir)
				},
			)
			file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params,)
			# file_path, patch_time_elapsed = patching_old(WSI_object=WSI_object,  **current_patch_params,)

		stitch_time_elapsed = -1
		if stitch:
			file_path = Path(patch_save_dir, slide_id+'.h5')
			if file_path.is_file():
				heatmap, stitch_time_elapsed = stitching(
					file_path,
					WSI_object,
					downscale=64,
					bg_color=tuple(patch_params.bg_color),
					draw_grid=patch_params.draw_grid,
				)
				stitch_path = Path(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(Path(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times
