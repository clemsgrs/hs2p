import os
import PIL
import tqdm
import time
import hydra
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import multiprocessing as mp

from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
from omegaconf import DictConfig

from utils import initialize_wandb, segment
from source.wsi import WholeSlideImage


def get_mask_percent(mask, val=0):
    '''
    Determine the percentage of a mask that is equal to gleason score gs
    input:
        mask: mask as a PIL Image
        val: given value we want to check for
    output:
        percentage of the numpy array that is equal to val
    '''
    mask_arr = np.array(mask)
    w, h, _ = mask_arr.shape
    binary_mask = mask_arr == val
    mask_percentage = np.sum(binary_mask) / (w*h)
    return mask_percentage


def scale_coordinates(x, y, scale):
    xx = int(x * scale)
    yy = int(y * scale)
    return xx, yy


def overlay_mask_on_slide(
    wsi_object,
    mask_object,
    vis_level: int,
    pixel_mapping: Dict[str,int],
    color_mapping: Optional[Dict[str,List[int]]] = None,
    alpha: float = 0.5,
):
    """
    Show a mask overlayed on a slide
    """

    slide = wsi_object.wsi.read_region((0,0), vis_level, wsi_object.wsi.level_dimensions[vis_level])
    mask = mask_object.wsi.read_region((0,0), vis_level, mask_object.wsi.level_dimensions[vis_level])

    # Mask data is present in the R channel
    mask = mask.split()[0]

    # Create alpha mask
    mask_arr = np.array(mask)
    alpha_int = int(round(255*alpha))
    if color_mapping is not None:
        alpha_content = np.zeros_like(mask_arr)
        for k, v in pixel_mapping.items():
            if color_mapping[k] is not None:
                alpha_content += (mask_arr == v)
        alpha_content = np.less(alpha_content, 1).astype('uint8') * alpha_int + (255 - alpha_int)
    else:
        alpha_content = np.less_equal(mask_arr, 0).astype('uint8') * alpha_int + (255 - alpha_int)
    alpha_content = Image.fromarray(alpha_content)

    preview_palette = np.zeros(shape=768, dtype=int)

    if color_mapping is None:
        ncat = len(pixel_mapping)
        if ncat <= 10:
            color_palette = sns.color_palette("tab10")[:ncat]
        elif ncat <= 20:
            color_palette = sns.color_palette("tab20")[:ncat]
        else:
            raise ValueError(f"Implementation supports up to 20 categories (provided pixel_mapping has {ncat})")
        color_mapping = {k: color_palette[i]*255 for i, k in enumerate(pixel_mapping.keys())}

    p = [0]*3*len(color_mapping)
    for k, v in pixel_mapping.items():
        if color_mapping[k] is not None:
            p[v*3:v*3+3] = color_mapping[k]
    n = len(p)
    preview_palette[0:n] = np.array(p).astype(int)

    mask.putpalette(data=preview_palette.tolist())
    mask_rgb = mask.convert(mode='RGB')

    overlayed_image = PIL.Image.composite(image1=slide, image2=mask_rgb, mask=alpha_content)
    return overlayed_image


def extract_top_tiles(
    wsi_object,
    mask_object,
    tile_df,
    spacing: float,
    tile_size: int,
    downsample: int,
    val: int,
    threshold: float = 0.,
    sort: bool = False,
    topk: Optional[int] = None,
):

    spacing_level = wsi_object.get_best_level_for_spacing(spacing)
    if downsample == -1:
        downsample_level = max(spacing_level, 0)
    else:
        downsample_level = wsi_object.get_best_level_for_downsample_custom(downsample)
    assert spacing_level <= downsample_level

    scale = wsi_object.wsi.level_downsamples[downsample_level] / wsi_object.wsi.level_downsamples[spacing_level]
    downsampled_tile_size = int(tile_size * 1. / scale)

    mask_data = mask_object.wsi.read_region((0,0), downsample_level, mask_object.wsi.level_dimensions[downsample_level]).convert('RGB')

    filtered_grid = []
    tile_percentages = []

    if tile_df is not None:

        # in tile_df, x & y coordinates are defined w.r.t level 0
        # need to downsample them to match input downsample value
        downsample_factor = wsi_object.wsi.level_downsamples[downsample_level]
        for x, y in tile_df[['x', 'y']].values:
            x_downsampled, y_downsampled = scale_coordinates(x, y, 1. / downsample_factor)
            coords = x_downsampled, y_downsampled, x_downsampled+downsampled_tile_size, y_downsampled+downsampled_tile_size
            masked_tile = mask_data.crop(coords)
            pct = get_mask_percent(masked_tile, val)
            if pct > threshold:
                # scale coordinates back to input spacing_level
                x_scaled, y_scaled = scale_coordinates(x_downsampled, y_downsampled, scale)
                filtered_grid.append((x_scaled,y_scaled))
                tile_percentages.append(pct)

        if sort:
            sorted_tile_percentages = sorted(tile_percentages, reverse=True)
            sorted_idxs = sorted(range(len(tile_percentages)), key=lambda k: tile_percentages[k], reverse=True)
            filtered_grid = [filtered_grid[idx] for idx in sorted_idxs]
            tile_percentages = sorted_tile_percentages

            if topk is not None:
                filtered_grid = filtered_grid[:topk]
                tile_percentages = tile_percentages[:topk]

    return filtered_grid, tile_percentages


def overlay_mask_on_tile(
    tile: PIL.Image,
    mask: PIL.Image,
    pixel_mapping: Dict[str,int],
    color_mapping: Optional[Dict[str,List[int]]] = None,
    alpha=0.6,
):

    # Create alpha mask
    mask_arr = np.array(mask)
    alpha_int = int(round(255*alpha))
    if color_mapping is not None:
        alpha_content = np.zeros_like(mask_arr)
        for k, v in pixel_mapping.items():
            if color_mapping[k] is not None:
                alpha_content += (mask_arr == v)
        alpha_content = np.less(alpha_content, 1).astype('uint8') * alpha_int + (255 - alpha_int)
    else:
        alpha_content = np.less_equal(mask_arr, 0).astype('uint8') * alpha_int + (255 - alpha_int)
    alpha_content = Image.fromarray(alpha_content)

    preview_palette = np.zeros(shape=768, dtype=int)

    if color_mapping is None:
        ncat = len(pixel_mapping)
        if ncat <= 10:
            color_palette = sns.color_palette("tab10")[:ncat]
        elif ncat <= 20:
            color_palette = sns.color_palette("tab20")[:ncat]
        else:
            raise ValueError(f"Implementation supports up to 20 categories (provided pixel_mapping has {ncat})")
        color_mapping = {k: color_palette[i]*255 for i, k in enumerate(pixel_mapping.keys())}

    p = [0]*3*len(color_mapping)
    for k, v in pixel_mapping.items():
        if color_mapping[k] is not None:
            p[v*3:v*3+3] = color_mapping[k]
    n = len(p)
    preview_palette[0:n] = np.array(p).astype(int)

    mask.putpalette(data=preview_palette.tolist())
    mask_rgb = mask.convert(mode='RGB')

    overlayed_image = PIL.Image.composite(image1=tile, image2=mask_rgb, mask=alpha_content)
    return overlayed_image


def sample_patches(
    slide_id: str,
    slide_fp: Path,
    annot_mask_fp: Path,
    output_dir: Path,
    pixel_mapping: Dict[str,int],
    seg_params,
    vis_params,
    filter_params,
    patch_params,
    seg_mask_fp: Optional[str] = None,
    enable_mp: bool = False,
    color_mapping: Optional[Dict[str,int]] = None,
    filtering_threshold: float = 0.,
    skip: List[str] = [],
    sort: bool = False,
    topk: Optional[int] = None,
):

    # Inialize WSI & annotation mask
    wsi_object = WholeSlideImage(slide_fp)
    annotation_mask = WholeSlideImage(annot_mask_fp)
    spacing_level = wsi_object.get_best_level_for_spacing(patch_params.spacing)

    vis_level = vis_params.vis_level
    if vis_level < 0:
        if len(wsi_object.level_dim) == 1:
            best_vis_level = 0
        else:
            best_vis_level = wsi_object.get_best_level_for_downsample_custom(
                vis_params.downsample
            )
        vis_params.vis_level = best_vis_level

    seg_level = seg_params.seg_level
    if seg_level < 0:
        if len(wsi_object.level_dim) == 1:
            best_seg_level = 0
        else:
            best_seg_level = wsi_object.get_best_level_for_downsample_custom(
                seg_params.downsample
            )
        seg_params.seg_level = best_seg_level

    w, h = wsi_object.level_dim[seg_params.seg_level]
    if w * h > 1e8:
        print(
            f"level_dim {w} x {h} is likely too large for successful segmentation, aborting"
        )
        return 0

    seg_time = -1
    wsi_object, seg_time = segment(
        wsi_object,
        patch_params.spacing,
        seg_params,
        filter_params,
        seg_mask_fp,
    )

    if seg_params.save_mask:
        seg_mask = wsi_object.visWSI(
            vis_level=vis_params.vis_level,
            line_thickness=vis_params.line_thickness,
        )
        seg_mask_save_dir = Path(output_dir, 'segmentation_mask')
        seg_mask_save_dir.mkdir(parents=True, exist_ok=True)
        seg_mask_path = Path(seg_mask_save_dir, f"{slide_id}.jpg")
        seg_mask.save(seg_mask_path)

    if vis_params.overlay_mask_on_slide:
        overlay_mask = overlay_mask_on_slide(
            wsi_object,
            annotation_mask,
            vis_params.vis_level,
            pixel_mapping,
            color_mapping,
        )
        overlay_mask_save_dir = Path(output_dir, 'annotation_mask')
        overlay_mask_save_dir.mkdir(parents=True, exist_ok=True)
        overlay_mask_path = Path(overlay_mask_save_dir, f"{slide_id}.jpg")
        overlay_mask.save(overlay_mask_path)

    # extract patches from identified tissue blobs
    start_time = time.time()
    _, tile_df = wsi_object.process_contours(
        seg_level=seg_level,
        spacing=patch_params.spacing,
        patch_size=patch_params.patch_size,
        overlap=patch_params.overlap,
        contour_fn=patch_params.contour_fn,
        drop_holes=patch_params.drop_holes,
        tissue_thresh=patch_params.tissue_thresh,
        use_padding=patch_params.use_padding,
        enable_mp=enable_mp,
    )
    patch_time_elapsed = time.time() - start_time

    slide_ids, coordinates, percentages, categories = [], [], [], []
    raw_tile_dir = Path(output_dir, 'patches', 'raw')
    overlay_tile_dir = Path(output_dir, 'patches', 'mask')

    # loop over annotation categories and extract top scoring patches
    # among previously extracted tissue patches
    for cat, pixel_val in pixel_mapping.items():
        if cat not in skip:
            coords, pct = extract_top_tiles(
                wsi_object,
                annotation_mask,
                tile_df,
                patch_params.spacing,
                patch_params.patch_size,
                patch_params.downsample,
                pixel_val,
                filtering_threshold,
                sort,
                topk,
            )
            slide_ids.extend([slide_id]*len(coords))
            coordinates.extend(coords)
            percentages.extend(pct)
            categories.extend([cat]*len(coords))

            if patch_params.save_patches_to_disk:
                for x, y in coords:
                    tile = wsi_object.wsi.read_region((x,y), spacing_level, (patch_params.patch_size, patch_params.patch_size)).convert('RGB')
                    fname = f'{slide_id}_{x}_{y}'
                    tile_fp = Path(raw_tile_dir, cat, f'{fname}.{patch_params.fmt}')
                    tile_fp.parent.mkdir(exist_ok=True, parents=True)
                    tile.save(tile_fp)

                    if vis_params.overlay_mask_on_patch:
                        masked_tile = annotation_mask.wsi.read_region((x,y), spacing_level, (patch_params.patch_size, patch_params.patch_size))
                        # mask data is present in the R channel
                        masked_tile = masked_tile.split()[0]
                        overlayed_tile = overlay_mask_on_tile(tile, masked_tile, pixel_mapping, color_mapping, alpha=0.5)
                        overlayed_tile_fp = Path(overlay_tile_dir, cat, f'{fname}_mask.{patch_params.fmt}')
                        overlayed_tile_fp.parent.mkdir(exist_ok=True, parents=True)
                        overlayed_tile.save(overlayed_tile_fp)

    if len(coordinates) > 0:
        x, y = list(map(list, zip(*coordinates)))
    else:
        x, y = [], []
    sampled_tiles_df = pd.DataFrame.from_dict({
        'slide_id': slide_ids,
        'category': categories,
        'x': x,
        'y': y,
        'pct': percentages,
    })
    return sampled_tiles_df


@hydra.main(version_base="1.2.0", config_path="config/sampling", config_name="default")
def main(cfg: DictConfig):

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")
        run_id = wandb_run.id

    pixel_mapping = {k: v for e in cfg.pixel_mapping for k, v in e.items()}
    if cfg.color_mapping is not None:
        color_mapping = {k: v for e in cfg.color_mapping for k, v in e.items()}
    else:
        color_mapping = None

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(cfg.csv)

    slide_ids = df['slide_id'].values.tolist()
    slide_fps = df['slide_path'].values.tolist()
    seg_mask_fps = [None] * len(slide_fps)
    if "segmentation_mask_path" in df.columns:
        seg_mask_fps = df['segmentation_mask_path'].values.tolist()
    annot_mask_fps = df['annotation_mask_path'].values.tolist()

    if cfg.speed.multiprocessing:

        args = [
            (
                sid,
                Path(slide_fp),
                Path(annot_mask_fp),
                output_dir,
                pixel_mapping,
                cfg.seg_params,
                cfg.vis_params,
                cfg.filter_params,
                cfg.patch_params,
                seg_mask_fp,
                False,
                color_mapping,
                cfg.filtering_threshold,
                cfg.skip_category,
                cfg.sort,
                cfg.topk,
            )
            for sid, slide_fp, seg_mask_fp, annot_mask_fp in zip(slide_ids, slide_fps, seg_mask_fps, annot_mask_fps)
        ]

        num_workers = mp.cpu_count()
        if num_workers > cfg.speed.num_workers:
            num_workers = cfg.speed.num_workers
        dfs = []
        with mp.Pool(num_workers) as pool:
            for i, r in enumerate(pool.starmap(sample_patches, args)):
                dfs.append(r)

        tile_df = pd.concat(dfs, ignore_index=True)
        tiles_fp = Path(output_dir, f'sampled_patches.csv')
        tile_df.to_csv(tiles_fp, index=False)

    else:

        dfs = []

        with tqdm.tqdm(
            zip(slide_ids, slide_fps, seg_mask_fps, annot_mask_fps),
            desc=f"",
            unit=" slide",
            leave=True,
        ) as t:

            for sid, slide_fp, seg_mask_fp, annot_mask_fp in t:

                t_df = sample_patches(
                    sid,
                    Path(slide_fp),
                    Path(annot_mask_fp),
                    output_dir,
                    pixel_mapping,
                    cfg.seg_params,
                    cfg.vis_params,
                    cfg.filter_params,
                    cfg.patch_params,
                    seg_mask_fp=seg_mask_fp,
                    enable_mp=True,
                    color_mapping=color_mapping,
                    filtering_threshold=cfg.filtering_threshold,
                    skip=cfg.skip_category,
                    sort=cfg.sort,
                    topk=cfg.topk,
                )
                if t_df is not None:
                    dfs.append(t_df)

            df = pd.concat(dfs, ignore_index=True)
            tiles_fp = Path(output_dir, f'sampled_patches.csv')
            df.to_csv(tiles_fp, index=False)



if __name__ == "__main__":

    main()
    # zip -r tiles.zip './tiles'

