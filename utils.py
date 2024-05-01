import re
import h5py
import time
import copy
import tqdm
import wandb
import subprocess
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Optional, Tuple

from source.wsi import WholeSlideImage
from source.utils import (
    VisualizeCoords,
    find_common_spacings,
    overlay_mask_on_slide,
    overlay_mask_on_tile,
    get_masked_tile,
    compute_time,
    initialize_df,
)


def write_dictconfig(d, f, child: bool = False, ntab=0):
    for k, v in d.items():
        if isinstance(v, dict):
            if not child:
                f.write(f"{k}:\n")
            else:
                for _ in range(ntab):
                    f.write("\t")
                f.write(f"- {k}:\n")
            write_dictconfig(v, f, True, ntab=ntab + 1)
        else:
            if isinstance(v, list):
                if not child:
                    f.write(f"{k}:\n")
                    for e in v:
                        f.write(f"\t- {e}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"{k}:\n")
                    for e in v:
                        for _ in range(ntab):
                            f.write("\t")
                        f.write(f"\t- {e}\n")
            else:
                if not child:
                    f.write(f"{k}: {v}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"- {k}: {v}\n")


def initialize_wandb(
    cfg: DictConfig,
    key: Optional[str] = "",
):
    command = f"wandb login {key}"
    subprocess.call(command, shell=True)
    if cfg.wandb.tags == None:
        tags = []
    else:
        tags = [str(t) for t in cfg.wandb.tags]
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.username,
        name=cfg.wandb.exp_name,
        group=cfg.wandb.group,
        dir=cfg.wandb.dir,
        config=config,
        tags=tags,
        resume="allow",
    )
    config_file_path = Path(run.dir, "run_config.yaml")
    d = OmegaConf.to_container(cfg, resolve=True)
    with open(config_file_path, "w+") as f:
        write_dictconfig(d, f)
        wandb.save(str(config_file_path))
        f.close()
    return run


def log_progress(processed_count, stop_logging, ntot):
    previous_count = 0
    while not stop_logging.is_set():
        current_count = processed_count.value
        if previous_count != current_count:
            wandb.log({"processed": current_count})
            previous_count = current_count
        if current_count >= ntot:
            break


def segment(
    wsi_object: WholeSlideImage,
    spacing: float,
    seg_params: DictConfig,
    filter_params: DictConfig,
    mask_fp: Optional[Path] = None,
):
    start_time = time.time()
    if mask_fp is not None:
        seg_level = wsi_object.loadSegmentation(
            mask_fp,
            spacing=spacing,
            downsample=seg_params.downsample,
            filter_params=filter_params,
            tissue_val=seg_params.tissue_pixel_value,
        )
        seg_params.seg_level = seg_level
    else:
        wsi_object.segmentTissue(
            spacing=spacing,
            seg_level=seg_params.seg_level,
            sthresh=seg_params.sthresh,
            mthresh=seg_params.mthresh,
            close=seg_params.close,
            use_otsu=seg_params.use_otsu,
            filter_params=filter_params,
        )
    seg_time_elapsed = time.time() - start_time
    return wsi_object, seg_time_elapsed


def patching(
    wsi_object: WholeSlideImage,
    save_dir: Path,
    seg_level: int,
    spacing: float,
    patch_size: int,
    overlap: float,
    contour_fn: str,
    drop_holes: bool,
    tissue_thresh: float,
    use_padding: bool,
    save_patches_to_disk: bool,
    save_patches_in_common_dir: bool,
    patch_format: str = "png",
    top_left: Optional[List[int]] = None,
    bot_right: Optional[List[int]] = None,
    num_workers: int = 1,
    verbose: bool = False,
):

    start_time = time.time()
    file_path, tile_df = wsi_object.process_contours(
        save_dir=save_dir,
        seg_level=seg_level,
        spacing=spacing,
        patch_size=patch_size,
        overlap=overlap,
        contour_fn=contour_fn,
        drop_holes=drop_holes,
        tissue_thresh=tissue_thresh,
        use_padding=use_padding,
        save_patches_to_disk=save_patches_to_disk,
        save_patches_in_common_dir=save_patches_in_common_dir,
        patch_format=patch_format,
        top_left=top_left,
        bot_right=bot_right,
        num_workers=num_workers,
        verbose=verbose,
    )
    patch_time_elapsed = time.time() - start_time
    return file_path, tile_df, patch_time_elapsed


def visualize(
    file_path: Path,
    wsi_object: WholeSlideImage,
    downscale: int = 64,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    draw_grid: bool = False,
    thickness: int = 2,
    verbose: bool = False,
):
    start = time.time()
    heatmap = VisualizeCoords(
        file_path,
        wsi_object,
        downscale=downscale,
        bg_color=bg_color,
        draw_grid=draw_grid,
        thickness=thickness,
        verbose=verbose,
    )
    total_time = time.time() - start
    return heatmap, total_time


def seg_and_patch(
    output_dir: Path,
    patch_save_dir: Path,
    mask_save_dir: Path,
    visu_save_dir: Path,
    seg_params,
    filter_params,
    vis_params,
    patch_params,
    slide_df: pd.DataFrame,
    visu: bool = False,
    patch: bool = False,
    process_list: Optional[Path] = None,
    num_workers: int = 1,
    verbose: bool = False,
    log_to_wandb: bool = False,
    backend: str = "asap",
):
    start_time = time.time()
    slide_paths = slide_df.slide_path.values.tolist()
    mask_paths = []
    if "segmentation_mask_path" in slide_df.columns:
        mask_paths = slide_df.segmentation_mask_path.values.tolist()
    spacings = []
    if "spacing" in slide_df.columns:
        spacings = slide_df.spacing.values.tolist()

    if process_list is None:
        df = initialize_df(
            slide_paths,
            mask_paths,
            spacings,
            seg_params,
            filter_params,
            vis_params,
            patch_params,
        )
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df["process"] == 1
    process_stack = df[mask]
    total = len(process_stack)
    already_processed = len(df) - total

    seg_times = 0.0
    patch_times = 0.0
    visu_times = 0.0

    dfs = []

    with tqdm.tqdm(
        range(total),
        desc=(f"Patch Extraction"),
        unit=" slide",
        ncols=100,
        initial=already_processed,
        total=total + already_processed,
        leave=True,
    ) as t:

        for i in t:

            idx = process_stack.index[i]
            slide_id = process_stack.loc[idx, "slide_id"]
            slide_path = Path(process_stack.loc[idx, "slide_path"])
            mask_path = None
            if "segmentation_mask_path" in process_stack.columns:
                mask_path = Path(process_stack.loc[idx, "segmentation_mask_path"])
            spacing = None
            if "spacing" in process_stack.columns:
                spacing = float(process_stack.loc[idx, "spacing"])
            t.display(f"Processing {slide_id}", pos=2)

            # Inialize WSI
            wsi_object = WholeSlideImage(slide_path, spacing, backend)

            vis_level = vis_params.vis_level
            if vis_level < 0:
                if len(wsi_object.level_dimensions) == 1:
                    vis_params.vis_level = 0
                else:
                    best_level = wsi_object.get_best_level_for_downsample_custom(
                        vis_params.downsample
                    )
                    vis_params.vis_level = best_level

            seg_level = seg_params.seg_level
            if seg_level < 0:
                if len(wsi_object.level_dimensions) == 1:
                    seg_params.seg_level = 0
                else:
                    best_level = wsi_object.get_best_level_for_downsample_custom(
                        seg_params.downsample
                    )
                    seg_params.seg_level = best_level

            w, h = wsi_object.level_dimensions[seg_params.seg_level]
            if w * h > 1e8:
                print(
                    f"level dimensions {w} x {h} is likely too large for successful segmentation, aborting"
                )
                df.loc[idx, "status"] = "failed_seg"
                continue

            seg_time_elapsed = -1
            wsi_object, seg_time_elapsed = segment(
                wsi_object,
                patch_params.spacing,
                seg_params,
                filter_params,
                mask_path,
            )

            if seg_params.save_mask:
                import pyvips

                tif_save_dir = Path(mask_save_dir, "tif")
                tif_save_dir.mkdir(exist_ok=True)
                mask_path = Path(tif_save_dir, f"{slide_id}.tif")
                mask = pyvips.Image.new_from_array(wsi_object.binary_mask.tolist())
                mask.tiffsave(
                    mask_path,
                    tile=True,
                    compression="jpeg",
                    bigtiff=True,
                    pyramid=True,
                    Q=70,
                )

            if seg_params.visualize_mask:
                mask = wsi_object.visWSI(
                    vis_level=vis_params.vis_level,
                    line_thickness=vis_params.line_thickness,
                )
                jpg_save_dir = Path(mask_save_dir, "jpg")
                jpg_save_dir.mkdir(exist_ok=True)
                mask_path = Path(jpg_save_dir, f"{slide_id}.jpg")
                mask.save(mask_path)

            patch_time_elapsed = -1
            if patch:
                slide_save_dir = Path(patch_save_dir, slide_id)
                file_path, tile_df, patch_time_elapsed = patching(
                    wsi_object=wsi_object,
                    save_dir=slide_save_dir,
                    seg_level=seg_params.seg_level,
                    spacing=patch_params.spacing,
                    patch_size=patch_params.patch_size,
                    overlap=patch_params.overlap,
                    contour_fn=patch_params.contour_fn,
                    drop_holes=patch_params.drop_holes,
                    tissue_thresh=patch_params.tissue_thresh,
                    use_padding=patch_params.use_padding,
                    save_patches_to_disk=patch_params.save_patches_to_disk,
                    save_patches_in_common_dir=patch_params.save_patches_in_common_dir,
                    patch_format=patch_params.format,
                    num_workers=num_workers,
                    verbose=verbose,
                )
                if tile_df is not None:
                    dfs.append(tile_df)

            visu_time_elapsed = -1
            if visu:
                # if file_path exists, patches were extracted
                if file_path.is_file():
                    heatmap, visu_time_elapsed = visualize(
                        file_path,
                        wsi_object,
                        downscale=vis_params.downscale,
                        bg_color=tuple(patch_params.bg_color),
                        draw_grid=patch_params.draw_grid,
                        thickness=patch_params.grid_thickness,
                        verbose=verbose,
                    )
                    visu_path = Path(visu_save_dir, f"{slide_id}.jpg")
                    heatmap.save(visu_path)
                    df.loc[idx, "has_patches"] = "yes"
                else:
                    df.loc[idx, "has_patches"] = "no"

            df.loc[idx, "process"] = 0
            df.loc[idx, "status"] = "processed"
            df.loc[idx, "vis_level"] = vis_params.vis_level
            df.loc[idx, "seg_level"] = seg_params.seg_level
            df.to_csv(Path(output_dir, "process_list.csv"), index=False)

            seg_times += seg_time_elapsed
            patch_times += patch_time_elapsed
            visu_times += visu_time_elapsed

            if log_to_wandb:
                wandb.log({"processed": already_processed + i + 1})

            # restore original values
            vis_params.vis_level = vis_level
            seg_params.seg_level = seg_level

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)

    seg_times /= total + 1e-5
    patch_times /= total + 1e-5
    visu_times /= total + 1e-5

    tile_df = pd.concat(dfs, ignore_index=True)
    tile_df.to_csv(Path(output_dir, f"tiles.csv"), index=False)

    df.to_csv(Path(output_dir, "process_list.csv"), index=False)
    print(f"\n" * 4)
    print("-" * 7, "summary", "-" * 7)
    print(f"total time taken: \t{mins}m{secs}s")
    print(f"average segmentation time per slide: \t{seg_times:.2f}s")
    print(f"average patching time per slide: \t{patch_times:.2f}s")
    print(f"average stiching time per slide: \t{visu_times:.2f}s")
    print("-" * 7, "-" * (len("summary") + 2), "-" * 7, sep="")

    return seg_times, patch_times


def seg_and_patch_slide(
    patch_save_dir: Path,
    mask_save_dir: Path,
    visu_save_dir: Path,
    seg_params,
    filter_params,
    vis_params,
    patch_params,
    slide_id: str,
    slide_fp: str,
    mask_fp: str,
    spacing: float,
    patch: bool = False,
    visu: bool = False,
    verbose: bool = False,
    backend: str = "asap",
):
    start_time = time.time()
    if verbose:
        print(f"Processing {slide_id}...")

    if mask_fp is not None:
        mask_fp = Path(mask_fp)

    # Inialize WSI
    wsi_object = WholeSlideImage(Path(slide_fp), spacing, backend)

    vis_level = vis_params.vis_level
    best_vis_level = copy.deepcopy(vis_level)
    if vis_level < 0:
        if len(wsi_object.level_dimensions) == 1:
            vis_params.vis_level = 0
            best_vis_level = 0
        else:
            best_vis_level = wsi_object.get_best_level_for_downsample_custom(
                vis_params.downsample
            )
            vis_params.vis_level = best_vis_level

    seg_level = seg_params.seg_level
    best_seg_level = copy.deepcopy(seg_level)
    if seg_level < 0:
        if len(wsi_object.level_dimensions) == 1:
            seg_params.seg_level = 0
            best_seg_level = 0
        else:
            best_seg_level = wsi_object.get_best_level_for_downsample_custom(
                seg_params.downsample
            )
            seg_params.seg_level = best_seg_level

    w, h = wsi_object.level_dimensions[seg_params.seg_level]
    if w * h > 1e8:
        print(
            f"{slide_id}: level dimensions ({w},{h}) is likely too large for successful segmentation, aborting"
        )
        status = "failed_seg"
        tile_df = pd.DataFrame.from_dict(
            {
                "slide_id": [],
                "tile_size": [],
                "spacing": [],
                "level": [],
                "level_dimensions": [],
                "x": [],
                "y": [],
                "contour": [],
            }
        )
        return tile_df, slide_id, status, best_vis_level, best_seg_level, None

    seg_time = -1
    wsi_object, seg_time = segment(
        wsi_object,
        patch_params.spacing,
        seg_params,
        filter_params,
        mask_fp,
    )

    if seg_params.save_mask:
        import pyvips

        tif_save_dir = Path(mask_save_dir, "tif")
        tif_save_dir.mkdir(exist_ok=True)
        mask_path = Path(tif_save_dir, f"{slide_id}.tif")
        mask = pyvips.Image.new_from_array(wsi_object.binary_mask.tolist())
        mask.tiffsave(
            mask_path, tile=True, compression="jpeg", bigtiff=True, pyramid=True, Q=70
        )

    if seg_params.visualize_mask:
        mask = wsi_object.visWSI(
            vis_level=vis_params.vis_level,
            line_thickness=vis_params.line_thickness,
        )
        jpg_save_dir = Path(mask_save_dir, "jpg")
        jpg_save_dir.mkdir(exist_ok=True)
        mask_path = Path(jpg_save_dir, f"{slide_id}.jpg")
        mask.save(mask_path)

    patch_time = -1
    if patch:
        slide_save_dir = Path(patch_save_dir, slide_id)
        file_path, tile_df, patch_time = patching(
            wsi_object=wsi_object,
            save_dir=slide_save_dir,
            seg_level=seg_params.seg_level,
            spacing=patch_params.spacing,
            patch_size=patch_params.patch_size,
            overlap=patch_params.overlap,
            contour_fn=patch_params.contour_fn,
            drop_holes=patch_params.drop_holes,
            tissue_thresh=patch_params.tissue_thresh,
            use_padding=patch_params.use_padding,
            save_patches_to_disk=patch_params.save_patches_to_disk,
            save_patches_in_common_dir=patch_params.save_patches_in_common_dir,
            patch_format=patch_params.format,
            num_workers=1,
            verbose=verbose,
        )
    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    process_time = mins * 60 + secs

    visu_time = -1
    if visu:
        # if file_path exists, patches were extracted
        if file_path.is_file():
            heatmap, visu_time = visualize(
                file_path,
                wsi_object,
                downscale=vis_params.downscale,
                bg_color=tuple(patch_params.bg_color),
                draw_grid=patch_params.draw_grid,
                thickness=patch_params.grid_thickness,
                verbose=verbose,
            )
            visu_path = Path(visu_save_dir, f"{slide_id}.jpg")
            heatmap.save(visu_path)

    status = "processed"

    # restore original values
    vis_params.vis_level = vis_level
    seg_params.seg_level = seg_level

    return tile_df, slide_id, status, best_vis_level, best_seg_level, process_time


def seg_and_patch_slide_mp(
    args,
):
    (
        patch_save_dir,
        mask_save_dir,
        visu_save_dir,
        seg_params,
        filter_params,
        vis_params,
        patch_params,
        slide_id,
        slide_fp,
        mask_fp,
        spacing,
        patch,
        visu,
        verbose,
        backend,
    ) = args

    tile_df, slide_id, status, best_vis_level, best_seg_level, process_time = (
        seg_and_patch_slide(
            patch_save_dir,
            mask_save_dir,
            visu_save_dir,
            seg_params,
            filter_params,
            vis_params,
            patch_params,
            slide_id,
            slide_fp,
            mask_fp,
            spacing,
            patch,
            visu,
            verbose,
            backend,
        )
    )

    return tile_df, slide_id, status, best_vis_level, best_seg_level, process_time


def get_mask_percent(mask, val=0):
    """
    Determine the percentage of a mask that is equal to gleason score gs
    input:
        mask: mask as a PIL Image
        val: given value we want to check for
    output:
        percentage of the numpy array that is equal to val
    """
    mask_arr = np.array(mask)
    binary_mask = mask_arr == val
    mask_percentage = np.sum(binary_mask) / np.size(mask_arr)
    return mask_percentage


def extract_top_tiles(
    wsi_object,
    mask_object,
    tile_df,
    spacing: float,
    tile_size: int,
    downsample: int,
    pixel_val: int,
    threshold: float = 0.0,
    sort: bool = False,
    topk: Optional[int] = None,
):

    common_spacings = find_common_spacings(
        wsi_object.spacings, mask_object.spacings, tolerance=0.1
    )
    assert (
        len(common_spacings) >= 1
    ), f"The provided segmentation mask (spacings={mask_object.spacings}) has no common spacing with the slide (spacings={wsi_object.spacings}). A minimum of 1 common spacing is required."

    if downsample == -1:
        overlay_spacing = spacing
        overlay_level = wsi_object.get_best_level_for_spacing(overlay_spacing)
    else:
        overlay_level = wsi_object.get_best_level_for_downsample_custom(downsample)
        overlay_spacing = wsi_object.get_level_spacing(overlay_level)

    # check if this spacing is present in common spacings
    is_in_common_spacings = overlay_spacing in [s for s, _ in common_spacings]
    if not is_in_common_spacings:
        # find spacing that is common to slide and mask and that is the closest to overlay_spacing
        closest = np.argmin([abs(overlay_spacing - s) for s, _ in common_spacings])
        closest_common_spacing = common_spacings[closest][0]
        overlay_spacing = closest_common_spacing
        overlay_level = wsi_object.get_best_level_for_spacing(overlay_spacing)

    mask_data = mask_object.wsi.get_slide(spacing=overlay_spacing)
    if mask_data.shape[-1] == 1:
        mask_data = np.squeeze(mask_data, axis=-1)
    mask_data = Image.fromarray(mask_data)
    mask_data = mask_data.split()[0]

    spacing_level = wsi_object.get_best_level_for_spacing(spacing)
    wsi_scale = tuple(
        a / b
        for a, b in zip(
            wsi_object.level_downsamples[overlay_level],
            wsi_object.level_downsamples[spacing_level],
        )
    )
    downsampled_tile_size = tuple(int(tile_size * 1.0 / s) for s in wsi_scale)

    filtered_grid = []
    tile_percentages = []

    if tile_df is not None:

        # in tile_df, x & y coordinates are defined w.r.t level 0 in the slide
        # need to downsample them to match input downsample value
        for x, y in tile_df[["x", "y"]].values:
            downsample_factor = wsi_object.level_downsamples[overlay_level]
            x_downsampled, y_downsampled = int(x * 1.0 / downsample_factor[0]), int(
                y * 1.0 / downsample_factor[1]
            )
            # crop mask tile
            coords = (
                x_downsampled,
                y_downsampled,
                x_downsampled + downsampled_tile_size[0],
                y_downsampled + downsampled_tile_size[1],
            )
            masked_tile = mask_data.crop(coords)
            pct = get_mask_percent(masked_tile, pixel_val)
            if pct > threshold:
                # scale coordinates back to input spacing_level
                filtered_grid.append((int(x), int(y)))
                tile_percentages.append(pct)

        if sort:
            sorted_tile_percentages = sorted(tile_percentages, reverse=True)
            sorted_idxs = sorted(
                range(len(tile_percentages)),
                key=lambda k: tile_percentages[k],
                reverse=True,
            )
            filtered_grid = [filtered_grid[idx] for idx in sorted_idxs]
            tile_percentages = sorted_tile_percentages

            if topk is not None:
                filtered_grid = filtered_grid[:topk]
                tile_percentages = tile_percentages[:topk]

    return filtered_grid, tile_percentages


def sample_patches(
    slide_id: str,
    slide_fp: Path,
    annot_mask_fp: Path,
    output_dir: Path,
    pixel_mapping: Dict[str, int],
    visu: bool,
    seg_params,
    vis_params,
    filter_params,
    patch_params,
    spacing: Optional[float] = None,
    seg_mask_fp: Optional[str] = None,
    num_workers: int = 1,
    color_mapping: Optional[Dict[str, int]] = None,
    filtering_threshold: float = 0.0,
    skip: List[str] = [],
    sort: bool = False,
    topk: Optional[int] = None,
    alpha: float = 0.5,
    seg_mask_save_dir: Optional[Path] = None,
    overlay_mask_save_dir: Optional[Path] = None,
    backend: str = "asap",
    eps: float = 1e-5,
):

    # Inialize WSI & annotation mask
    wsi_object = WholeSlideImage(slide_fp, spacing, backend)
    annotation_mask = WholeSlideImage(annot_mask_fp, backend=backend)

    vis_level = vis_params.vis_level
    if vis_level < 0:
        if len(wsi_object.level_dimensions) == 1:
            best_vis_level = 0
        else:
            best_vis_level = wsi_object.get_best_level_for_downsample_custom(
                vis_params.downsample
            )
        vis_params.vis_level = best_vis_level

    seg_level = seg_params.seg_level
    if seg_level < 0:
        if len(wsi_object.level_dimensions) == 1:
            best_seg_level = 0
        else:
            best_seg_level = wsi_object.get_best_level_for_downsample_custom(
                seg_params.downsample
            )
        seg_params.seg_level = best_seg_level

    w, h = wsi_object.level_dimensions[seg_params.seg_level]
    if w * h > 1e8:
        print(
            f"level dimensions {w} x {h} is likely too large for successful segmentation, aborting"
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
        seg_mask_path = Path(seg_mask_save_dir, f"{slide_id}.jpg")
        seg_mask.save(seg_mask_path)

    # extract patches from identified tissue blobs
    start_time = time.time()
    _, tile_df = wsi_object.process_contours(
        seg_level=seg_params.seg_level,
        spacing=patch_params.spacing,
        patch_size=patch_params.patch_size,
        overlap=patch_params.overlap,
        contour_fn=patch_params.contour_fn,
        drop_holes=patch_params.drop_holes,
        tissue_thresh=patch_params.tissue_thresh,
        use_padding=patch_params.use_padding,
        num_workers=num_workers,
    )
    patch_time_elapsed = time.time() - start_time

    slide_ids, coordinates, percentages, categories = [], [], [], []

    patch_dir = Path(output_dir, "patches")

    h5_dir = Path(patch_dir, "h5")
    h5_dir.mkdir(exist_ok=True, parents=True)
    hdf5_file_path = Path(h5_dir, f"{slide_id}.h5")

    raw_tile_dir = Path(patch_dir, "raw")
    overlay_tile_dir = Path(patch_dir, "mask")

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

            slide_ids.extend([slide_id] * len(coords))
            coordinates.extend(coords)
            percentages.extend(pct)
            categories.extend([cat] * len(coords))

            # save coords to h5py file
            h5_file = h5py.File(hdf5_file_path, "a")
            cat_coords = np.array(coords)
            data_shape = cat_coords.shape
            data_type = cat_coords.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = h5_file.create_dataset(
                cat,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type,
            )
            dset[:] = cat_coords
            dset.attrs["patch_size"] = patch_params.patch_size
            dset.attrs["patch_level"] = wsi_object.get_best_level_for_spacing(
                patch_params.spacing
            )
            h5_file.close()

            if patch_params.save_patches_to_disk:

                cat_ = re.sub(r"\s+", "_", cat)

                with tqdm.tqdm(
                    coords,
                    desc=(f"Processing {slide_id}"),
                    unit=f" {cat_} patch",
                    initial=0,
                    total=len(coords),
                    leave=False,
                ) as t:

                    for x, y in t:

                        tile = wsi_object.wsi.get_patch(
                            x,
                            y,
                            patch_params.patch_size,
                            patch_params.patch_size,
                            spacing=patch_params.spacing,
                            center=False,
                        )
                        tile = Image.fromarray(tile).convert("RGB")
                        fname = f"{slide_id}_{x}_{y}"
                        tile_fp = Path(
                            raw_tile_dir, cat_, f"{fname}.{patch_params.fmt}"
                        )
                        tile_fp.parent.mkdir(exist_ok=True, parents=True)
                        tile.save(tile_fp)

                        if vis_params.overlay_mask_on_patch:

                            tile, masked_tile = get_masked_tile(
                                wsi_object,
                                annotation_mask,
                                tile,
                                x,
                                y,
                                patch_params.spacing,
                                (patch_params.patch_size, patch_params.patch_size),
                            )

                            overlayed_tile = overlay_mask_on_tile(
                                tile,
                                masked_tile,
                                pixel_mapping,
                                color_mapping,
                                alpha=alpha,
                            )
                            overlayed_tile_fp = Path(
                                overlay_tile_dir,
                                cat_,
                                f"{fname}_mask.{patch_params.fmt}",
                            )
                            overlayed_tile_fp.parent.mkdir(exist_ok=True, parents=True)
                            overlayed_tile.save(overlayed_tile_fp)

    # restore original values
    vis_params.vis_level = vis_level
    seg_params.seg_level = seg_level

    if vis_params.overlay_mask_on_slide:
        overlay_mask = overlay_mask_on_slide(
            wsi_object,
            annotation_mask,
            vis_params.vis_level,
            pixel_mapping,
            color_mapping,
            alpha,
            vis_params.downscale,
        )
        overlay_mask_path = Path(overlay_mask_save_dir, f"{slide_id}.jpg")
        overlay_mask.save(overlay_mask_path)

    if visu and hdf5_file_path.exists():
        visu_save_dir = Path(output_dir, "visualization")
        visu_save_dir.mkdir(exist_ok=True)
        heatmaps = [None]
        for cat in pixel_mapping.keys():
            if cat not in skip:
                heatmap = VisualizeCoords(
                    hdf5_file_path,
                    wsi_object,
                    downscale=vis_params.downscale,
                    draw_grid=True,
                    thickness=patch_params.grid_thickness,
                    key=cat,
                    heatmap=heatmaps[-1],
                    mask_object=annotation_mask,
                    pixel_mapping=pixel_mapping,
                    color_mapping=color_mapping,
                    alpha=alpha,
                )
                heatmaps.append(heatmap)
        # the last heatmaps element contains the final visualization image
        heatmap = [h for h in heatmaps if h is not None]
        if len(heatmap) > 0:
            heatmap = heatmap[-1]
            visu_path = Path(visu_save_dir, f"{slide_id}.jpg")
            heatmap.save(visu_path)

    if len(coordinates) > 0:
        x, y = list(map(list, zip(*coordinates)))
    else:
        x, y = [], []
    sampled_tiles_df = pd.DataFrame.from_dict(
        {
            "slide_id": slide_ids,
            "category": categories,
            "x": x,
            "y": y,
            "pct": percentages,
        }
    )

    return sampled_tiles_df


def sample_patches_mp(
    args,
):
    (
        slide_id,
        slide_fp,
        annot_mask_fp,
        output_dir,
        pixel_mapping,
        visu,
        seg_params,
        vis_params,
        filter_params,
        patch_params,
        spacing,
        seg_mask_fp,
        num_workers,
        color_mapping,
        filtering_threshold,
        skip,
        sort,
        topk,
        alpha,
        seg_mask_save_dir,
        overlay_mask_save_dir,
        backend,
    ) = args

    sampled_tiles_df = sample_patches(
        slide_id,
        slide_fp,
        annot_mask_fp,
        output_dir,
        pixel_mapping,
        visu,
        seg_params,
        vis_params,
        filter_params,
        patch_params,
        spacing,
        seg_mask_fp,
        num_workers,
        color_mapping,
        filtering_threshold,
        skip,
        sort,
        topk,
        alpha,
        seg_mask_save_dir,
        overlay_mask_save_dir,
        backend,
    )

    return sampled_tiles_df
