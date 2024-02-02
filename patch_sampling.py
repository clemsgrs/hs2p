import os
import re
import h5py
import tqdm
import time
import wandb
import hydra
import datetime
import subprocess
import pandas as pd
import numpy as np
import multiprocessing as mp

from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
from omegaconf import DictConfig

from utils import initialize_wandb, segment
from source.wsi import WholeSlideImage
from source.utils import (
    find_common_spacings,
    VisualizeCoords,
    overlay_mask_on_slide,
    overlay_mask_on_tile,
    get_masked_tile,
)


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

    common_spacings = find_common_spacings(wsi_object.spacings, mask_object.spacings, tolerance=0.1)
    assert len(common_spacings) >= 1, f"The provided segmentation mask (spacings={mask_object.spacings}) has no common spacing with the slide (spacings={wsi_object.spacings}). A minimum of 1 common spacing is required."

    if downsample == -1:
        overlay_spacing = spacing
        overlay_level = wsi_object.get_best_level_for_spacing(overlay_spacing)
    else:
        overlay_level = wsi_object.get_best_level_for_downsample_custom(downsample)
        overlay_spacing = wsi_object.get_level_spacing(overlay_level)

    # check if this spacing is present in common spacings
    is_in_common_spacings = overlay_spacing in [s for s,_ in common_spacings]
    if not is_in_common_spacings:
        # find spacing that is common to slide and mask and that is the closest to overlay_spacing
        closest = np.argmin([abs(overlay_spacing-s) for s,_ in common_spacings])
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
    enable_mp: bool = False,
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
        enable_mp=enable_mp,
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


@hydra.main(
    version_base="1.2.0", config_path="config/sampling", config_name="witali_liver"
)
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

    seg_mask_save_dir = Path(output_dir, "segmentation_mask")
    overlay_mask_save_dir = Path(output_dir, "annotation_mask")
    seg_mask_save_dir.mkdir(exist_ok=True)
    overlay_mask_save_dir.mkdir(exist_ok=True)

    df = pd.read_csv(cfg.csv)

    slide_ids = df["slide_id"].values.tolist()
    slide_fps = df["slide_path"].values.tolist()
    seg_mask_fps = [None] * len(slide_fps)
    if "segmentation_mask_path" in df.columns:
        seg_mask_fps = [Path(f) for f in df["segmentation_mask_path"].values.tolist()]
    annot_mask_fps = df["annotation_mask_path"].values.tolist()

    spacings = [None] * len(slide_ids)
    if "spacing" in df.columns:
        spacings = df.spacing.values.tolist()

    if cfg.speed.multiprocessing:

        args = [
            (
                sid,
                Path(slide_fp),
                Path(annot_mask_fp),
                output_dir,
                pixel_mapping,
                cfg.visu,
                cfg.seg_params,
                cfg.vis_params,
                cfg.filter_params,
                cfg.patch_params,
                spacing,
                seg_mask_fp,
                False,
                color_mapping,
                cfg.filtering_threshold,
                cfg.skip_category,
                cfg.sort,
                cfg.topk,
                cfg.alpha,
                seg_mask_save_dir,
                overlay_mask_save_dir,
                cfg.backend,
            )
            for sid, slide_fp, seg_mask_fp, annot_mask_fp, spacing in zip(
                slide_ids, slide_fps, seg_mask_fps, annot_mask_fps, spacings
            )
        ]

        wd = Path(__file__).parent
        log_file = Path(wd, "log_nproc.py")
        command_line = [
            "python3",
            f"{log_file}",
            "--output_dir",
            f"{overlay_mask_save_dir}",
            "--fmt",
            "jpg",
            "--total",
            f"{len(slide_ids)}",
        ]
        if cfg.wandb.enable:
            command_line = command_line + ["--log_to_wandb", "--id", f"{run_id}", "--project", f"{cfg.wandb.project}", "--username", f"{cfg.wandb.username}"]
        subprocess.Popen(command_line)

        num_workers = mp.cpu_count()
        if num_workers > cfg.speed.num_workers:
            num_workers = cfg.speed.num_workers
        dfs = []
        with mp.Pool(num_workers) as pool:
            for i, r in enumerate(pool.starmap(sample_patches, args)):
                dfs.append(r)
        if cfg.wandb.enable:
            wandb.log({"processed": len(dfs)})

        tile_df = pd.concat(dfs, ignore_index=True)
        tiles_fp = Path(output_dir, f"sampled_patches.csv")
        tile_df.to_csv(tiles_fp, index=False)

    else:

        dfs = []

        with tqdm.tqdm(
            zip(slide_ids, slide_fps, seg_mask_fps, annot_mask_fps, spacings),
            desc=f"Sampling Patches",
            unit=" slide",
            initial=0,
            total=len(slide_ids),
            leave=True,
        ) as t:

            for i, (sid, slide_fp, seg_mask_fp, annot_mask_fp, spacing) in enumerate(t):

                t_df = sample_patches(
                    sid,
                    Path(slide_fp),
                    Path(annot_mask_fp),
                    output_dir,
                    pixel_mapping,
                    cfg.visu,
                    cfg.seg_params,
                    cfg.vis_params,
                    cfg.filter_params,
                    cfg.patch_params,
                    spacing=spacing,
                    seg_mask_fp=seg_mask_fp,
                    enable_mp=True,
                    color_mapping=color_mapping,
                    filtering_threshold=cfg.filtering_threshold,
                    skip=cfg.skip_category,
                    sort=cfg.sort,
                    topk=cfg.topk,
                    alpha=cfg.alpha,
                    seg_mask_save_dir=seg_mask_save_dir,
                    overlay_mask_save_dir=overlay_mask_save_dir,
                    backend=cfg.backend,
                )
                if t_df is not None:
                    dfs.append(t_df)

                if cfg.wandb.enable:
                    wandb.log({"processed": i + 1})

            df = pd.concat(dfs, ignore_index=True)
            tiles_fp = Path(output_dir, f"sampled_patches.csv")
            df.to_csv(tiles_fp, index=False)


if __name__ == "__main__":

    main()
