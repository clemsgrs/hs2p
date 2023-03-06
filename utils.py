import time
import tqdm
import wandb
import subprocess
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, Tuple

from source.wsi import WholeSlideImage
from source.utils import VisualizeCoords, compute_time, initialize_df


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
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.username,
        name=cfg.wandb.exp_name,
        group=cfg.wandb.group,
        dir=cfg.wandb.dir,
        config=config,
        tags=cfg.wandb.tags,
        resume="allow",
    )
    config_file_path = Path(run.dir, "run_config.yaml")
    d = OmegaConf.to_container(cfg, resolve=True)
    with open(config_file_path, "w+") as f:
        write_dictconfig(d, f)
        wandb.save(str(config_file_path))
        f.close()
    return run


def segment(
    wsi_object: WholeSlideImage,
    spacing: float,
    seg_params: DictConfig,
    filter_params: DictConfig,
    mask_fp: Optional[Path] = None,
):
    start_time = time.time()
    if mask_fp is not None:
        # print(f"Loading segmentation mask at: {mask_fp}")
        seg_level = wsi_object.loadSegmentation(
            mask_fp,
            spacing=spacing,
            filter_params=filter_params,
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
    patch_format: str = "png",
    top_left: Optional[List[int]] = None,
    bot_right: Optional[List[int]] = None,
    enable_mp: bool = True,
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
        patch_format=patch_format,
        top_left=top_left,
        bot_right=bot_right,
        enable_mp=enable_mp,
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
    verbose: bool = False,
):
    start = time.time()
    heatmap = VisualizeCoords(
        file_path,
        wsi_object,
        downscale=downscale,
        bg_color=bg_color,
        alpha=-1,
        draw_grid=draw_grid,
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
    verbose: bool = False,
    log_to_wandb: bool = False,
):
    start_time = time.time()
    slide_paths = slide_df.slide_path.values.tolist()
    mask_paths = []
    if "mask_path" in slide_df.columns:
        mask_paths = slide_df.mask_path.values.tolist()

    if process_list is None:
        df = initialize_df(
            slide_paths, mask_paths, seg_params, filter_params, vis_params, patch_params
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
        desc=(f"Seg&Patch"),
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
            if "mask_path" in process_stack.columns:
                mask_path = Path(process_stack.loc[idx, "mask_path"])
            t.display(f"Processing {slide_id}", pos=2)

            # Inialize WSI
            wsi_object = WholeSlideImage(slide_path)

            vis_level = vis_params.vis_level
            if vis_level < 0:
                if len(wsi_object.level_dim) == 1:
                    vis_params.vis_level = 0
                else:
                    best_level = wsi_object.get_best_level_for_downsample_custom(
                        vis_params.downsample
                    )
                    vis_params.vis_level = best_level

            seg_level = seg_params.seg_level
            if seg_level < 0:
                if len(wsi_object.level_dim) == 1:
                    seg_params.seg_level = 0
                else:
                    best_level = wsi_object.get_best_level_for_downsample_custom(
                        seg_params.downsample
                    )
                    seg_params.seg_level = best_level

            w, h = wsi_object.level_dim[seg_params.seg_level]
            if w * h > 1e8:
                print(
                    f"level_dim {w} x {h} is likely too large for successful segmentation, aborting"
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
                mask = wsi_object.visWSI(
                    vis_level=vis_params.vis_level,
                    line_thickness=vis_params.line_thickness,
                )
                mask_path = Path(mask_save_dir, f"{slide_id}.jpg")
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
                    patch_format=patch_params.format,
                    enable_mp=True,
                    verbose=verbose,
                )
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
    visu: bool = False,
    patch: bool = False,
    verbose: bool = False,
):
    if verbose:
        print(f"Processing {slide_id}...")

    if mask_fp is not None:
        mask_fp = Path(mask_fp)

    # Inialize WSI
    wsi_object = WholeSlideImage(Path(slide_fp))

    vis_level = vis_params.vis_level
    if vis_level < 0:
        if len(wsi_object.level_dim) == 1:
            vis_params.vis_level = 0
            best_vis_level = 0
        else:
            best_vis_level = wsi_object.get_best_level_for_downsample_custom(
                vis_params.downsample
            )
            vis_params.vis_level = best_vis_level

    seg_level = seg_params.seg_level
    if seg_level < 0:
        if len(wsi_object.level_dim) == 1:
            seg_params.seg_level = 0
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
        status = "failed_seg"
        tile_df = pd.DataFrame.from_dict(
            {
                "slide_id": [],
                "tile_size": [],
                "spacing": [],
                "level": [],
                "level_dim": [],
                "x": [],
                "y": [],
                "contour": [],
            }
        )
        return tile_df, slide_id, status, best_vis_level, best_seg_level

    seg_time = -1
    wsi_object, seg_time = segment(
        wsi_object,
        patch_params.spacing,
        seg_params,
        filter_params,
        mask_fp,
    )

    if seg_params.save_mask:
        mask = wsi_object.visWSI(
            vis_level=vis_params.vis_level,
            line_thickness=vis_params.line_thickness,
        )
        mask_path = Path(mask_save_dir, f"{slide_id}.jpg")
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
            patch_format=patch_params.format,
            enable_mp=False,
            verbose=verbose,
        )

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
                verbose=verbose,
            )
            visu_path = Path(visu_save_dir, f"{slide_id}.jpg")
            heatmap.save(visu_path)

    status = "processed"

    # restore original values
    vis_params.vis_level = vis_level
    seg_params.seg_level = seg_level

    return tile_df, slide_id, status, best_vis_level, best_seg_level
