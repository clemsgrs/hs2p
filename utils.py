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
    tags: Optional[List] = None,
    key: Optional[str] = "",
):
    command = f"wandb login {key}"
    subprocess.call(command, shell=True)
    if tags == None:
        tags = []
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.username,
        name=cfg.wandb.exp_name,
        group=cfg.wandb.group,
        dir=cfg.wandb.dir,
        config=config,
        tags=tags,
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
    mask_file: Optional[Path] = None,
):
    start_time = time.time()
    if mask_file is not None:
        wsi_object.initSegmentation(mask_file)
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
    verbose: bool = False,
):

    start_time = time.time()
    file_path = wsi_object.process_contours(
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
        verbose=verbose,
    )
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


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
    slide_list: Optional[List[str]] = None,
    seg: bool = False,
    visu: bool = False,
    patch: bool = False,
    process_list: Optional[Path] = None,
    verbose: bool = False,
    log_to_wandb: bool = False
):
    start_time = time.time()
    with open(slide_list, "r") as f:
        slide_paths = sorted([s.strip() for s in f])

    if process_list is None:
        df = initialize_df(
            slide_paths, seg_params, filter_params, vis_params, patch_params
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

            df.loc[idx, "vis_level"] = vis_params.vis_level
            df.loc[idx, "seg_level"] = seg_params.seg_level

            seg_time_elapsed = -1
            if seg:
                wsi_object, seg_time_elapsed = segment(
                    wsi_object,
                    patch_params.spacing,
                    seg_params,
                    filter_params,
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
                slide_save_dir = Path(
                    patch_save_dir, slide_id, f"{patch_params.patch_size}"
                )
                file_path, patch_time_elapsed = patching(
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
                    verbose=verbose,
                )

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

    df.to_csv(Path(output_dir, "process_list.csv"), index=False)
    print(f"\n" * 4)
    print("-" * 7, "summary", "-" * 7)
    print(f"total time taken: \t{mins}m{secs}s")
    print(f"average segmentation time per slide: \t{seg_times:.2f}s")
    print(f"average patching time per slide: \t{patch_times:.2f}s")
    print(f"average stiching time per slide: \t{visu_times:.2f}s")
    print("-" * 7, "-" * (len("summary") + 2), "-" * 7, sep="")

    return seg_times, patch_times
