import os
import re
import PIL
import tqdm
import time
import wandb
import hydra
import pyvips
import datetime
import subprocess
import pandas as pd
import numpy as np
import seaborn as sns
import multiprocessing as mp

from PIL import Image, ImageOps
from pathlib import Path
from typing import Dict, List, Optional
from omegaconf import DictConfig

from utils import initialize_wandb, segment
from source.wsi import WholeSlideImage
from source.utils import get_mode


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
    w, h = mask_arr.shape
    binary_mask = mask_arr == val
    mask_percentage = np.sum(binary_mask) / (w * h)
    return mask_percentage


def overlay_mask_on_slide(
    wsi_object,
    mask_object,
    vis_params,
    pixel_mapping: Dict[str, int],
    color_mapping: Optional[Dict[str, List[int]]] = None,
    alpha: float = 0.5,
):
    """
    Show a mask overlayed on a slide
    """

    x_mask = mask_object.level_dimensions[0][0]
    mask_min_level = np.argmin(
        [abs(x_wsi - x_mask) for x_wsi, _ in wsi_object.level_dimensions]
    )

    if vis_params.vis_level < 0:
        if len(wsi_object.level_dimensions) == 1:
            vis_level = 0
            assert mask_min_level == 0
        else:
            vis_level = wsi_object.get_best_level_for_downsample(vis_params.downsample)
            vis_level = max(vis_level, mask_min_level)
    else:
        vis_level = max(vis_params.vis_level, mask_min_level)

    mask_vis_level = vis_level - mask_min_level

    s = wsi_object.open_page(vis_level)
    region = pyvips.Region.new(s).fetch(0, 0, s.width, s.height)
    slide = Image.frombuffer(
        mode=wsi_object.mode, size=(s.width, s.height), data=region
    ).convert("RGBA")

    m = mask_object.open_page(mask_vis_level)
    region = pyvips.Region.new(m).fetch(0, 0, m.width, m.height)
    mask = Image.frombuffer(
        mode=mask_object.mode, size=(m.width, m.height), data=region
    )
    # Mask data is present in the R channel
    mask = mask.split()[0]

    # Create alpha mask
    mask_arr = np.array(mask)
    alpha_int = int(round(255 * alpha))
    if color_mapping is not None:
        alpha_content = np.zeros_like(mask_arr)
        for k, v in pixel_mapping.items():
            if color_mapping[k] is not None:
                alpha_content += mask_arr == v
        alpha_content = np.less(alpha_content, 1).astype("uint8") * alpha_int + (
            255 - alpha_int
        )
    else:
        alpha_content = np.less_equal(mask_arr, 0).astype("uint8") * alpha_int + (
            255 - alpha_int
        )
    alpha_content = Image.fromarray(alpha_content)

    preview_palette = np.zeros(shape=768, dtype=int)

    if color_mapping is None:
        ncat = len(pixel_mapping)
        if ncat <= 10:
            color_palette = sns.color_palette("tab10")[:ncat]
        elif ncat <= 20:
            color_palette = sns.color_palette("tab20")[:ncat]
        else:
            raise ValueError(
                f"Implementation supports up to 20 categories (provided pixel_mapping has {ncat})"
            )
        color_mapping = {
            k: tuple(255 * x for x in color_palette[i])
            for i, k in enumerate(pixel_mapping.keys())
        }

    p = [0] * 3 * len(color_mapping)
    for k, v in pixel_mapping.items():
        if color_mapping[k] is not None:
            p[v * 3 : v * 3 + 3] = color_mapping[k]
    n = len(p)
    preview_palette[0:n] = np.array(p).astype(int)

    mask.putpalette(data=preview_palette.tolist())
    mask_rgb = mask.convert(mode="RGB")

    overlayed_image = PIL.Image.composite(
        image1=slide, image2=mask_rgb, mask=alpha_content
    )
    return overlayed_image


def overlay_mask_on_tile(
    tile: PIL.Image,
    mask: PIL.Image,
    pixel_mapping: Dict[str, int],
    color_mapping: Optional[Dict[str, List[int]]] = None,
    alpha=0.5,
):

    # Create alpha mask
    mask_arr = np.array(mask)
    alpha_int = int(round(255 * alpha))
    if color_mapping is not None:
        alpha_content = np.zeros_like(mask_arr)
        for k, v in pixel_mapping.items():
            if color_mapping[k] is not None:
                alpha_content += mask_arr == v
        alpha_content = np.less(alpha_content, 1).astype("uint8") * alpha_int + (
            255 - alpha_int
        )
    else:
        alpha_content = np.less_equal(mask_arr, 0).astype("uint8") * alpha_int + (
            255 - alpha_int
        )
    alpha_content = Image.fromarray(alpha_content)

    preview_palette = np.zeros(shape=768, dtype=int)

    if color_mapping is None:
        ncat = len(pixel_mapping)
        if ncat <= 10:
            color_palette = sns.color_palette("tab10")[:ncat]
        elif ncat <= 20:
            color_palette = sns.color_palette("tab20")[:ncat]
        else:
            raise ValueError(
                f"Implementation supports up to 20 categories (provided pixel_mapping has {ncat})"
            )
        color_mapping = {
            k: tuple(255 * x for x in color_palette[i])
            for i, k in enumerate(pixel_mapping.keys())
        }

    p = [0] * 3 * len(color_mapping)
    for k, v in pixel_mapping.items():
        if color_mapping[k] is not None:
            p[v * 3 : v * 3 + 3] = color_mapping[k]
    n = len(p)
    preview_palette[0:n] = np.array(p).astype(int)

    mask.putpalette(data=preview_palette.tolist())
    mask_rgb = mask.convert(mode="RGB")

    overlayed_image = PIL.Image.composite(
        image1=tile, image2=mask_rgb, mask=alpha_content
    )
    return overlayed_image


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

    wsi_spacing_level = wsi_object.get_best_level_for_spacing(spacing)
    x_mask = mask_object.level_dimensions[0][0]
    mask_min_level = np.argmin(
        [abs(x_wsi - x_mask) for x_wsi, _ in wsi_object.level_dimensions]
    )
    if downsample == -1:
        wsi_downsample_level = max(wsi_spacing_level, mask_min_level)
    else:
        downsample_level = wsi_object.get_best_level_for_downsample(downsample)
        wsi_downsample_level = max(downsample_level, mask_min_level)
    assert wsi_spacing_level <= wsi_downsample_level
    mask_downsample_level = wsi_downsample_level - mask_min_level

    wsi_scale = tuple(
        np.array(wsi_object.level_downsamples[wsi_downsample_level])
        / np.array(wsi_object.level_downsamples[wsi_spacing_level])
    )
    downsampled_tile_size_x, downsampled_tile_size_y = int(
        tile_size * 1.0 / wsi_scale[0]
    ), int(tile_size * 1.0 / wsi_scale[1])

    mask = pyvips.Image.new_from_file(str(mask_object.path), page=mask_downsample_level)
    region = pyvips.Region.new(mask).fetch(0, 0, mask.width, mask.height)
    mask_data = Image.frombuffer(
        mode=mask_object.mode, size=(mask.width, mask.height), data=region
    )
    mask_data = mask_data.split()[0]

    filtered_grid = []
    tile_percentages = []

    if tile_df is not None:

        # in tile_df, x & y coordinates are defined w.r.t level 0 in the slide
        # need to downsample them to match input downsample value
        for x, y in tile_df[["x", "y"]].values:
            scale = wsi_object.level_downsamples[wsi_downsample_level]
            x_downsampled, y_downsampled = int(x * 1.0 / scale[0]), int(
                y * 1.0 / scale[1]
            )
            # crop mask tile
            coords = (
                x_downsampled,
                y_downsampled,
                x_downsampled + downsampled_tile_size_x,
                y_downsampled + downsampled_tile_size_y,
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
):

    # Inialize WSI & annotation mask
    wsi_object = WholeSlideImage(slide_fp, spacing)
    annotation_mask = WholeSlideImage(annot_mask_fp)
    wsi_spacing_level = wsi_object.get_best_level_for_spacing(patch_params.spacing)

    x_mask = annotation_mask.level_dimensions[0][0]
    mask_min_level = np.argmin(
        [abs(x_wsi - x_mask) for x_wsi, _ in wsi_object.level_dimensions]
    )
    mask_spacing_level = annotation_mask.get_best_level_for_spacing(
        patch_params.spacing, ignore_warning=True
    )

    vis_level = vis_params.vis_level
    if vis_level < 0:
        if len(wsi_object.level_dimensions) == 1:
            best_vis_level = 0
        else:
            best_vis_level = wsi_object.get_best_level_for_downsample(
                vis_params.downsample
            )
        vis_params.vis_level = best_vis_level

    seg_level = seg_params.seg_level
    if seg_level < 0:
        if len(wsi_object.level_dimensions) == 1:
            best_seg_level = 0
        else:
            best_seg_level = wsi_object.get_best_level_for_downsample(
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
    raw_tile_dir = Path(output_dir, "patches", "raw")
    overlay_tile_dir = Path(output_dir, "patches", "mask")

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

                        wsi = wsi_object.open_page(wsi_spacing_level)
                        # need to move x, y from level 0 to wsi_spacing_level
                        wsi_scale = tuple(
                            np.array(wsi_object.level_downsamples[wsi_spacing_level])
                            / np.array(wsi_object.level_downsamples[0])
                        )
                        x_scaled, y_scaled = int(x * 1.0 / wsi_scale[0]), int(
                            y * 1.0 / wsi_scale[1]
                        )
                        ps_x, ps_y = min(
                            patch_params.patch_size, wsi.width - x_scaled
                        ), min(patch_params.patch_size, wsi.height - y_scaled)
                        region = pyvips.Region.new(wsi).fetch(
                            x_scaled, y_scaled, ps_x, ps_y
                        )
                        tile = Image.frombuffer(
                            mode=wsi_object.mode, size=(ps_x, ps_y), data=region
                        ).convert("RGB")
                        if (
                            ps_x != patch_params.patch_size
                            or ps_y != patch_params.patch_size
                        ):
                            tile = ImageOps.pad(
                                tile,
                                (patch_params.patch_size, patch_params.patch_size),
                                color=None,
                                centering=(0, 0),
                            )

                        fname = f"{slide_id}_{x}_{y}"
                        tile_fp = Path(
                            raw_tile_dir, cat_, f"{fname}.{patch_params.fmt}"
                        )
                        tile_fp.parent.mkdir(exist_ok=True, parents=True)
                        tile.save(tile_fp)

                        if vis_params.overlay_mask_on_patch:
                            # need to scale x, y from slide level 0 to mask_spacing_level referential
                            mask_scale = tuple(
                                np.array(
                                    wsi_object.level_downsamples[
                                        mask_min_level + mask_spacing_level
                                    ]
                                )
                                / np.array(wsi_object.level_downsamples[0])
                            )
                            x_scaled, y_scaled = int(x * 1.0 / mask_scale[0]), int(
                                y * 1.0 / mask_scale[1]
                            )
                            # need to scale tile size from wsi_spacing_level to mask_spacing_level
                            ts_scale = tuple(
                                np.array(
                                    wsi_object.level_downsamples[
                                        mask_min_level + mask_spacing_level
                                    ]
                                )
                                / np.array(
                                    wsi_object.level_downsamples[wsi_spacing_level]
                                )
                            )
                            ts_x, ts_y = int(
                                patch_params.patch_size * 1.0 / ts_scale[0]
                            ), int(patch_params.patch_size * 1.0 / ts_scale[1])
                            # read annotation tile from mask
                            mask = pyvips.Image.new_from_file(
                                str(annotation_mask.path), page=mask_spacing_level
                            )
                            ps_x, ps_y = min(ts_x, mask.width - x_scaled), min(
                                ts_y, mask.height - y_scaled
                            )
                            region = pyvips.Region.new(mask).fetch(
                                x_scaled, y_scaled, ps_x, ps_y
                            )
                            mode = get_mode(mask.bands)
                            masked_tile = Image.frombuffer(
                                mode=mode, size=(ps_x, ps_y), data=region
                            )
                            masked_tile = masked_tile.split()[0]
                            if ps_x != ts_x or ps_y != ts_y:
                                masked_tile = ImageOps.pad(
                                    masked_tile,
                                    (ts_x, ts_y),
                                    color=None,
                                    centering=(0, 0),
                                )
                            if ts_scale[0] > 1 or ts_scale[1] > 1:
                                # 2 possible ways to go:
                                # - upsample annotation tile to match true tile size
                                # - read tile from slide to match annotation tile size (end up with images smaller than input patch size)
                                # option 1
                                masked_tile = masked_tile.resize(
                                    tuple(
                                        int(e * ts_scale[i])
                                        for i, e in enumerate(masked_tile.size)
                                    ),
                                    Image.NEAREST,
                                )
                                # option 2
                                # wsi = wsi_object.open_page(mask_min_level+mask_spacing_level)
                                # r = pyvips.Region.new(wsi).fetch(x_scaled, y_scaled, ps_x, ps_y)
                                # tile = Image.frombuffer(mode=wsi_object.mode, size=(ps_x,ps_y), data=r).convert("RGB")
                                # if ps_x != ts_x or ps_y != ts_y:
                                #     tile = ImageOps.pad(tile, (ts_x,ts_y), color=None, centering=(0, 0))

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

    if vis_params.overlay_mask_on_slide:
        overlay_mask = overlay_mask_on_slide(
            wsi_object,
            annotation_mask,
            vis_params,
            pixel_mapping,
            color_mapping,
            alpha=alpha,
        )
        overlay_mask_path = Path(overlay_mask_save_dir, f"{slide_id}.jpg")
        overlay_mask.save(overlay_mask_path)

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

    # restore original values
    vis_params.vis_level = vis_level
    seg_params.seg_level = seg_level

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
        seg_mask_fps = df["segmentation_mask_path"].values.tolist()
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
            )
            for sid, slide_fp, seg_mask_fp, annot_mask_fp, spacing in zip(
                slide_ids, slide_fps, seg_mask_fps, annot_mask_fps, spacings
            )
        ]

        command_line = [
            "python3",
            "log_nproc.py",
            "--output_dir",
            f"{overlay_mask_save_dir}",
            "--fmt",
            "jpg",
            "--total",
            f"{len(slide_ids)}",
        ]
        if cfg.wandb.enable:
            command_line = command_line + ["--log_to_wandb", "--id", f"{run_id}"]
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
