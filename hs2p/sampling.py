import os
import tqdm
import argparse
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing as mp
from pathlib import Path

from hs2p.utils import setup, load_csv, fix_random_seeds
from hs2p.wsi import extract_coordinates, filter_coordinates, sample_coordinates, save_coordinates, visualize_coordinates, overlay_mask_on_slide, SamplingParameters


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("hs2p", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--skip-datetime", action="store_true", help="skip run id datetime prefix"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command using \"path.key=value\".",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def process_slide_wrapper(kwargs):
    return process_slide(**kwargs)


def process_slide(
    *,
    wsi_path: Path,
    mask_path: Path,
    cfg,
    mask_visualize_dir,
    sampling_visualize_dir,
    sampling_params: SamplingParameters,
    disable_tqdm: bool = False,
    num_workers: int = 4,
):
    """
    Process a single slide: sample tile coordinates and visualize if needed.
    """
    wsi_name = wsi_path.stem.replace(" ", "_")
    try:

        if cfg.visualize and sampling_visualize_dir is not None:
            preview_palette = np.zeros(shape=768, dtype=int)
            if sampling_params.color_mapping is None:
                ncat = len(sampling_params.pixel_mapping)
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
                    for i, k in enumerate(sampling_params.pixel_mapping.keys())
                }
            else:
                color_mapping = sampling_params.color_mapping
            p = [0] * 3 * len(color_mapping)
            for k, v in sampling_params.pixel_mapping.items():
                if color_mapping[k] is not None:
                    p[v * 3 : v * 3 + 3] = color_mapping[k]
            n = len(p)
            preview_palette[0:n] = np.array(p).astype(int)
        else:
            color_mapping = None
            preview_palette = None

        if not cfg.tiling.sampling_params.independant_sampling:
            tissue_mask_visu_path = None
            if cfg.visualize and mask_visualize_dir is not None:
                tissue_mask_visu_path = Path(mask_visualize_dir, f"{wsi_name}-tissue.png")
            coordinates, tile_level, resize_factor, tile_size_lv0 = extract_coordinates(
                wsi_path=wsi_path,
                mask_path=mask_path,
                backend=cfg.tiling.backend,
                tiling_params=cfg.tiling.params,
                segment_params=cfg.tiling.seg_params,
                filter_params=cfg.tiling.filter_params,
                sampling_params=sampling_params,
                mask_visu_path=tissue_mask_visu_path,
                disable_tqdm=disable_tqdm,
                num_workers=num_workers,
            )
            filtered_coordinates = filter_coordinates(
                wsi_path=wsi_path,
                mask_path=mask_path,
                backend=cfg.tiling.backend,
                coordinates=coordinates,
                tile_level=tile_level,
                segment_params=cfg.tiling.seg_params,
                tiling_params=cfg.tiling.params,
                sampling_params=sampling_params,
                disable_tqdm=disable_tqdm,
            ) # a dict mapping annotation -> coordinates
            for annotation, coordinates in filtered_coordinates.items():
                if len(coordinates) == 0:
                    continue
                coordinates_dir = Path(cfg.output_dir, "coordinates", annotation)
                coordinates_dir.mkdir(exist_ok=True, parents=True)
                coordinates_path = Path(coordinates_dir, f"{wsi_name}.npy")
                save_coordinates(
                    coordinates=coordinates,
                    target_spacing=cfg.tiling.params.spacing,
                    tile_level=tile_level,
                    tile_size=cfg.tiling.params.tile_size,
                    resize_factor=resize_factor,
                    tile_size_lv0=tile_size_lv0,
                    save_path=coordinates_path,
                )
                if cfg.visualize and sampling_visualize_dir is not None:
                    visualize_coordinates(
                        wsi_path=wsi_path,
                        coordinates=coordinates,
                        tile_size_lv0=tile_size_lv0,
                        save_dir=sampling_visualize_dir,
                        downsample=cfg.tiling.visu_params.downsample,
                        backend=cfg.tiling.backend,
                        mask_path=mask_path,
                        annotation=annotation,
                        palette=preview_palette,
                    )
        else:
            for annotation in sampling_params.pixel_mapping.keys():
                if sampling_params.tissue_percentage[annotation] is None:
                    continue
                annotation_mask_dir = mask_visualize_dir / annotation
                annotation_mask_dir.mkdir(exist_ok=True, parents=True)
                tissue_mask_visu_path = None
                if cfg.visualize and mask_visualize_dir is not None:
                    tissue_mask_visu_path = Path(annotation_mask_dir, f"{wsi_name}.jpg")
                coordinates, tile_level, resize_factor, tile_size_lv0 = sample_coordinates(
                    wsi_path=wsi_path,
                    mask_path=mask_path,
                    backend=cfg.tiling.backend,
                    tiling_params=cfg.tiling.params,
                    segment_params=cfg.tiling.seg_params,
                    filter_params=cfg.tiling.filter_params,
                    sampling_params=sampling_params,
                    annotation=annotation,
                    mask_visu_path=tissue_mask_visu_path,
                    disable_tqdm=disable_tqdm,
                    num_workers=num_workers,
                )
                if len(coordinates) == 0:
                    continue
                coordinates_dir = Path(cfg.output_dir, "coordinates", annotation)
                coordinates_dir.mkdir(exist_ok=True, parents=True)
                coordinates_path = Path(coordinates_dir, f"{wsi_name}.npy")
                save_coordinates(
                    coordinates=coordinates,
                    target_spacing=cfg.tiling.params.spacing,
                    tile_level=tile_level,
                    tile_size=cfg.tiling.params.tile_size,
                    resize_factor=resize_factor,
                    tile_size_lv0=tile_size_lv0,
                    save_path=coordinates_path,
                )
                if cfg.visualize and sampling_visualize_dir is not None:
                    visualize_coordinates(
                        wsi_path=wsi_path,
                        coordinates=coordinates,
                        tile_size_lv0=tile_size_lv0,
                        save_dir=sampling_visualize_dir,
                        downsample=cfg.tiling.visu_params.downsample,
                        backend=cfg.tiling.backend,
                        mask_path=mask_path,
                        annotation=annotation,
                        palette=preview_palette,
                    )
        if cfg.visualize and mask_visualize_dir is not None:
            mask_visu_path = Path(mask_visualize_dir, f"{wsi_name}.png")
            overlay_mask = overlay_mask_on_slide(
                wsi_path=wsi_path,
                annotation_mask_path=mask_path,
                downsample=cfg.tiling.visu_params.downsample,
                palette=preview_palette,
                pixel_mapping=sampling_params.pixel_mapping,
                color_mapping=color_mapping,
            )
            overlay_mask.save(mask_visu_path)
        return str(wsi_path), {"status": "success"}

    except Exception as e:
        return str(wsi_path), {
            "status": "failed",
            "error": str(e),
            "traceback": str(traceback.format_exc()),
        }


def main(args):
    
    cfg = setup(args)
    output_dir = Path(cfg.output_dir)

    fix_random_seeds(cfg.seed)

    wsi_paths, mask_paths = load_csv(cfg)

    parallel_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        parallel_workers = min(
            parallel_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        )

    process_list = output_dir / "process_list.csv"
    if process_list.is_file() and cfg.resume:
        process_df = pd.read_csv(process_list)
    else:
        data = {
            "wsi_name": [p.stem for p in wsi_paths],
            "wsi_path": [str(p) for p in wsi_paths],
            "mask_path": [str(p) if p is not None else p for p in mask_paths],
            "sampling_status": ["tbp"] * len(wsi_paths),
            "error": [str(np.nan)] * len(wsi_paths),
            "traceback": [str(np.nan)] * len(wsi_paths),
        }
        process_df = pd.DataFrame(data)

    skip_sampling = process_df["sampling_status"].str.contains("success").all()

    pixel_mapping = {k: v for e in cfg.tiling.sampling_params.pixel_mapping for k, v in e.items()}
    tissue_percentage = {k: v for e in cfg.tiling.sampling_params.tissue_percentage for k, v in e.items()}
    if "tissue" not in tissue_percentage:
        tissue_percentage["tissue"] = cfg.tiling.params.min_tissue_percentage
    if cfg.tiling.sampling_params.color_mapping is not None:
        color_mapping = {k: v for e in cfg.tiling.sampling_params.color_mapping for k, v in e.items()}
    else:
        color_mapping = None
    
    sampling_params = SamplingParameters(
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        tissue_percentage=tissue_percentage,
    )

    if not skip_sampling:

        mask = process_df["sampling_status"] != "success"
        process_stack = process_df[mask]
        total = len(process_stack)

        wsi_paths_to_process = [
            Path(x) for x in process_stack.wsi_path.values.tolist()
        ]
        mask_paths_to_process = [
            Path(x) if x is not None else x
            for x in process_stack.mask_path.values.tolist()
        ]

        # setup directories for coordinates and visualization
        coordinates_dir = output_dir / "coordinates"
        coordinates_dir.mkdir(exist_ok=True, parents=True)
        mask_visualize_dir = None
        sampling_visualize_dir = None
        if cfg.visualize:
            visualize_dir = output_dir / "visualization"
            mask_visualize_dir = Path(visualize_dir, "mask")
            sampling_visualize_dir = Path(visualize_dir, "sampling")
            mask_visualize_dir.mkdir(exist_ok=True, parents=True)
            sampling_visualize_dir.mkdir(exist_ok=True, parents=True)

        sampling_updates = {}
        with mp.Pool(processes=parallel_workers) as pool:
            args_list = [
                {
                    "wsi_path": wsi_fp,
                    "mask_path": mask_fp,
                    "cfg": cfg,
                    "mask_visualize_dir": mask_visualize_dir,
                    "sampling_visualize_dir": sampling_visualize_dir,
                    "sampling_params": sampling_params,
                    "disable_tqdm": True,
                    "num_workers": parallel_workers,
                }
                for wsi_fp, mask_fp in zip(
                    wsi_paths_to_process, mask_paths_to_process
                )
            ]
            results = list(
                tqdm.tqdm(
                    pool.imap(process_slide_wrapper, args_list),
                    total=total,
                    desc="Slide sampling",
                    unit="slide",
                    leave=True,
                )
            )
        for wsi_path, status_info in results:
            sampling_updates[wsi_path] = status_info

        for wsi_path, status_info in sampling_updates.items():
            process_df.loc[
                process_df["wsi_path"] == wsi_path, "sampling_status"
            ] = status_info["status"]
            if "error" in status_info:
                process_df.loc[
                    process_df["wsi_path"] == wsi_path, "error"
                ] = status_info["error"]
                process_df.loc[
                    process_df["wsi_path"] == wsi_path, "traceback"
                ] = status_info["traceback"]
        process_df.to_csv(process_list, index=False)

        # summary logging
        total_slides = len(process_df)
        failed_sampling = process_df[process_df["sampling_status"] == "failed"]
        print("=+=" * 10)
        print(f"Total number of slides: {total_slides}")
        print(f"Failed sampling: {len(failed_sampling)}")
        for annotation, pct in tissue_percentage.items():
            if pct is None:
                continue
            slides_with_tiles = [
                str(p)
                for p in wsi_paths
                if Path(coordinates_dir, annotation, f"{p.stem}.npy").is_file()
            ]
            no_tiles = process_df[~process_df["wsi_path"].isin(slides_with_tiles)]
            print(f"No {annotation} tiles after sampling step: {len(no_tiles)}")
        print("=+=" * 10)

    else:
        print("=+=" * 10)
        print("All slides have been sampled. Skipping sampling step.")
        print("=+=" * 10)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
