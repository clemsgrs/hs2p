import os
import tqdm
import argparse
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing as mp
from pathlib import Path
from collections.abc import Sequence

from hs2p.api import (
    FilterConfig,
    SegmentationConfig,
    TilingConfig,
    TilingResult,
    _build_cli_configs,
    _validate_required_columns,
    compute_config_hash,
    save_tiling_result,
)
from hs2p.utils import setup, load_csv, fix_random_seeds
from hs2p.wsi import (
    extract_coordinates,
    filter_coordinates,
    sample_coordinates,
    visualize_coordinates,
    SamplingParameters,
)


def _validate_visualization_color_mapping(
    *,
    pixel_mapping: dict[str, int],
    color_mapping: dict[str, Sequence[int] | None],
):
    missing_annotations = sorted(set(pixel_mapping.keys()) - set(color_mapping.keys()))
    if missing_annotations:
        raise ValueError(
            "color_mapping is missing annotation keys required by pixel_mapping: "
            + ", ".join(missing_annotations)
        )

    for annotation, color in color_mapping.items():
        if color is None:
            continue
        if isinstance(color, (str, bytes)):
            raise ValueError(
                f"color_mapping['{annotation}'] must be None or a length-3 RGB sequence"
            )
        if not isinstance(color, Sequence) or len(color) != 3:
            raise ValueError(
                f"color_mapping['{annotation}'] must be None or a length-3 RGB sequence"
            )
        if any(
            (not isinstance(c, (int, np.integer)) or c < 0 or c > 255) for c in color
        ):
            raise ValueError(
                f"color_mapping['{annotation}'] must contain integers in [0, 255]"
            )


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("hs2p", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--skip-datetime", action="store_true", help="skip run id datetime prefix"
    )
    parser.add_argument(
        "--skip-logging", action="store_true", help="skip logging configuration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "opts",
        help='Modify config options at the end of the command using "path.key=value".',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def process_slide_wrapper(kwargs):
    return process_slide(**kwargs)


def _resolve_inner_workers(cfg, parallel_workers: int) -> int:
    inner_workers = getattr(cfg.speed, "inner_workers", 1)
    inner_workers = int(inner_workers)
    if inner_workers < 1:
        raise ValueError("cfg.speed.inner_workers must be >= 1")
    return min(inner_workers, parallel_workers)


def _save_sampling_coordinates(
    *,
    sample_id: str,
    image_path: Path,
    mask_path: Path | None,
    backend: str,
    cfg,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
    annotation: str,
    coordinates: list[tuple[int, int]],
    extraction,
    sampling_params: SamplingParameters,
):
    x = np.array([x for x, _ in coordinates], dtype=np.int64)
    y = np.array([y for _, y in coordinates], dtype=np.int64)
    result = TilingResult(
        sample_id=sample_id,
        image_path=image_path,
        mask_path=mask_path,
        backend=backend,
        x=x,
        y=y,
        tile_index=np.arange(len(coordinates), dtype=np.int32),
        tissue_fraction=None,
        target_spacing_um=tiling_config.target_spacing_um,
        target_tile_size_px=tiling_config.target_tile_size_px,
        read_level=extraction.read_level,
        read_spacing_um=extraction.read_spacing_um,
        read_tile_size_px=extraction.read_tile_size_px,
        tile_size_lv0=extraction.tile_size_lv0,
        overlap=tiling_config.overlap,
        tissue_threshold=tiling_config.tissue_threshold,
        num_tiles=len(coordinates),
        config_hash=compute_config_hash(
            tiling=tiling_config,
            segmentation=segmentation_config,
            filtering=filter_config,
            extra={
                "annotation": annotation,
                "sampling": {
                    "pixel_mapping": sampling_params.pixel_mapping,
                    "tissue_percentage": sampling_params.tissue_percentage,
                },
            },
        ),
    )
    annotation_dir = Path(cfg.output_dir, "coordinates", annotation)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    return save_tiling_result(result, output_dir=cfg.output_dir, coordinates_dir=annotation_dir)


def process_slide(
    *,
    sample_id: str,
    wsi_path: Path,
    mask_path: Path | None,
    cfg,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
    mask_visualize_dir,
    sampling_visualize_dir,
    sampling_params: SamplingParameters,
    disable_tqdm: bool = False,
    num_workers: int = 4,
):
    """
    Process a single slide: sample tile coordinates and visualize if needed.
    """
    wsi_name = sample_id
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
            _validate_visualization_color_mapping(
                pixel_mapping=sampling_params.pixel_mapping,
                color_mapping=color_mapping,
            )
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
                tissue_mask_visu_path = Path(mask_visualize_dir, f"{wsi_name}.png")
            extraction = extract_coordinates(
                wsi_path=wsi_path,
                mask_path=mask_path,
                backend=cfg.tiling.backend,
                tiling_params=tiling_config,
                segment_params=segmentation_config,
                filter_params=filter_config,
                sampling_params=sampling_params,
                mask_visu_path=tissue_mask_visu_path,
                preview_downsample=cfg.tiling.visu_params.downsample,
                preview_palette=preview_palette,
                preview_pixel_mapping=sampling_params.pixel_mapping,
                preview_color_mapping=color_mapping,
                disable_tqdm=disable_tqdm,
                num_workers=num_workers,
            )
            filtered_coordinates, filtered_contour_indices = filter_coordinates(
                wsi_path=wsi_path,
                mask_path=mask_path,
                backend=cfg.tiling.backend,
                coordinates=extraction.coordinates,
                contour_indices=extraction.contour_indices,
                tile_level=extraction.read_level,
                segment_params=segmentation_config,
                tiling_params=tiling_config,
                sampling_params=sampling_params,
                disable_tqdm=disable_tqdm,
            )  # a dict mapping annotation -> coordinates
            for annotation, coordinates in filtered_coordinates.items():
                if len(coordinates) == 0:
                    continue
                _save_sampling_coordinates(
                    sample_id=sample_id,
                    image_path=wsi_path,
                    mask_path=mask_path,
                    backend=cfg.tiling.backend,
                    cfg=cfg,
                    tiling_config=tiling_config,
                    segmentation_config=segmentation_config,
                    filter_config=filter_config,
                    annotation=annotation,
                    coordinates=coordinates,
                    extraction=extraction,
                    sampling_params=sampling_params,
                )
                if cfg.visualize and sampling_visualize_dir is not None:
                    visualize_coordinates(
                        wsi_path=wsi_path,
                        coordinates=coordinates,
                        tile_size_lv0=extraction.tile_size_lv0,
                        save_dir=sampling_visualize_dir,
                        sample_id=sample_id,
                        downsample=cfg.tiling.visu_params.downsample,
                        backend=cfg.tiling.backend,
                        mask_path=mask_path,
                        annotation=annotation,
                        palette=preview_palette,
                        pixel_mapping=sampling_params.pixel_mapping,
                        color_mapping=color_mapping,
                    )
        else:
            for annotation in sampling_params.pixel_mapping.keys():
                if sampling_params.tissue_percentage[annotation] is None:
                    continue
                tissue_mask_visu_path = None
                if cfg.visualize and mask_visualize_dir is not None:
                    tissue_mask_visu_path = mask_visualize_dir / annotation / f"{wsi_name}.jpg"
                extraction = sample_coordinates(
                    wsi_path=wsi_path,
                    mask_path=mask_path,
                    backend=cfg.tiling.backend,
                    tiling_params=tiling_config,
                    segment_params=segmentation_config,
                    filter_params=filter_config,
                    sampling_params=sampling_params,
                    annotation=annotation,
                    mask_visu_path=tissue_mask_visu_path,
                    preview_downsample=cfg.tiling.visu_params.downsample,
                    disable_tqdm=disable_tqdm,
                    num_workers=num_workers,
                )
                coordinates = extraction.coordinates
                if len(coordinates) == 0:
                    continue
                _save_sampling_coordinates(
                    sample_id=sample_id,
                    image_path=wsi_path,
                    mask_path=mask_path,
                    backend=cfg.tiling.backend,
                    cfg=cfg,
                    tiling_config=tiling_config,
                    segmentation_config=segmentation_config,
                    filter_config=filter_config,
                    annotation=annotation,
                    coordinates=coordinates,
                    extraction=extraction,
                    sampling_params=sampling_params,
                )
                if cfg.visualize and sampling_visualize_dir is not None:
                    visualize_coordinates(
                        wsi_path=wsi_path,
                        coordinates=coordinates,
                        tile_size_lv0=extraction.tile_size_lv0,
                        save_dir=sampling_visualize_dir,
                        sample_id=sample_id,
                        downsample=cfg.tiling.visu_params.downsample,
                        backend=cfg.tiling.backend,
                        mask_path=mask_path,
                        annotation=annotation,
                        palette=preview_palette,
                        pixel_mapping=sampling_params.pixel_mapping,
                        color_mapping=color_mapping,
                    )
        return sample_id, {"status": "success"}

    except Exception as e:
        return sample_id, {
            "status": "failed",
            "error": str(e),
            "traceback": str(traceback.format_exc()),
        }


def main(args):

    cfg = setup(args)
    output_dir = Path(cfg.output_dir)

    fix_random_seeds(cfg.seed)

    whole_slides = load_csv(cfg)

    parallel_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        parallel_workers = min(
            parallel_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        )
    inner_workers = _resolve_inner_workers(cfg, parallel_workers)

    process_list = output_dir / "process_list.csv"
    if process_list.is_file() and cfg.resume:
        process_df = pd.read_csv(process_list)
        _validate_required_columns(
            process_df,
            required_columns={
                "sample_id",
                "image_path",
                "mask_path",
                "sampling_status",
                "error",
                "traceback",
            },
            file_path=process_list,
            file_label="sampling process_list.csv",
        )
        process_df["mask_path"] = process_df["mask_path"].apply(
            lambda x: str(x) if pd.notna(x) else None
        )
    else:
        data = {
            "sample_id": [slide.sample_id for slide in whole_slides],
            "image_path": [str(slide.image_path) for slide in whole_slides],
            "mask_path": [str(slide.mask_path) if slide.mask_path is not None else slide.mask_path for slide in whole_slides],
            "sampling_status": ["tbp"] * len(whole_slides),
            "error": [str(np.nan)] * len(whole_slides),
            "traceback": [str(np.nan)] * len(whole_slides),
        }
        process_df = pd.DataFrame(data)

    skip_sampling = process_df.empty or process_df["sampling_status"].fillna("").astype(str).str.contains("success").all()

    pixel_mapping = {
        k: v for e in cfg.tiling.sampling_params.pixel_mapping for k, v in e.items()
    }
    tissue_percentage = {
        k: v for e in cfg.tiling.sampling_params.tissue_percentage for k, v in e.items()
    }
    tissue_key_present = True
    if "tissue" not in tissue_percentage:
        tissue_key_present = False
        tissue_percentage["tissue"] = cfg.tiling.params.tissue_threshold
    if cfg.tiling.sampling_params.color_mapping is not None:
        color_mapping = {
            k: v for e in cfg.tiling.sampling_params.color_mapping for k, v in e.items()
        }
    else:
        color_mapping = None

    sampling_params = SamplingParameters(
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        tissue_percentage=tissue_percentage,
    )
    tiling_config, segmentation_config, filter_config = _build_cli_configs(cfg)

    if not skip_sampling:

        mask = process_df["sampling_status"] != "success"
        process_stack = process_df[mask]
        total = len(process_stack)

        wsi_paths_to_process = [Path(x) for x in process_stack.image_path.values.tolist()]
        mask_paths_to_process = [
            Path(x) if x is not None and not pd.isna(x) else x
            for x in process_stack.mask_path.values.tolist()
        ]
        sample_ids_to_process = process_stack.sample_id.astype(str).tolist()

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

        sampling_updates: dict[str, dict[str, str]] = {}
        with mp.Pool(processes=parallel_workers) as pool:
            args_list = [
                {
                    "sample_id": sample_id,
                    "wsi_path": wsi_fp,
                    "mask_path": mask_fp,
                    "cfg": cfg,
                    "tiling_config": tiling_config,
                    "segmentation_config": segmentation_config,
                    "filter_config": filter_config,
                    "mask_visualize_dir": mask_visualize_dir,
                    "sampling_visualize_dir": sampling_visualize_dir,
                    "sampling_params": sampling_params,
                    "disable_tqdm": True,
                    "num_workers": inner_workers,
                }
                for wsi_fp, mask_fp, sample_id in zip(wsi_paths_to_process, mask_paths_to_process, sample_ids_to_process)
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
        for result_sample_id, status_info in results:
            sampling_updates[result_sample_id] = status_info

        for result_sample_id, status_info in sampling_updates.items():
            process_df.loc[process_df["sample_id"] == result_sample_id, "sampling_status"] = (
                status_info["status"]
            )
            if "error" in status_info:
                process_df.loc[process_df["sample_id"] == result_sample_id, "error"] = (
                    status_info["error"]
                )
                process_df.loc[process_df["sample_id"] == result_sample_id, "traceback"] = (
                    status_info["traceback"]
                )
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
            if not tissue_key_present and annotation == "tissue":
                continue
            slides_with_tiles = [
                slide.sample_id
                for slide in whole_slides
                if Path(coordinates_dir, annotation, f"{slide.sample_id}.tiles.npz").is_file()
            ]
            no_tiles = process_df[~process_df["sample_id"].isin(slides_with_tiles)]
            print(f"No {annotation} tiles after sampling step: {len(no_tiles)}")
        print("=+=" * 10)

    else:
        print("=+=" * 10)
        print("All slides have been sampled. Skipping sampling step.")
        print("=+=" * 10)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
