import argparse
import traceback
from dataclasses import replace
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path

from hs2p.api import (
    FilterConfig,
    SegmentationConfig,
    TilingConfig,
    save_tiling_result,
)
from hs2p.configs.resolvers import (
    resolve_filter_config,
    resolve_sampling_spec,
    resolve_sampling_strategy,
    resolve_segmentation_config,
    resolve_tiling_config,
    validate_color_mapping,
)
import hs2p.progress as progress
from hs2p.sampling_support import (
    build_sampling_preview_assets,
    build_sampling_process_rows,
    resolve_sampling_workers,
    save_sampling_coordinates,
    validate_sampling_artifact_row,
)
from hs2p.utils import setup, load_csv
from hs2p.wsi import (
    CoordinateExtractionResult,
    UnifiedCoordinateRequest,
    execute_coordinate_request,
    write_coordinate_preview,
)
from hs2p.wsi.types import (
    CoordinateOutputMode,
    CoordinateSelectionStrategy,
    ResolvedSamplingSpec,
)
from hs2p.tiling_artifacts import validate_required_columns, write_process_list
from hs2p.wsi.backend import resolve_backend

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


def process_slide(
    *,
    sample_id: str,
    wsi_path: Path,
    mask_path: Path | None,
    cfg,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
    mask_preview_dir,
    sampling_preview_dir,
    resolved_sampling_spec: ResolvedSamplingSpec,
    selection_strategy: str | None = None,
    disable_tqdm: bool = False,
    num_workers: int = 4,
):
    """
    Process a single slide: sample tile coordinates and write previews if needed.
    """
    wsi_name = sample_id
    if selection_strategy is None:
        selection_strategy = resolve_sampling_strategy(cfg)
    try:
        backend_selection = resolve_backend(
            cfg.tiling.backend,
            wsi_path=wsi_path,
            mask_path=mask_path,
        )
        if backend_selection.reason is not None:
            progress.emit_progress_log(
                f"[backend] {sample_id}: {backend_selection.reason}"
            )
        effective_tiling_config = (
            tiling_config
            if backend_selection.backend == tiling_config.backend
            else replace(tiling_config, backend=backend_selection.backend)
        )
        preview_palette, color_mapping = build_sampling_preview_assets(
            resolved_sampling_spec,
            save_previews=cfg.save_previews
            and (sampling_preview_dir is not None or mask_preview_dir is not None),
        )
        mask_preview_path = None
        mask_preview_paths_by_annotation = None
        if cfg.save_previews and mask_preview_dir is not None:
            if selection_strategy == CoordinateSelectionStrategy.INDEPENDENT_SAMPLING:
                mask_preview_paths_by_annotation = {
                    annotation: mask_preview_dir / annotation / f"{wsi_name}.jpg"
                    for annotation in resolved_sampling_spec.active_annotations
                }
            else:
                mask_preview_path = Path(mask_preview_dir, f"{wsi_name}.png")

        response = execute_coordinate_request(
            UnifiedCoordinateRequest(
                wsi_path=wsi_path,
                mask_path=mask_path,
                backend=backend_selection.backend,
                segment_params=segmentation_config,
                tiling_params=effective_tiling_config,
                filter_params=filter_config,
                sampling_spec=resolved_sampling_spec,
                selection_strategy=selection_strategy,
                output_mode=CoordinateOutputMode.PER_ANNOTATION,
                mask_preview_path=mask_preview_path,
                mask_preview_paths_by_annotation=mask_preview_paths_by_annotation,
                preview_downsample=cfg.tiling.preview.downsample,
                preview_palette=preview_palette,
                preview_pixel_mapping=(
                    resolved_sampling_spec.pixel_mapping
                    if color_mapping is not None
                    else None
                ),
                preview_color_mapping=color_mapping,
                disable_tqdm=disable_tqdm,
                num_workers=num_workers,
            )
        )
        per_annotation_results = response.per_annotation_results or {}
        rows: list[dict[str, object]] = []
        for annotation in resolved_sampling_spec.active_annotations:
            extraction = per_annotation_results.get(annotation)
            coordinates = extraction.coordinates if extraction is not None else []
            artifacts = None
            if len(coordinates) > 0:
                artifacts = save_sampling_coordinates(
                    sample_id=sample_id,
                    image_path=wsi_path,
                    mask_path=mask_path,
                    backend=backend_selection.backend,
                    cfg=cfg,
                    tiling_config=effective_tiling_config,
                    segmentation_config=segmentation_config,
                    filter_config=filter_config,
                    annotation=annotation,
                    coordinates=coordinates,
                    extraction=extraction,
                    resolved_sampling_spec=resolved_sampling_spec,
                    save_tiling_result=save_tiling_result,
                    selection_strategy=selection_strategy,
                )
                if cfg.save_previews and sampling_preview_dir is not None:
                    write_coordinate_preview(
                        wsi_path=wsi_path,
                        coordinates=coordinates,
                        tile_size_lv0=extraction.tile_size_lv0,
                        save_dir=sampling_preview_dir,
                        sample_id=sample_id,
                        downsample=cfg.tiling.preview.downsample,
                        backend=backend_selection.backend,
                        mask_path=mask_path,
                        annotation=annotation,
                        palette=preview_palette,
                        pixel_mapping=resolved_sampling_spec.pixel_mapping,
                        color_mapping=color_mapping,
                    )
            rows.append(
                {
                    "sample_id": sample_id,
                    "annotation": annotation,
                    "image_path": str(wsi_path),
                    "annotation_mask_path": (
                        str(mask_path) if mask_path is not None else None
                    ),
                    "sampling_status": "success",
                    "num_tiles": len(coordinates),
                    "coordinates_npz_path": (
                        str(artifacts.coordinates_npz_path) if artifacts is not None else np.nan
                    ),
                    "coordinates_meta_path": (
                        str(artifacts.coordinates_meta_path)
                        if artifacts is not None
                        else np.nan
                    ),
                    "error": np.nan,
                    "traceback": np.nan,
                }
            )
        return sample_id, {"status": "success", "rows": rows}

    except Exception as e:
        active_annotations = (
            list(resolved_sampling_spec.active_annotations)
            if resolved_sampling_spec is not None
            else [np.nan]
        )
        rows = []
        for annotation in active_annotations:
            rows.append(
                {
                    "sample_id": sample_id,
                    "annotation": annotation,
                    "image_path": str(wsi_path),
                    "annotation_mask_path": (
                        str(mask_path) if mask_path is not None else None
                    ),
                    "sampling_status": "failed",
                    "num_tiles": 0,
                    "coordinates_npz_path": np.nan,
                    "coordinates_meta_path": np.nan,
                    "error": str(e),
                    "traceback": str(traceback.format_exc()),
                }
            )
        return sample_id, {"status": "failed", "rows": rows}

def main(args):
    reporter = progress.create_cli_progress_reporter(
        output_dir=getattr(args, "output_dir", None)
    )
    with progress.activate_progress_reporter(reporter):
        try:
            cfg = setup(args)
            output_dir = Path(cfg.output_dir)

            whole_slides = load_csv(cfg, mask_column="annotation_mask_path")
            tiling_config = resolve_tiling_config(cfg)
            segmentation_config = resolve_segmentation_config(cfg)
            filter_config = resolve_filter_config(cfg)
            resolved_sampling_spec = resolve_sampling_spec(cfg, tiling=tiling_config)
            selection_strategy = resolve_sampling_strategy(cfg)
            active_annotations = resolved_sampling_spec.active_annotations
            progress.emit_progress(
                "run.started",
                command="sampling",
                slide_count=len(whole_slides),
                backend=tiling_config.backend,
                target_spacing_um=tiling_config.target_spacing_um,
                target_tile_size_px=tiling_config.target_tile_size_px,
                output_dir=str(output_dir),
                num_workers=int(cfg.speed.num_workers),
                resume=bool(cfg.resume),
                read_coordinates_from=None,
            )

            process_list = output_dir / "process_list.csv"
            if process_list.is_file() and cfg.resume:
                process_df = pd.read_csv(process_list)
                validate_required_columns(
                    process_df,
                    required_columns={
                        "sample_id",
                        "annotation",
                        "image_path",
                        "annotation_mask_path",
                        "sampling_status",
                        "num_tiles",
                        "coordinates_npz_path",
                        "coordinates_meta_path",
                        "error",
                        "traceback",
                    },
                    file_path=process_list,
                    file_label="sampling process_list.csv",
                )
                process_df["annotation_mask_path"] = process_df["annotation_mask_path"].apply(
                    lambda x: str(x) if pd.notna(x) else None
                )
            else:
                process_df = pd.DataFrame(
                    build_sampling_process_rows(
                        whole_slides=whole_slides,
                        active_annotations=active_annotations,
                    ),
                    columns=[
                        "sample_id",
                        "annotation",
                        "image_path",
                        "annotation_mask_path",
                        "sampling_status",
                        "num_tiles",
                        "coordinates_npz_path",
                        "coordinates_meta_path",
                        "error",
                        "traceback",
                    ],
                )

            slides_to_process = []
            for slide in whole_slides:
                slide_rows = process_df[process_df["sample_id"] == slide.sample_id]
                try:
                    if slide_rows.empty or set(slide_rows["annotation"].tolist()) != set(
                        active_annotations
                    ):
                        raise ValueError("sampling rows missing required annotations")
                    for annotation in active_annotations:
                        row = slide_rows[slide_rows["annotation"] == annotation]
                        if row.empty:
                            raise ValueError("sampling row missing annotation")
                        validate_sampling_artifact_row(
                            row=row.iloc[0].to_dict(),
                            whole_slide=slide,
                            tiling_config=tiling_config,
                            segmentation_config=segmentation_config,
                            filter_config=filter_config,
                            expected_tissue_threshold=resolved_sampling_spec.tissue_percentage[
                                annotation
                            ],
                            selection_strategy=selection_strategy,
                        )
                except Exception:
                    slides_to_process.append(slide)

            total_slides = len(whole_slides)
            if slides_to_process:
                total = len(slides_to_process)
                parallel_workers, inner_workers = resolve_sampling_workers(
                    cfg, slide_count=total
                )

                tiles_dir = output_dir / "tiles"
                tiles_dir.mkdir(exist_ok=True, parents=True)
                mask_preview_dir = None
                sampling_preview_dir = None
                if cfg.save_previews:
                    preview_dir = output_dir / "preview"
                    mask_preview_dir = Path(preview_dir, "mask")
                    sampling_preview_dir = Path(preview_dir, "sampling")
                    mask_preview_dir.mkdir(exist_ok=True, parents=True)
                    sampling_preview_dir.mkdir(exist_ok=True, parents=True)

                progress.emit_progress("sampling.started", total=total)
                sampling_updates: dict[str, dict[str, str]] = {}
                completed = 0
                failed = 0
                sampled_tiles = 0
                with mp.Pool(processes=parallel_workers) as pool:
                    args_list = [
                        {
                            "sample_id": slide.sample_id,
                            "wsi_path": slide.image_path,
                            "annotation_mask_path": slide.mask_path,
                            "cfg": cfg,
                            "tiling_config": tiling_config,
                            "segmentation_config": segmentation_config,
                            "filter_config": filter_config,
                            "mask_preview_dir": mask_preview_dir,
                            "sampling_preview_dir": sampling_preview_dir,
                            "resolved_sampling_spec": resolved_sampling_spec,
                            "selection_strategy": selection_strategy,
                            "disable_tqdm": True,
                            "num_workers": inner_workers,
                        }
                        for slide in slides_to_process
                    ]
                    for result_sample_id, status_info in pool.imap(
                        process_slide_wrapper, args_list
                    ):
                        sampling_updates[result_sample_id] = status_info
                        sampled_tiles += sum(
                            int(row.get("num_tiles", 0) or 0)
                            for row in status_info["rows"]
                            if row.get("sampling_status") == "success"
                        )
                        if status_info["status"] == "success":
                            completed += 1
                        else:
                            failed += 1
                        progress.emit_progress(
                            "sampling.progress",
                            total=total,
                            completed=completed,
                            failed=failed,
                            pending=max(0, total - completed - failed),
                            sampled_tiles=sampled_tiles,
                        )

                for result_sample_id, status_info in sampling_updates.items():
                    process_df = process_df[process_df["sample_id"] != result_sample_id]
                    process_df = pd.concat(
                        [process_df, pd.DataFrame(status_info["rows"])],
                        ignore_index=True,
                    )
                write_process_list(
                    process_df.to_dict(orient="records"),
                    process_list,
                )
            else:
                sampled_tiles = 0

            failed_sampling = process_df[process_df["sampling_status"] == "failed"][
                "sample_id"
            ].nunique()
            completed_sampling = total_slides - failed_sampling
            zero_tile_successes_by_annotation = {
                annotation: int(
                    (
                        (process_df["annotation"] == annotation)
                        & (process_df["sampling_status"] == "success")
                        & (process_df["num_tiles"].fillna(0).astype(int) == 0)
                    ).sum()
                )
                for annotation in active_annotations
            }
            sampled_tiles = int(
                process_df.loc[
                    process_df["sampling_status"] == "success", "num_tiles"
                ]
                .fillna(0)
                .astype(int)
                .sum()
            )
            progress.emit_progress(
                "sampling.finished",
                total=total_slides,
                completed=completed_sampling,
                failed=int(failed_sampling),
                pending=0,
                sampled_tiles=sampled_tiles,
                output_dir=str(output_dir),
                process_list_path=str(process_list),
                zero_tile_successes_by_annotation=zero_tile_successes_by_annotation,
            )
            progress.emit_progress(
                "run.finished",
                command="sampling",
                output_dir=str(output_dir),
                process_list_path=str(process_list),
                logs_dir=str(output_dir / "logs"),
            )
        except Exception as exc:
            progress.emit_progress("run.failed", stage="sampling", error=str(exc))
            raise


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
