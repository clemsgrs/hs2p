
import argparse
import multiprocessing as mp
import os
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns

from hs2p.api import (
    FilterConfig,
    SegmentationConfig,
    TilingConfig,
    save_tiling_result,
)
from hs2p.configs import FilterConfig, SegmentationConfig, TilingConfig
from hs2p.configs.resolvers import (
    resolve_filter_config,
    resolve_sampling_spec,
    resolve_sampling_strategy,
    resolve_segmentation_config,
    resolve_tiling_config,
    validate_color_mapping,
)
from hs2p.preprocessing import TileGeometry, TilingResult
from hs2p.artifacts import load_tiling_result, validate_required_columns, write_process_list
import hs2p.progress as progress
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
    SamplingSpec,
)
from hs2p.wsi.backend import resolve_backend


def resolve_sampling_workers(cfg, *, slide_count: int) -> tuple[int, int]:
    if hasattr(cfg.speed, "inner_workers"):
        raise ValueError(
            "cfg.speed.inner_workers is no longer supported; sampling always uses 1 inner worker per slide"
        )

    parallel_workers = min(mp.cpu_count(), int(cfg.speed.num_workers))
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        parallel_workers = min(
            parallel_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        )
    parallel_workers = max(1, parallel_workers)
    outer_workers = max(1, min(parallel_workers, int(slide_count)))
    return outer_workers, 1


def selection_strategy_from_cfg(cfg) -> str:
    return resolve_sampling_strategy(cfg)


def build_sampling_preview_assets(
    resolved_sampling_spec: SamplingSpec,
    *,
    save_previews: bool,
):
    if not save_previews:
        return None, None

    preview_palette = np.zeros(shape=768, dtype=int)
    if resolved_sampling_spec.color_mapping is None:
        ncat = len(resolved_sampling_spec.pixel_mapping)
        if ncat <= 10:
            color_palette = sns.color_palette("tab10")[:ncat]
        elif ncat <= 20:
            color_palette = sns.color_palette("tab20")[:ncat]
        else:
            raise ValueError(
                f"Implementation supports up to 20 categories (provided pixel_mapping has {ncat})"
            )
        color_mapping = {
            k: tuple(int(round(255 * x)) for x in color_palette[i])
            for i, k in enumerate(resolved_sampling_spec.pixel_mapping.keys())
        }
    else:
        color_mapping = resolved_sampling_spec.color_mapping
    validate_color_mapping(
        pixel_mapping=resolved_sampling_spec.pixel_mapping,
        color_mapping=color_mapping,
    )
    p = [0] * 3 * len(color_mapping)
    for k, v in resolved_sampling_spec.pixel_mapping.items():
        if color_mapping[k] is not None:
            p[v * 3 : v * 3 + 3] = color_mapping[k]
    preview_palette[0 : len(p)] = np.array(p).astype(int)
    return preview_palette, color_mapping


def save_sampling_coordinates(
    *,
    sample_id: str,
    image_path: Path,
    mask_path: Path | None,
    backend: str,
    cfg: Any,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
    annotation: str,
    coordinates: list[tuple[int, int]],
    extraction: CoordinateExtractionResult,
    resolved_sampling_spec: SamplingSpec,
    save_tiling_result,
    selection_strategy: str | None = None,
    output_mode: str = CoordinateOutputMode.PER_ANNOTATION,
):
    if selection_strategy is None:
        selection_strategy = selection_strategy_from_cfg(cfg)
    annotation_threshold = resolved_sampling_spec.tissue_percentage[annotation]
    coordinate_array = np.asarray(coordinates, dtype=np.int64)
    if coordinate_array.size == 0:
        coordinate_array = np.empty((0, 2), dtype=np.int64)
    result = TilingResult(
        tiles=TileGeometry(
            x=coordinate_array[:, 0],
            y=coordinate_array[:, 1],
            tissue_fractions=np.zeros(len(coordinates), dtype=np.float32),
            tile_index=np.arange(len(coordinates), dtype=np.int32),
            requested_tile_size_px=tiling_config.requested_tile_size_px,
            requested_spacing_um=tiling_config.requested_spacing_um,
            read_level=extraction.read_level,
            read_tile_size_px=extraction.read_tile_size_px,
            read_spacing_um=extraction.read_spacing_um,
            tile_size_lv0=extraction.tile_size_lv0,
            is_within_tolerance=True,
            base_spacing_um=extraction.read_spacing_um,
            slide_dimensions=[0, 0],
            level_downsamples=[1.0],
            overlap=tiling_config.overlap,
            min_tissue_fraction=annotation_threshold,
        ),
        sample_id=sample_id,
        image_path=image_path,
        mask_path=mask_path,
        backend=backend,
        requested_backend=backend,
        step_px_lv0=extraction.step_px_lv0,
        tolerance=tiling_config.tolerance,
        tissue_method=segmentation_config.method,
        seg_downsample=segmentation_config.downsample,
        seg_level=0,
        seg_spacing_um=0.0,
        seg_sthresh=segmentation_config.sthresh,
        seg_sthresh_up=segmentation_config.sthresh_up,
        seg_mthresh=segmentation_config.mthresh,
        seg_close=segmentation_config.close,
        ref_tile_size_px=filter_config.ref_tile_size,
        a_t=filter_config.a_t,
        a_h=filter_config.a_h,
        filter_white=filter_config.filter_white,
        filter_black=filter_config.filter_black,
        white_threshold=filter_config.white_threshold,
        black_threshold=filter_config.black_threshold,
        fraction_threshold=filter_config.fraction_threshold,
        filter_grayspace=filter_config.filter_grayspace,
        grayspace_saturation_threshold=filter_config.grayspace_saturation_threshold,
        grayspace_fraction_threshold=filter_config.grayspace_fraction_threshold,
        filter_blur=filter_config.filter_blur,
        blur_threshold=filter_config.blur_threshold,
        qc_spacing_um=filter_config.qc_spacing_um,
        annotation=annotation,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
    )
    annotation_dir = Path(cfg.output_dir, "tiles", annotation)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    return save_tiling_result(
        result, output_dir=cfg.output_dir, tiles_dir=annotation_dir
    )


def build_sampling_process_rows(
    *,
    whole_slides,
    active_annotations: tuple[str, ...],
):
    rows = []
    for slide in whole_slides:
        for annotation in active_annotations:
            rows.append(
                {
                    "sample_id": slide.sample_id,
                    "annotation": annotation,
                    "image_path": str(slide.image_path),
                    "mask_path": (
                        str(slide.mask_path) if slide.mask_path is not None else None
                    ),
                    "requested_backend": None,
                    "backend": None,
                    "sampling_status": "tbp",
                    "num_tiles": 0,
                    "coordinates_npz_path": np.nan,
                    "coordinates_meta_path": np.nan,
                    "error": np.nan,
                    "traceback": np.nan,
                }
            )
    return rows


def validate_sampling_artifact_row(
    *,
    row: dict[str, object],
    whole_slide,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
    expected_tissue_threshold: float,
    selection_strategy: str,
) -> None:
    if row.get("sampling_status") != "success":
        raise ValueError("sampling row is not successful")
    num_tiles = int(row.get("num_tiles", 0))
    if num_tiles == 0:
        return
    npz_path = Path(str(row["coordinates_npz_path"]))
    meta_path = Path(str(row["coordinates_meta_path"]))
    result = load_tiling_result(npz_path, meta_path)
    if result.sample_id != whole_slide.sample_id:
        raise ValueError("sampling sample_id mismatch")
    if result.image_path != whole_slide.image_path:
        raise ValueError("sampling image_path mismatch")
    if result.mask_path != whole_slide.mask_path:
        raise ValueError("sampling mask_path mismatch")
    if result.backend != tiling_config.backend:
        raise ValueError("sampling backend mismatch")
    if result.requested_spacing_um != tiling_config.requested_spacing_um:
        raise ValueError("sampling requested_spacing_um mismatch")
    if result.requested_tile_size_px != tiling_config.requested_tile_size_px:
        raise ValueError("sampling requested_tile_size_px mismatch")
    if result.overlap != tiling_config.overlap:
        raise ValueError("sampling overlap mismatch")
    if result.min_tissue_fraction != expected_tissue_threshold:
        raise ValueError("sampling tissue_threshold mismatch")
    if result.tolerance != tiling_config.tolerance:
        raise ValueError("sampling tolerance mismatch")
    if result.seg_downsample != segmentation_config.downsample:
        raise ValueError("sampling seg_downsample mismatch")
    if result.seg_sthresh != segmentation_config.sthresh:
        raise ValueError("sampling sthresh mismatch")
    if result.seg_sthresh_up != segmentation_config.sthresh_up:
        raise ValueError("sampling sthresh_up mismatch")
    if result.seg_mthresh != segmentation_config.mthresh:
        raise ValueError("sampling mthresh mismatch")
    if result.seg_close != segmentation_config.close:
        raise ValueError("sampling close mismatch")
    if result.tissue_method != segmentation_config.method:
        raise ValueError("sampling tissue_method mismatch")
    if result.ref_tile_size_px != filter_config.ref_tile_size:
        raise ValueError("sampling ref_tile_size mismatch")
    if result.a_t != filter_config.a_t:
        raise ValueError("sampling a_t mismatch")
    if result.a_h != filter_config.a_h:
        raise ValueError("sampling a_h mismatch")
    if result.filter_white != filter_config.filter_white:
        raise ValueError("sampling filter_white mismatch")
    if result.filter_black != filter_config.filter_black:
        raise ValueError("sampling filter_black mismatch")
    if filter_config.filter_white and result.white_threshold != filter_config.white_threshold:
        raise ValueError("sampling white_threshold mismatch")
    if filter_config.filter_black and result.black_threshold != filter_config.black_threshold:
        raise ValueError("sampling black_threshold mismatch")
    if (
        (filter_config.filter_white or filter_config.filter_black)
        and result.fraction_threshold != filter_config.fraction_threshold
    ):
        raise ValueError("sampling fraction_threshold mismatch")
    if result.filter_grayspace != filter_config.filter_grayspace:
        raise ValueError("sampling filter_grayspace mismatch")
    if (
        filter_config.filter_grayspace
        and (
        result.grayspace_saturation_threshold
        != filter_config.grayspace_saturation_threshold
        )
    ):
        raise ValueError("sampling grayspace_saturation_threshold mismatch")
    if (
        filter_config.filter_grayspace
        and (
        result.grayspace_fraction_threshold
        != filter_config.grayspace_fraction_threshold
        )
    ):
        raise ValueError("sampling grayspace_fraction_threshold mismatch")
    if result.filter_blur != filter_config.filter_blur:
        raise ValueError("sampling filter_blur mismatch")
    if filter_config.filter_blur and result.blur_threshold != filter_config.blur_threshold:
        raise ValueError("sampling blur_threshold mismatch")
    if (
        (
            filter_config.filter_white
            or filter_config.filter_black
            or filter_config.filter_grayspace
            or filter_config.filter_blur
        )
        and result.qc_spacing_um != filter_config.qc_spacing_um
    ):
        raise ValueError("sampling qc_spacing_um mismatch")
    if result.annotation != row["annotation"]:
        raise ValueError("sampling annotation mismatch")
    if result.selection_strategy != selection_strategy:
        raise ValueError("sampling selection_strategy mismatch")
    if result.output_mode != CoordinateOutputMode.PER_ANNOTATION:
        raise ValueError("sampling output_mode mismatch")


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
    resolved_sampling_spec: SamplingSpec,
    selection_strategy: str | None = None,
    disable_tqdm: bool = False,
    num_workers: int = 4,
):
    """
    Process a single slide: sample tile coordinates and write previews if needed.
    """
    wsi_name = sample_id
    requested_backend = cfg.tiling.backend
    actual_backend = None
    if selection_strategy is None:
        selection_strategy = resolve_sampling_strategy(cfg)
    try:
        backend_selection = resolve_backend(
            requested_backend,
            wsi_path=wsi_path,
            mask_path=mask_path,
        )
        actual_backend = backend_selection.backend
        if backend_selection.reason is not None:
            progress.emit_progress_log(
                f"[backend] {sample_id}: {backend_selection.reason}"
            )
        preview_cfg = cfg.tiling.preview
        effective_tiling_config = (
            tiling_config
            if backend_selection.backend == tiling_config.backend
            else replace(tiling_config, backend=backend_selection.backend)
        )
        preview_palette, color_mapping = build_sampling_preview_assets(
            resolved_sampling_spec,
            save_previews=preview_cfg.save
            and (sampling_preview_dir is not None or mask_preview_dir is not None),
        )
        mask_preview_path = None
        mask_preview_paths_by_annotation = None
        if preview_cfg.save and mask_preview_dir is not None:
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
                preview_downsample=preview_cfg.downsample,
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
            coordinates = (
                list(zip(extraction.x.tolist(), extraction.y.tolist()))
                if extraction is not None
                else []
            )
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
                if preview_cfg.save and sampling_preview_dir is not None:
                    write_coordinate_preview(
                        wsi_path=wsi_path,
                        coordinates=coordinates,
                        tile_size_lv0=extraction.tile_size_lv0,
                        save_dir=sampling_preview_dir,
                        sample_id=sample_id,
                        downsample=preview_cfg.downsample,
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
                    "mask_path": (
                        str(mask_path) if mask_path is not None else None
                    ),
                    "requested_backend": requested_backend,
                    "backend": actual_backend,
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
                    "mask_path": (
                        str(mask_path) if mask_path is not None else None
                    ),
                    "requested_backend": requested_backend,
                    "backend": actual_backend,
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

            whole_slides = load_csv(cfg, require_mask_column=True)
            if any(slide.mask_path is None for slide in whole_slides):
                raise ValueError("Sampling CSV requires mask_path for every row")
            tiling_config = resolve_tiling_config(cfg)
            segmentation_config = resolve_segmentation_config(cfg)
            filter_config = resolve_filter_config(cfg)
            resolved_sampling_spec = resolve_sampling_spec(cfg, tiling=tiling_config)
            selection_strategy = resolve_sampling_strategy(cfg)
            preview_cfg = cfg.tiling.preview
            active_annotations = resolved_sampling_spec.active_annotations
            progress.emit_progress(
                "run.started",
                command="sampling",
                slide_count=len(whole_slides),
                backend=tiling_config.backend,
                requested_spacing_um=tiling_config.requested_spacing_um,
                requested_tile_size_px=tiling_config.requested_tile_size_px,
                output_dir=str(output_dir),
                num_workers=int(cfg.speed.num_workers),
                resume=bool(cfg.resume),
                read_coordinates_from=None,
            )

            process_list = output_dir / "process_list.csv"
            if process_list.is_file() and cfg.resume:
                process_df = pd.read_csv(process_list)
                legacy_mask_columns = [
                    column
                    for column in ("tissue_mask_path", "annotation_mask_path")
                    if column in process_df.columns
                ]
                if legacy_mask_columns:
                    raise ValueError(
                        "Unsupported sampling process_list.csv schema in "
                        f"{process_list}; deprecated mask columns present: "
                        + ", ".join(sorted(legacy_mask_columns))
                    )
                validate_required_columns(
                    process_df,
                    required_columns={
                        "sample_id",
                        "annotation",
                        "image_path",
                        "mask_path",
                        "requested_backend",
                        "backend",
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
                process_df["mask_path"] = process_df["mask_path"].apply(
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
                        "mask_path",
                        "requested_backend",
                        "backend",
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
                if preview_cfg.save:
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
                            "mask_path": slide.mask_path,
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
