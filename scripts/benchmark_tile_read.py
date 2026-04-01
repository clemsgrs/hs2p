#!/usr/bin/env python3

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from hs2p.api import coerce_wsd_path
from scripts.benchmark_tile_read_support import (
    BenchmarkProgressReporter,
    MODE_CONFIG,
    consume_region_tiles,
    summarize_results,
    write_csv,
)

ProgressCallback = Callable[[int, int], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark tile-read strategies from a fresh tiling result generated from an hs2p config file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="Path to an hs2p config file whose CSV contains exactly one slide.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where benchmark CSV outputs are written.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_CONFIG),
        default=list(MODE_CONFIG),
        help="Read strategies to benchmark.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of timed repetitions per mode.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed warmup repetitions per mode.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Use only the first N tiles from the artifact. Set to 0 to use all tiles.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Worker count passed to CuCIM batch reads.",
    )
    return parser.parse_args()



def benchmark_wsd_mode(
    *,
    result,
    plans,
    read_step_px: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[float, int, int]:
    import wholeslidedata as wsd

    wsi = wsd.WholeSlideImage(
        coerce_wsd_path(result.image_path, backend=result.backend),
        backend=result.backend,
    )
    tile_size_px = int(result.effective_tile_size_px)
    checksum = 0
    tile_count = 0
    start = time.perf_counter()
    for plan in plans:
        region = wsi.get_patch(
            int(plan.x),
            int(plan.y),
            int(plan.read_size_px),
            int(plan.read_size_px),
            spacing=float(result.effective_spacing_um),
            center=False,
        )
        region_checksum, region_tiles = consume_region_tiles(
            np.asarray(region),
            block_size=int(plan.block_size),
            tile_size_px=tile_size_px,
            read_step_px=read_step_px,
        )
        checksum += region_checksum
        tile_count += region_tiles
        if progress_callback is not None:
            progress_callback(1, region_tiles)
    elapsed = time.perf_counter() - start
    return elapsed, tile_count, checksum


def _require_cucim():
    try:
        return importlib.import_module("cucim")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CuCIM is required for the requested benchmark modes but is not installed."
        ) from exc


def benchmark_cucim_batch_mode(
    *,
    result,
    plans,
    read_step_px: int,
    num_workers: int,
    gpu_decode: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[float, int, int]:
    from hs2p.wsi.backends.cucim import CuCIMReader
    from scripts.benchmark_tile_utils import group_read_plans_by_read_size

    _require_cucim()
    reader = CuCIMReader(
        str(result.image_path),
        spacing_override=float(result.base_spacing_um),
        gpu_decode=gpu_decode,
    )
    tile_size_px = int(result.effective_tile_size_px)
    checksum = 0
    tile_count = 0
    start = time.perf_counter()
    for read_size_px, size_plans in group_read_plans_by_read_size(plans).items():
        locations = [(int(plan.x), int(plan.y)) for plan in size_plans]
        regions = reader.read_regions(
            locations,
            int(result.read_level),
            (int(read_size_px), int(read_size_px)),
            num_workers=int(num_workers),
        )
        for plan, region in zip(size_plans, regions):
            region_checksum, region_tiles = consume_region_tiles(
                np.asarray(region),
                block_size=int(plan.block_size),
                tile_size_px=tile_size_px,
                read_step_px=read_step_px,
            )
            checksum += region_checksum
            tile_count += region_tiles
            if progress_callback is not None:
                progress_callback(1, region_tiles)
    elapsed = time.perf_counter() - start
    return elapsed, tile_count, checksum


def run_mode(
    *,
    mode: str,
    result,
    repeat_index: int,
    read_step_px: int,
    num_workers: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    from scripts.benchmark_tile_utils import build_read_plans

    mode_cfg = MODE_CONFIG[mode]
    plans = build_read_plans(result, use_supertiles=mode_cfg["use_supertiles"])
    pixels_read = sum(int(plan.read_size_px) * int(plan.read_size_px) for plan in plans)

    if mode_cfg["reader"] == "wsd":
        elapsed, tile_count, checksum = benchmark_wsd_mode(
            result=result,
            plans=plans,
            read_step_px=read_step_px,
            progress_callback=progress_callback,
        )
    elif mode_cfg["reader"] == "cucim_batch":
        elapsed, tile_count, checksum = benchmark_cucim_batch_mode(
            result=result,
            plans=plans,
            read_step_px=read_step_px,
            num_workers=num_workers,
            gpu_decode=bool(mode_cfg.get("gpu_decode", False)),
            progress_callback=progress_callback,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return {
        "sample_id": result.sample_id,
        "image_path": str(result.image_path),
        "mode": mode,
        "description": mode_cfg["description"],
        "repeat_index": repeat_index,
        "tiles": tile_count,
        "read_calls": len(plans),
        "tiles_per_read": round(tile_count / max(len(plans), 1), 6),
        "read_level": int(result.read_level),
        "effective_tile_size_px": int(result.effective_tile_size_px),
        "read_step_px": int(read_step_px),
        "step_px_lv0": int(result.step_px_lv0 or result.tile_size_lv0),
        "elapsed_s": round(elapsed, 6),
        "tiles_per_second": round(tile_count / elapsed if elapsed > 0 else 0.0, 2),
        "megapixels_read": round(pixels_read / 1_000_000.0, 6),
        "megapixels_per_second": round(
            (pixels_read / 1_000_000.0) / elapsed if elapsed > 0 else 0.0,
            2,
        ),
        "checksum": checksum,
        "num_workers": int(num_workers),
    }


def load_single_slide_result_from_config(
    *,
    config_file: Path,
    num_workers: int,
):
    from hs2p.api import tile_slide
    from hs2p.configs.resolvers import (
        resolve_filter_config,
        resolve_segmentation_config,
        resolve_tiling_config,
    )
    from hs2p.utils import load_csv
    from hs2p.utils.setup import get_cfg_from_file

    cfg = get_cfg_from_file(config_file)
    whole_slides = load_csv(cfg)
    if len(whole_slides) != 1:
        raise ValueError(
            f"Benchmark config must resolve to exactly one slide, got {len(whole_slides)}"
        )
    tiling = resolve_tiling_config(cfg)
    segmentation = resolve_segmentation_config(cfg)
    filtering = resolve_filter_config(cfg)
    result = tile_slide(
        whole_slide=whole_slides[0],
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        num_workers=int(num_workers),
    )
    return result


def _load_existing_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    from scripts.benchmark_tile_utils import build_read_plans, limit_tiling_result

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "benchmark_summary.csv"
    runs_path = args.output_dir / "benchmark_runs.csv"
    existing_summary_rows = _load_existing_csv(summary_path)
    existing_runs_rows = _load_existing_csv(runs_path)
    existing_modes = {row["mode"] for row in existing_summary_rows}

    skipped = [m for m in args.modes if m in existing_modes]
    modes_to_run = [m for m in args.modes if m not in existing_modes]

    console = Console()
    if skipped:
        console.print(f"Skipping already-computed modes: {', '.join(skipped)}", highlight=False)
    if not modes_to_run:
        console.print("All modes already computed. Nothing to do.")
        return 0

    args = argparse.Namespace(**{**vars(args), "modes": modes_to_run})

    result = load_single_slide_result_from_config(
        config_file=args.config_file,
        num_workers=int(args.num_workers),
    )
    result = limit_tiling_result(result, max_tiles=int(args.max_tiles))
    from hs2p.wsi.streaming.plans import resolve_read_step_px

    read_step_px = resolve_read_step_px(result)

    total_runs = len(args.modes) * (int(args.warmup) + int(args.repeat))

    if any(MODE_CONFIG[mode].get("gpu_decode") for mode in args.modes):
        import os
        os.environ["ENABLE_CUSLIDE2"] = "1"
    if any(mode.startswith("cucim_") for mode in args.modes):
        _require_cucim()

    timed_rows: list[dict[str, Any]] = []
    run_counter = 0
    with BenchmarkProgressReporter(total_runs=total_runs) as reporter:
        reporter.print_banner(
            result=result,
            modes=list(args.modes),
            repeat=int(args.repeat),
            warmup=int(args.warmup),
        )
        for mode in args.modes:
            mode_cfg = MODE_CONFIG[mode]
            plans = build_read_plans(result, use_supertiles=mode_cfg["use_supertiles"])
            total_tiles = sum(int(plan.block_size) * int(plan.block_size) for plan in plans)
            for warmup_idx in range(int(args.warmup)):
                run_counter += 1
                reporter.start_run(
                    run_counter=run_counter,
                    phase="warmup",
                    mode=mode,
                    iteration_index=warmup_idx,
                    iteration_total=int(args.warmup),
                    total_read_calls=len(plans),
                    total_tiles=total_tiles,
                )
                run_mode(
                    mode=mode,
                    result=result,
                    repeat_index=warmup_idx,
                    read_step_px=read_step_px,
                    num_workers=int(args.num_workers),
                    progress_callback=reporter.advance,
                )
                reporter.finish_run()
            for repeat_index in range(int(args.repeat)):
                run_counter += 1
                reporter.start_run(
                    run_counter=run_counter,
                    phase="timed",
                    mode=mode,
                    iteration_index=repeat_index,
                    iteration_total=int(args.repeat),
                    total_read_calls=len(plans),
                    total_tiles=total_tiles,
                )
                row = run_mode(
                    mode=mode,
                    result=result,
                    repeat_index=repeat_index,
                    read_step_px=read_step_px,
                    num_workers=int(args.num_workers),
                    progress_callback=reporter.advance,
                )
                reporter.finish_run()
                reporter.print_result_row(row)
                timed_rows.append(row)

    summary_rows = summarize_results(timed_rows)
    runs_csv_path = write_csv(existing_runs_rows + timed_rows, runs_path)
    summary_csv_path = write_csv(existing_summary_rows + summary_rows, summary_path)

    print(f"\nWrote {runs_csv_path}", flush=True)
    print(f"Wrote {summary_csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
