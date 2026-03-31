#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from hs2p.api import coerce_wsd_path


MODE_CONFIG = {
    "regular_wsd": {
        "reader": "wsd",
        "use_supertiles": False,
        "gpu_decode": False,
        "description": "wholeslidedata tile-by-tile reads",
    },
    "supertiles_wsd": {
        "reader": "wsd",
        "use_supertiles": True,
        "gpu_decode": False,
        "description": "wholeslidedata dense 8x8/4x4 supertile reads",
    },
    "cucim_batch_regular": {
        "reader": "cucim_batch",
        "use_supertiles": False,
        "gpu_decode": False,
        "description": "CuCIM batched regular tile reads",
    },
    "cucim_batch_supertiles": {
        "reader": "cucim_batch",
        "use_supertiles": True,
        "gpu_decode": False,
        "description": "CuCIM batched dense 8x8/4x4 supertile reads",
    },
    "cucim_batch_gpu_regular": {
        "reader": "cucim_batch",
        "use_supertiles": False,
        "gpu_decode": True,
        "description": "CuCIM batched regular tile reads with GPU decode",
    },
    "cucim_batch_gpu_supertiles": {
        "reader": "cucim_batch",
        "use_supertiles": True,
        "gpu_decode": True,
        "description": "CuCIM batched dense 8x8/4x4 supertile reads with GPU decode",
    },
}

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


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for CSV output: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["mode"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for mode, mode_rows in grouped.items():
        elapsed = [float(row["elapsed_s"]) for row in mode_rows]
        tiles_per_second = [float(row["tiles_per_second"]) for row in mode_rows]
        megapixels_per_second = [
            float(row["megapixels_per_second"]) for row in mode_rows
        ]
        summary_rows.append(
            {
                "mode": mode,
                "description": mode_rows[0]["description"],
                "repeat": len(mode_rows),
                "tiles": int(mode_rows[0]["tiles"]),
                "read_calls": int(mode_rows[0]["read_calls"]),
                "tiles_per_read": float(mode_rows[0]["tiles_per_read"]),
                "mean_elapsed_s": round(statistics.mean(elapsed), 6),
                "std_elapsed_s": round(
                    statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0,
                    6,
                ),
                "mean_tiles_per_second": round(statistics.mean(tiles_per_second), 2),
                "std_tiles_per_second": round(
                    statistics.pstdev(tiles_per_second)
                    if len(tiles_per_second) > 1
                    else 0.0,
                    2,
                ),
                "mean_megapixels_per_second": round(
                    statistics.mean(megapixels_per_second), 2
                ),
            }
        )
    return summary_rows


def _touch_tile(tile: np.ndarray) -> int:
    arr = np.asarray(tile)
    return int(arr[0, 0].sum()) + int(arr[-1, -1].sum())


def _consume_region_tiles(
    region: np.ndarray,
    *,
    block_size: int,
    tile_size_px: int,
    read_step_px: int,
) -> tuple[int, int]:
    from scripts.benchmark_tile_utils import TileReadPlan, iter_tiles_from_region

    checksum = 0
    tile_count = 0
    for tile in iter_tiles_from_region(
        region,
        plan=TileReadPlan(x=0, y=0, read_size_px=int(region.shape[0]), block_size=block_size),
        tile_size_px=tile_size_px,
        read_step_px=read_step_px,
    ):
        checksum += _touch_tile(tile)
        tile_count += 1
    return checksum, tile_count


class BenchmarkProgressReporter:
    def __init__(
        self,
        *,
        total_runs: int,
        console: Console | None = None,
        force_rich: bool | None = None,
        plain_update_interval_s: float = 2.0,
    ) -> None:
        self.total_runs = max(1, int(total_runs))
        self.console = console or Console()
        self.use_rich = self.console.is_terminal if force_rich is None else force_rich
        self.plain_update_interval_s = max(0.1, float(plain_update_interval_s))
        self._progress: Progress | None = None
        self._overall_task_id: int | None = None
        self._run_task_id: int | None = None
        self._active_total_read_calls = 0
        self._active_total_tiles = 0
        self._active_completed_read_calls = 0
        self._active_completed_tiles = 0
        self._last_plain_update = 0.0

    def __enter__(self) -> "BenchmarkProgressReporter":
        if self.use_rich:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TextColumn("tiles {task.fields[tiles_status]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
                expand=True,
            )
            self._progress.start()
            self._overall_task_id = self._progress.add_task(
                "[cyan]All benchmark runs",
                total=self.total_runs,
                tiles_status="",
            )
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._progress is not None:
            self._progress.stop()

    def print_banner(self, *, result, modes: list[str], repeat: int, warmup: int) -> None:
        self.console.print(
            (
                f"[bold]Benchmarking[/bold] sample={result.sample_id} "
                f"tiles={int(result.num_tiles):,} read_tile={int(result.read_tile_size_px)}px "
                f"modes={', '.join(modes)} repeat={int(repeat)} warmup={int(warmup)}"
            ),
            highlight=False,
        )

    def start_run(
        self,
        *,
        run_counter: int,
        phase: str,
        mode: str,
        iteration_index: int,
        iteration_total: int,
        total_read_calls: int,
        total_tiles: int,
    ) -> None:
        self._active_total_read_calls = max(1, int(total_read_calls))
        self._active_total_tiles = max(1, int(total_tiles))
        self._active_completed_read_calls = 0
        self._active_completed_tiles = 0
        self._last_plain_update = 0.0
        description = (
            f"[green]{run_counter}/{self.total_runs}[/green] "
            f"{phase} {mode} ({int(iteration_index) + 1}/{int(iteration_total)})"
        )
        if self._progress is not None and self._overall_task_id is not None:
            if self._run_task_id is not None:
                self._progress.remove_task(self._run_task_id)
            self._progress.update(
                self._overall_task_id,
                description=f"[cyan]All benchmark runs ({run_counter}/{self.total_runs})",
            )
            self._run_task_id = self._progress.add_task(
                description,
                total=self._active_total_read_calls,
                completed=0,
                tiles_status=f"0/{self._active_total_tiles:,}",
            )
            self._progress.refresh()
            return
            self.console.print(
                (
                    f"[{run_counter}/{self.total_runs}] {phase} {mode} "
                    f"({int(iteration_index) + 1}/{int(iteration_total)}) "
                    f"read_calls=0/{self._active_total_read_calls:,} "
                    f"tiles=0/{self._active_total_tiles:,}"
                ),
                highlight=False,
            )

    def advance(self, regions: int, tiles: int) -> None:
        self._active_completed_read_calls += int(regions)
        self._active_completed_tiles += int(tiles)
        tiles_status = f"{self._active_completed_tiles:,}/{self._active_total_tiles:,}"
        if self._progress is not None and self._run_task_id is not None:
            self._progress.update(
                self._run_task_id,
                advance=int(regions),
                tiles_status=tiles_status,
            )
            return
        now = time.monotonic()
        if (
            self._last_plain_update == 0.0
            or now - self._last_plain_update >= self.plain_update_interval_s
            or self._active_completed_read_calls >= self._active_total_read_calls
        ):
            self.console.print(
                (
                    f"  progress read_calls={self._active_completed_read_calls:,}/"
                    f"{self._active_total_read_calls:,} "
                    f"tiles={tiles_status}"
                ),
                highlight=False,
            )
            self._last_plain_update = now

    def finish_run(self) -> None:
        if self._progress is not None and self._overall_task_id is not None:
            if self._run_task_id is not None:
                self._progress.update(
                    self._run_task_id,
                    completed=self._active_total_read_calls,
                    tiles_status=(
                        f"{self._active_total_tiles:,}/{self._active_total_tiles:,}"
                    ),
                )
            self._progress.advance(self._overall_task_id, 1)

    def print_result_row(self, row: dict[str, Any]) -> None:
        self.console.print(
            (
                f"{row['mode']:<24} rep={int(row['repeat_index']) + 1} "
                f"tiles={int(row['tiles']):>7,d} "
                f"read_calls={int(row['read_calls']):>7,d} "
                f"elapsed={float(row['elapsed_s']):>8.3f}s "
                f"throughput={float(row['tiles_per_second']):>10,.0f} tiles/s"
            ),
            highlight=False,
        )


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
    tile_size_px = int(result.read_tile_size_px)
    checksum = 0
    tile_count = 0
    start = time.perf_counter()
    for plan in plans:
        region = wsi.get_patch(
            int(plan.x),
            int(plan.y),
            int(plan.read_size_px),
            int(plan.read_size_px),
            spacing=float(result.read_spacing_um),
            center=False,
        )
        region_checksum, region_tiles = _consume_region_tiles(
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
    from hs2p.wsi.cucim_reader import CuImageReader
    from scripts.benchmark_tile_utils import group_read_plans_by_read_size

    _require_cucim()
    reader = CuImageReader(result.image_path, gpu_decode=gpu_decode)
    tile_size_px = int(result.read_tile_size_px)
    checksum = 0
    tile_count = 0
    start = time.perf_counter()
    for read_size_px, size_plans in group_read_plans_by_read_size(plans).items():
        locations = [(int(plan.x), int(plan.y)) for plan in size_plans]
        regions = reader.read_region(
            locations,
            (int(read_size_px), int(read_size_px)),
            level=int(result.read_level),
            num_workers=int(num_workers),
        )
        for plan, region in zip(size_plans, regions):
            region_checksum, region_tiles = _consume_region_tiles(
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
        "read_tile_size_px": int(result.read_tile_size_px),
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
    from hs2p.utils.config import get_cfg_from_file

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
    read_step_px = int(result.read_step_px)

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
