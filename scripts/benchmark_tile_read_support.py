
import csv
import statistics
import time
from pathlib import Path
from typing import Any

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


def touch_tile(tile: np.ndarray) -> int:
    arr = np.asarray(tile)
    return int(arr[0, 0].sum()) + int(arr[-1, -1].sum())


def consume_region_tiles(
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
        checksum += touch_tile(tile)
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
                f"tiles={len(result.x):,} "
                f"read_tile={int(result.read_tile_size_px)}px "
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
