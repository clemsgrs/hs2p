#!/usr/bin/env python3
"""Benchmark tile-store creation with read/encode/write time breakdown.

Replicates the ``extract_tiles_to_tar`` pipeline from ``hs2p.api`` with
per-phase timing instrumentation so that read, JPEG-encode, and tar-write
costs can be measured independently.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hs2p.api import extract_tiles_to_tar
from scripts.benchmark_tile_store_support import (
    ExtractionPhaseRecorder,
    build_result_row,
    resolve_jpeg_backend,
    summarize_results,
)

ProgressCallback = Callable[[int, int], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark tile-store creation (read/encode/write breakdown) "
            "from a fresh tiling result generated from an hs2p config file."
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
        "--repeat",
        type=int,
        default=1,
        help="Number of timed repetitions.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Untimed warmup repetitions.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Use only the first N tiles. Set to 0 to use all tiles.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="Worker counts to sweep.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG encoding quality (1-100).",
    )
    parser.add_argument(
        "--jpeg-backend",
        choices=("turbojpeg", "pil"),
        default=argparse.SUPPRESS,
        help="JPEG encoder used for the benchmark only.",
    )
    parser.add_argument(
        "--supertile-sizes",
        type=int,
        nargs="+",
        default=[8, 4, 2],
        help="Supertile sizes to allow in the read planner, largest first.",
    )
    return parser.parse_args()

def benchmark_tile_store(
    *,
    result,
    jpeg_quality: int,
    jpeg_backend: str = "turbojpeg",
    supertile_sizes: tuple[int, ...] | None = None,
    num_workers: int,
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, float | int]:
    """Run one extraction pass and return per-phase timing metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = ExtractionPhaseRecorder(progress_callback=progress_callback)

    with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
        total_start = time.perf_counter()
        extract_tiles_to_tar(
            result,
            output_dir=Path(tmpdir),
            jpeg_quality=int(jpeg_quality),
            jpeg_backend=str(jpeg_backend),
            supertile_sizes=supertile_sizes,
            num_workers=int(num_workers),
            phase_recorder=recorder,
        )
        total_s = time.perf_counter() - total_start
    total_s = max(total_s, recorder.read_s + recorder.encode_s + recorder.write_s)

    return {
        "tile_count": recorder.tile_count,
        "jpeg_bytes": recorder.jpeg_bytes,
        "read_s": recorder.read_s,
        "encode_s": recorder.encode_s,
        "write_s": recorder.write_s,
        "total_s": total_s,
    }

# ------------------------------------------------------------------
# Chart
# ------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

_C_LINE   = "#1a6faf"  # main line  – steel blue
_C_BAND   = "#a8c8e8"  # error band – desaturated blue
_C_TEXT   = "#222222"  # primary text
_C_MUTED  = "#666666"  # secondary text
_C_GRID   = "#e8e8e8"  # gridlines


def plot_results(
    summary_rows: list[dict[str, Any]],
    *,
    output_path: Path,
    max_tiles: int,
    jpeg_quality: int,
    jpeg_backend: str,
) -> None:
    workers = [int(r["num_workers"]) for r in summary_rows]
    tps     = [float(r["mean_tiles_per_second"]) for r in summary_rows]
    err     = [float(r["std_tiles_per_second"]) for r in summary_rows]
    has_err = any(e > 0 for e in err)

    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.yaxis.grid(True, color=_C_GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    ax.set_xscale("log", base=2)

    y_min = min(tps) * 0.72
    y_max = max(tps) * 1.48

    if has_err:
        lower = [max(0.0, t - e) for t, e in zip(tps, err)]
        upper = [t + e for t, e in zip(tps, err)]
        ax.fill_between(workers, lower, upper, color=_C_BAND, alpha=0.35, zorder=3)

    ax.plot(workers, tps, color=_C_LINE, linewidth=2.4,
            solid_capstyle="round", solid_joinstyle="round", zorder=4)

    for w, t in zip(workers, tps):
        ax.scatter(w, t, s=65, color=_C_LINE, zorder=6)
        ax.scatter(w, t, s=20, color="white",  zorder=7)

    v_offset = y_max * 0.04
    for w, t in zip(workers, tps):
        ax.text(w, t + v_offset, f"{t:,.0f}",
                ha="center", va="bottom", fontsize=8.5,
                color=_C_TEXT, fontweight="semibold")

    ax.set_xlabel("Number of workers", fontsize=11, labelpad=10, color=_C_TEXT)
    ax.set_ylabel("Throughput  (tiles / s)", fontsize=11, labelpad=10, color=_C_TEXT)
    ax.set_xticks(workers)
    ax.set_xticklabels([str(w) for w in workers], fontsize=10, color=_C_TEXT)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", labelsize=10, colors=_C_TEXT)
    ax.tick_params(axis="x", colors=_C_TEXT)
    log_pad = 0.30
    ax.set_xlim(workers[0] * 2 ** (-log_pad), workers[-1] * 2 ** log_pad)
    ax.set_ylim(y_min, y_max)

    tiles_label = f"{max_tiles:,}" if max_tiles > 0 else "all"
    fig.text(0.13, 0.97, "Tile Store Throughput",
             ha="left", va="top", fontsize=14, fontweight="bold", color=_C_TEXT)
    fig.text(
        0.13,
        0.925,
        f"JPEG backend={jpeg_backend}  ·  quality={jpeg_quality}  ·  tiles={tiles_label}",
             ha="left", va="top", fontsize=8.5, color=_C_MUTED)

    # inset table: read/encode/write breakdown per worker count
    tbl_ax = fig.add_axes([0.60, 0.845, 0.37, 0.145])
    tbl_ax.axis("off")
    table_data = [
        [str(int(r["num_workers"])),
         f"{float(r['mean_read_pct']):.0f}%",
         f"{float(r['mean_encode_pct']):.0f}%",
         f"{float(r['mean_write_pct']):.0f}%"]
        for r in summary_rows
    ]
    tbl = tbl_ax.table(
        cellText=table_data,
        colLabels=["workers", "read", "encode", "write"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor(_C_GRID)
        if row == 0:
            cell.set_text_props(color=_C_MUTED, fontweight="semibold")
            cell.set_facecolor("#f4f8fc")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color=_C_MUTED)
        cell.set_height(0.155)

    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.13, right=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcdefaults()
    print(f"Chart saved → {output_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> int:
    from scripts.benchmark_tile_read import (
        BenchmarkProgressReporter,
        load_single_slide_result_from_config,
        write_csv,
    )
    from scripts.benchmark_tile_utils import limit_tiling_result

    args = parse_args()
    workers = sorted(args.workers)
    jpeg_backend = resolve_jpeg_backend(
        config_file=args.config_file,
        cli_jpeg_backend=getattr(args, "jpeg_backend", None),
    )
    supertile_sizes = tuple(int(size) for size in args.supertile_sizes)

    result = load_single_slide_result_from_config(
        config_file=args.config_file,
        num_workers=workers[0],
    )
    result = limit_tiling_result(result, max_tiles=int(args.max_tiles))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(workers) * (int(args.warmup) + int(args.repeat))

    timed_rows: list[dict[str, Any]] = []
    run_counter = 0
    with BenchmarkProgressReporter(total_runs=total_runs) as reporter:
        reporter.print_banner(
            result=result,
            modes=[f"workers={w}" for w in workers],
            repeat=int(args.repeat),
            warmup=int(args.warmup),
        )
        for nw in workers:
            mode_label = f"tile_store w={nw}"
            for warmup_idx in range(int(args.warmup)):
                run_counter += 1
                reporter.start_run(
                    run_counter=run_counter,
                    phase="warmup",
                    mode=mode_label,
                    iteration_index=warmup_idx,
                    iteration_total=int(args.warmup),
                    total_read_calls=len(result.coordinates),
                    total_tiles=len(result.coordinates),
                )
                benchmark_tile_store(
                    result=result,
                    jpeg_quality=int(args.jpeg_quality),
                    jpeg_backend=jpeg_backend,
                    supertile_sizes=supertile_sizes,
                    num_workers=nw,
                    output_dir=args.output_dir,
                    progress_callback=reporter.advance,
                )
                reporter.finish_run()

            for repeat_index in range(int(args.repeat)):
                run_counter += 1
                reporter.start_run(
                    run_counter=run_counter,
                    phase="timed",
                    mode=mode_label,
                    iteration_index=repeat_index,
                    iteration_total=int(args.repeat),
                    total_read_calls=len(result.coordinates),
                    total_tiles=len(result.coordinates),
                )
                metrics = benchmark_tile_store(
                    result=result,
                    jpeg_quality=int(args.jpeg_quality),
                    jpeg_backend=jpeg_backend,
                    supertile_sizes=supertile_sizes,
                    num_workers=nw,
                    output_dir=args.output_dir,
                    progress_callback=reporter.advance,
                )
                reporter.finish_run()

                row = build_result_row(
                    sample_id=str(result.sample_id),
                    image_path=str(result.image_path),
                    repeat_index=repeat_index,
                    tiles=metrics["tile_count"],
                    jpeg_quality=int(args.jpeg_quality),
                    jpeg_backend=jpeg_backend,
                    num_workers=nw,
                    read_s=metrics["read_s"],
                    encode_s=metrics["encode_s"],
                    write_s=metrics["write_s"],
                    total_s=metrics["total_s"],
                    jpeg_bytes=metrics["jpeg_bytes"],
                )
                reporter.console.print(
                    (
                        f"workers={nw:<3} rep={repeat_index + 1} "
                        f"tiles={int(row['tiles']):>7,d} "
                        f"read={float(row['read_pct']):>5.1f}% "
                        f"encode={float(row['encode_pct']):>5.1f}% "
                        f"write={float(row['write_pct']):>5.1f}% "
                        f"elapsed={float(row['total_s']):>8.3f}s "
                        f"throughput={float(row['tiles_per_second']):>10,.0f} tiles/s"
                    ),
                    highlight=False,
                )
                timed_rows.append(row)

    summary_rows = summarize_results(timed_rows)
    runs_csv_path = write_csv(timed_rows, args.output_dir / "benchmark_runs.csv")
    summary_csv_path = write_csv(summary_rows, args.output_dir / "benchmark_summary.csv")

    print(f"\nWrote {runs_csv_path}", flush=True)
    print(f"Wrote {summary_csv_path}", flush=True)

    if len(summary_rows) > 1:
        chart_path = args.output_dir / "throughput.png"
        plot_results(
            summary_rows,
            output_path=chart_path,
            max_tiles=int(args.max_tiles),
            jpeg_quality=int(args.jpeg_quality),
            jpeg_backend=jpeg_backend,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
