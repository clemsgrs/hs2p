#!/usr/bin/env python3
"""Benchmark tile-store creation with read/encode/write time breakdown.

Replicates the ``extract_tiles_to_tar`` pipeline from ``hs2p.api`` with
per-phase timing instrumentation so that read, JPEG-encode, and tar-write
costs can be measured independently.
"""
from __future__ import annotations

import argparse
import io
import statistics
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hs2p.api import _iter_tile_arrays_for_tar_extraction

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
    return parser.parse_args()


# ------------------------------------------------------------------
# Core benchmark
# ------------------------------------------------------------------


def benchmark_tile_store(
    *,
    result,
    jpeg_quality: int,
    num_workers: int,
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, float | int]:
    """Run one extraction pass and return per-phase timing metrics."""
    import turbojpeg
    from PIL import Image

    _jpeg_encoder = turbojpeg.TurboJPEG()

    output_dir.mkdir(parents=True, exist_ok=True)

    read_s = 0.0
    encode_s = 0.0
    write_s = 0.0
    tile_count = 0
    jpeg_bytes = 0

    temp_tar_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".tar", dir=output_dir, delete=False
        ) as tmp:
            temp_tar_path = Path(tmp.name)

        total_start = time.perf_counter()

        with tarfile.open(temp_tar_path, "w") as tf:
            iterator = _iter_tile_arrays_for_tar_extraction(
                result=result, num_workers=num_workers,
            )

            while True:
                t0 = time.perf_counter()
                try:
                    tile_arr = next(iterator)
                except StopIteration:
                    break
                t1 = time.perf_counter()
                read_s += t1 - t0

                if tile_arr.shape[2] > 3:
                    tile_arr = tile_arr[:, :, :3]

                if result.read_tile_size_px != result.target_tile_size_px:
                    img = Image.fromarray(tile_arr).convert("RGB")
                    img = img.resize(
                        (result.target_tile_size_px, result.target_tile_size_px),
                        Image.LANCZOS,
                    )
                    tile_arr = np.asarray(img)

                encoded = _jpeg_encoder.encode(
                    tile_arr, quality=jpeg_quality,
                )
                buf = io.BytesIO(encoded)
                t2 = time.perf_counter()
                encode_s += t2 - t1

                info = tarfile.TarInfo(name=f"{tile_count:06d}.jpg")
                info.size = len(encoded)
                tf.addfile(info, buf)
                t3 = time.perf_counter()
                write_s += t3 - t2

                jpeg_bytes += info.size
                tile_count += 1
                if progress_callback is not None:
                    progress_callback(1, 1)

        total_s = time.perf_counter() - total_start
    finally:
        if temp_tar_path is not None:
            temp_tar_path.unlink(missing_ok=True)

    return {
        "tile_count": tile_count,
        "jpeg_bytes": jpeg_bytes,
        "read_s": read_s,
        "encode_s": encode_s,
        "write_s": write_s,
        "total_s": total_s,
    }


# ------------------------------------------------------------------
# Result formatting
# ------------------------------------------------------------------


def build_result_row(
    *,
    sample_id: str,
    image_path: str,
    repeat_index: int,
    tiles: int,
    jpeg_quality: int,
    num_workers: int,
    read_s: float,
    encode_s: float,
    write_s: float,
    total_s: float,
    jpeg_bytes: int,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "repeat_index": repeat_index,
        "tiles": tiles,
        "jpeg_quality": jpeg_quality,
        "num_workers": num_workers,
        "read_s": round(read_s, 6),
        "encode_s": round(encode_s, 6),
        "write_s": round(write_s, 6),
        "total_s": round(total_s, 6),
        "read_pct": round(100 * read_s / total_s, 2) if total_s > 0 else 0.0,
        "encode_pct": round(100 * encode_s / total_s, 2) if total_s > 0 else 0.0,
        "write_pct": round(100 * write_s / total_s, 2) if total_s > 0 else 0.0,
        "tiles_per_second": round(tiles / total_s, 2) if total_s > 0 else 0.0,
        "jpeg_bytes": jpeg_bytes,
        "jpeg_mb_per_second": round(
            (jpeg_bytes / 1_000_000) / total_s, 2
        ) if total_s > 0 else 0.0,
    }


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    def _mean(vals):
        return round(statistics.mean(vals), 6)

    def _pstdev(vals):
        return round(statistics.pstdev(vals), 6) if len(vals) > 1 else 0.0

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["num_workers"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for nw, nw_rows in grouped.items():
        read_s = [float(r["read_s"]) for r in nw_rows]
        encode_s = [float(r["encode_s"]) for r in nw_rows]
        write_s = [float(r["write_s"]) for r in nw_rows]
        total_s = [float(r["total_s"]) for r in nw_rows]
        read_pct = [float(r["read_pct"]) for r in nw_rows]
        encode_pct = [float(r["encode_pct"]) for r in nw_rows]
        write_pct = [float(r["write_pct"]) for r in nw_rows]
        tps = [float(r["tiles_per_second"]) for r in nw_rows]
        summary.append(
            {
                "num_workers": nw,
                "tiles": int(nw_rows[0]["tiles"]),
                "jpeg_quality": int(nw_rows[0]["jpeg_quality"]),
                "mean_read_s": _mean(read_s),
                "mean_encode_s": _mean(encode_s),
                "mean_write_s": _mean(write_s),
                "mean_total_s": _mean(total_s),
                "mean_read_pct": round(statistics.mean(read_pct), 2),
                "mean_encode_pct": round(statistics.mean(encode_pct), 2),
                "mean_write_pct": round(statistics.mean(write_pct), 2),
                "mean_tiles_per_second": round(statistics.mean(tps), 2),
                "std_tiles_per_second": round(_pstdev(tps), 2),
            }
        )
    return summary


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
    fig.text(0.13, 0.925, f"JPEG quality={jpeg_quality}  ·  tiles={tiles_label}",
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
    from benchmark_tile_read_strategies import (
        BenchmarkProgressReporter,
        load_single_slide_result_from_config,
        write_csv,
    )
    from hs2p.benchmarking import limit_tiling_result

    args = parse_args()
    workers = sorted(args.workers)

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
                    total_read_calls=int(result.num_tiles),
                    total_tiles=int(result.num_tiles),
                )
                benchmark_tile_store(
                    result=result,
                    jpeg_quality=int(args.jpeg_quality),
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
                    total_read_calls=int(result.num_tiles),
                    total_tiles=int(result.num_tiles),
                )
                metrics = benchmark_tile_store(
                    result=result,
                    jpeg_quality=int(args.jpeg_quality),
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
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
