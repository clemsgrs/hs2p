#!/usr/bin/env python3
"""
Benchmark hs2p tiling throughput (tiles/s) on a balanced slide sample as a
function of num_workers.

Reads slides from a CSV file (sample_id, image_path, optional mask_path,
optional spacing_at_level_0), optionally subsamples a balanced --n-slides set
stratified by file size, then runs the hs2p CLI for each worker count and
produces a publication-quality chart.

Usage
-----
    python scripts/benchmark_throughput.py \
        --csv /data/pathology/.../histai_mixed.csv \
        --n-slides 1000 \
        --workers 8 16 32 64 128 \
        --output-dir /tmp/hs2p-benchmark \
        --backend openslide

Regenerate chart only (skip benchmarking):
    python scripts/benchmark_throughput.py \
        --chart-only /tmp/hs2p-benchmark/results.csv
"""

import argparse
import csv
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark hs2p tiling throughput across num_workers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=False,
        help="CSV with columns: sample_id, image_path, (optional) mask_path, "
        "(optional) spacing_at_level_0.",
    )
    parser.add_argument(
        "--n-slides",
        type=int,
        default=1000,
        help="Number of balanced slides to sample from the CSV (stratified by file size). "
        "Set to 0 to use all slides.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="List of num_workers values to sweep.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of timed repetitions per worker count (results are averaged).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/hs2p-benchmark"),
        help="Directory for benchmark outputs (results CSV, chart, tiling artifacts).",
    )
    parser.add_argument(
        "--backend",
        default="asap",
        help="WSI reading backend passed to the tiling config.",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        default=0.5,
        help="Target spacing in µm/px.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size in pixels at target spacing.",
    )
    parser.add_argument(
        "--dataset-label",
        default="HistAI mixed",
        help="Dataset name shown in the chart subtitle and legend.",
    )
    parser.add_argument(
        "--collect-slide-metadata",
        action="store_true",
        default=False,
        help="Open each slide to collect median width/height stats for the chart subtitle (slow for large datasets).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("/tmp/benchmark-slides"),
        help="Destination directory when --copy-locally is set.",
    )
    parser.add_argument(
        "--copy-locally",
        action="store_true",
        help="Copy slides to --local-dir before benchmarking to eliminate network I/O.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to invoke the hs2p CLI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for slide sampling.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=0,
        help="Total number of slides in the full dataset. When set, a projection.txt "
        "file is written with estimated wall-clock times for each worker count. "
        "Set to 0 to skip.",
    )
    parser.add_argument(
        "--chart-only",
        type=Path,
        default=None,
        metavar="RESULTS_CSV",
        help="Skip benchmarking; regenerate chart from an existing results CSV.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Slide loading & sampling
# ---------------------------------------------------------------------------


def load_slides_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Read sample_id, image_path, (optional) mask_path, (optional) spacing_at_level_0."""
    slides: list[dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        has_mask = "mask_path" in fieldnames
        has_sample_id = "sample_id" in fieldnames
        has_spacing = "spacing_at_level_0" in fieldnames
        for row in reader:
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"]) if has_mask and row.get("mask_path") else None
            if mask_path and not mask_path.is_file():
                mask_path = None
            raw_spacing = row.get("spacing_at_level_0", "") if has_spacing else ""
            spacing_at_level_0 = float(raw_spacing) if raw_spacing.strip() else None
            sample_id = row["sample_id"] if has_sample_id else image_path.stem
            size_bytes = image_path.stat().st_size if image_path.is_file() else 0
            slides.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "spacing_at_level_0": spacing_at_level_0,
                    "size_bytes": size_bytes,
                }
            )
    return slides


def stratified_sample(
    slides: list[dict[str, Any]],
    n: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample n slides stratified into three equal size bins (small / medium / large)."""
    rng = random.Random(seed)
    if len(slides) <= n:
        return list(slides)

    sizes = [s["size_bytes"] for s in slides]
    q33 = float(np.percentile(sizes, 33))
    q66 = float(np.percentile(sizes, 66))

    small = [s for s in slides if s["size_bytes"] < q33]
    medium = [s for s in slides if q33 <= s["size_bytes"] < q66]
    large = [s for s in slides if s["size_bytes"] >= q66]

    per_bin = n // 3
    remainder = n - per_bin * 3

    sampled: list[dict[str, Any]] = []
    for i, bucket in enumerate([small, medium, large]):
        take = per_bin + (1 if i < remainder else 0)
        sampled.extend(rng.sample(bucket, min(take, len(bucket))))

    # top-up if any bucket was smaller than requested
    if len(sampled) < n:
        pool = [s for s in slides if s not in sampled]
        sampled.extend(rng.sample(pool, min(n - len(sampled), len(pool))))

    rng.shuffle(sampled)
    return sampled[:n]


def build_balanced_sample(
    slides: list[dict[str, Any]],
    *,
    n_slides: int,
    seed: int,
) -> list[dict[str, Any]]:
    return stratified_sample(slides, n_slides, seed=seed)[:n_slides]


# ---------------------------------------------------------------------------
# Local copy
# ---------------------------------------------------------------------------


def copy_slides_locally(
    slides: list[dict[str, Any]],
    local_dir: Path,
) -> list[dict[str, Any]]:
    """Copy slide (and mask) files to local_dir, return updated slide list.

    Files are named ``{sample_id}{original_suffix}`` to avoid collisions
    when slides from different directories share the same filename.
    Already-present files are skipped.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    updated: list[dict[str, Any]] = []
    n = len(slides)
    for i, slide in enumerate(slides, start=1):
        src_image = slide["image_path"]
        dst_image = local_dir / f"{slide['sample_id']}{src_image.suffix}"
        if not dst_image.exists():
            print(f"  [{i}/{n}] copying {src_image.name} ...", flush=True)
            shutil.copy2(src_image, dst_image)
        else:
            print(f"  [{i}/{n}] {dst_image.name} already present, skipping.", flush=True)

        dst_mask: Path | None = None
        if slide["mask_path"] is not None:
            src_mask = slide["mask_path"]
            dst_mask = local_dir / f"{slide['sample_id']}.mask{src_mask.suffix}"
            if not dst_mask.exists():
                shutil.copy2(src_mask, dst_mask)

        updated.append(
            {
                **slide,
                "image_path": dst_image,
                "mask_path": dst_mask,
                # size_bytes stays as-is (reflects original file)
            }
        )
    return updated


# ---------------------------------------------------------------------------
# Slide metadata
# ---------------------------------------------------------------------------


def collect_slide_stats(
    slides: list[dict[str, Any]],
    target_spacing: float,
    backend: str = "asap",
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Read basic slide metadata via WholeSlideImage and return summary statistics.

    Uses level_dimensions at the best level for target_spacing, which gives
    viewers a concrete sense of the data scale.  Returns an empty dict on failure.
    """
    from hs2p.wsi.wsi import WholeSlideImage

    widths: list[int] = []
    heights: list[int] = []
    for slide in slides[:1000]:
        try:
            wsi = WholeSlideImage(
                path=slide["image_path"],
                backend=backend,
                spacing_at_level_0=slide.get("spacing_at_level_0"),
            )
            level, _ = wsi.get_best_level_for_spacing(target_spacing, tolerance)
            w, h = wsi.level_dimensions[level]
            widths.append(w)
            heights.append(h)
        except Exception:
            continue

    if not widths:
        return {}
    return {
        "median_width_px": float(np.median(widths)),
        "median_height_px": float(np.median(heights)),
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def write_slides_csv(slides: list[dict[str, Any]], path: Path) -> None:
    """Write a slides CSV compatible with the hs2p CLI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    has_spacing = any(s.get("spacing_at_level_0") is not None for s in slides)
    fieldnames = ["sample_id", "image_path", "mask_path"]
    if has_spacing:
        fieldnames.append("spacing_at_level_0")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in slides:
            row: dict[str, Any] = {
                "sample_id": s["sample_id"],
                "image_path": str(s["image_path"]),
                "mask_path": str(s["mask_path"]) if s["mask_path"] else "",
            }
            if has_spacing:
                row["spacing_at_level_0"] = s["spacing_at_level_0"] or ""
            writer.writerow(row)


def write_config(
    *,
    csv_path: Path,
    output_dir: Path,
    num_workers: int,
    backend: str,
    target_spacing: float,
    tile_size: int,
    config_path: Path,
) -> None:
    """Write a minimal hs2p tiling config YAML for one benchmark run."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "csv": str(csv_path),
        "output_dir": str(output_dir),
        "resume": False,
        "visualize": False,
        "seed": 0,
        "tiling": {
            "read_tiles_from": None,
            "backend": backend,
            "params": {
                "target_spacing_um": target_spacing,
                "target_tile_size_px": tile_size,
                "tolerance": 0.05,
                "overlap": 0.0,
                "tissue_threshold": 0.25,
                "drop_holes": False,
                "use_padding": True,
            },
            "seg_params": {
                "downsample": 64,
                "sthresh": 8,
                "sthresh_up": 255,
                "mthresh": 7,
                "close": 4,
                "use_otsu": False,
                "use_hsv": True,
            },
            "filter_params": {
                "ref_tile_size": tile_size,
                "a_t": 4,
                "a_h": 2,
                "max_n_holes": 8,
                "filter_white": False,
                "filter_black": False,
                "white_threshold": 220,
                "black_threshold": 25,
                "fraction_threshold": 0.9,
            },
            "visu_params": {"downsample": 32},
            "sampling_params": {
                "independant_sampling": False,
                "pixel_mapping": [{"background": 0}, {"tissue": 1}],
                "color_mapping": [{"background": None}, {"tissue": [157, 219, 129]}],
                "tissue_percentage": [
                    {"background": None},
                    {"tissue": 0.01},
                ],
            },
        },
        "speed": {"num_workers": num_workers},
        "wandb": {"enable": False},
    }
    OmegaConf.save(config=OmegaConf.create(cfg), f=config_path)


def parse_process_list(path: Path) -> dict[str, Any]:
    """Read process_list.csv and return aggregate metrics."""
    if not path.is_file():
        return {"num_slides": 0, "total_tiles": 0, "slides_with_tiles": 0, "failed": 0}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    total_tiles = sum(int(float(r.get("num_tiles") or 0)) for r in rows)
    slides_with_tiles = sum(int(float(r.get("num_tiles") or 0)) > 0 for r in rows)
    failed = sum(r.get("tiling_status") == "failed" for r in rows)
    return {
        "num_slides": len(rows),
        "total_tiles": total_tiles,
        "slides_with_tiles": slides_with_tiles,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_one(
    *,
    slides: list[dict[str, Any]],
    num_workers: int,
    run_dir: Path,
    python_executable: str,
    backend: str,
    target_spacing: float,
    tile_size: int,
) -> dict[str, Any]:
    """Run the hs2p CLI once and return timing + tile count metrics."""
    run_dir.mkdir(parents=True, exist_ok=True)
    slides_csv = run_dir / "slides.csv"
    config_path = run_dir / "config.yaml"
    tiling_output = run_dir / "tiling"
    log_path = run_dir / "tiling.log"

    write_slides_csv(slides, slides_csv)
    write_config(
        csv_path=slides_csv,
        output_dir=tiling_output,
        num_workers=num_workers,
        backend=backend,
        target_spacing=target_spacing,
        tile_size=tile_size,
        config_path=config_path,
    )

    cmd = [
        python_executable,
        "-m",
        "hs2p.tiling",
        "--config-file",
        str(config_path),
        "--skip-datetime",
        "--skip-logging",
    ]
    t0 = time.perf_counter()
    with log_path.open("w") as log_fh:
        process = subprocess.Popen(
            cmd,
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:  # type: ignore[union-attr]
            print(line, end="", flush=True)
            log_fh.write(line)
        process.wait()
    elapsed = time.perf_counter() - t0
    result = process

    stats = parse_process_list(tiling_output / "process_list.csv")
    return {
        "num_workers": num_workers,
        "elapsed_seconds": elapsed,
        "exit_code": result.returncode,
        **stats,
        "tiles_per_second": stats["total_tiles"] / elapsed if elapsed > 0 else 0.0,
    }


def benchmark(
    *,
    slides: list[dict[str, Any]],
    workers: list[int],
    repeat: int,
    python_executable: str,
    backend: str,
    target_spacing: float,
    tile_size: int,
    output_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_runs = len(workers) * repeat
    run_idx = 0

    for nw in workers:
        rep_results: list[dict[str, Any]] = []
        for rep in range(1, repeat + 1):
            run_idx += 1
            print(
                f"  [{run_idx}/{total_runs}] workers={nw}, repeat={rep}/{repeat} ...",
                flush=True,
            )
            run_dir = output_root / "runs" / f"workers-{nw:02d}" / f"rep-{rep:02d}"
            result = run_one(
                slides=slides,
                num_workers=nw,
                run_dir=run_dir,
                python_executable=python_executable,
                backend=backend,
                target_spacing=target_spacing,
                tile_size=tile_size,
            )
            rep_results.append(result)
            status = "OK" if result["exit_code"] == 0 else f"exit={result['exit_code']}"
            print(
                f"    → {result['total_tiles']:,} tiles in "
                f"{result['elapsed_seconds']:.1f}s "
                f"({result['tiles_per_second']:,.0f} tiles/s)  [{status}]",
                flush=True,
            )

        mean_tps = float(np.mean([r["tiles_per_second"] for r in rep_results]))
        mean_elapsed = float(np.mean([r["elapsed_seconds"] for r in rep_results]))
        std_tps = (
            float(np.std([r["tiles_per_second"] for r in rep_results]))
            if repeat > 1
            else 0.0
        )
        records.append(
            {
                "num_workers": nw,
                "repeat": repeat,
                "mean_tiles_per_second": round(mean_tps, 2),
                "std_tiles_per_second": round(std_tps, 2),
                "mean_elapsed_seconds": round(mean_elapsed, 3),
                "total_tiles": rep_results[0]["total_tiles"],
                "slides_processed": rep_results[0]["num_slides"],
                "slides_with_tiles": rep_results[0]["slides_with_tiles"],
                "failed_slides": rep_results[0]["failed"],
            }
        )

    return records


# ---------------------------------------------------------------------------
# Results CSV I/O
# ---------------------------------------------------------------------------


def save_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def load_results_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "num_workers": int(row["num_workers"]),
                "mean_tiles_per_second": float(row["mean_tiles_per_second"]),
                "std_tiles_per_second": float(row["std_tiles_per_second"]),
                "total_tiles": int(row["total_tiles"]),
                "slides_processed": int(row["slides_processed"]),
            }
            for row in reader
        ]


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

# GitHub-dark-inspired palette
# Publication-ready color palette (light background, restrained)
_C_LINE   = "#1a6faf"   # main throughput line  – steel blue
_C_BAND   = "#a8c8e8"   # error band fill        – desaturated blue
_C_ACCENT = "#c0392b"   # speedup callout        – muted red
_C_TEXT   = "#222222"   # primary text
_C_MUTED  = "#666666"   # secondary text / axis labels
_C_GRID   = "#e8e8e8"   # horizontal gridlines


def _fmt_k(x: float, _pos: Any) -> str:
    """Format y-axis ticks as e.g. '24k'."""
    return f"{x / 1_000:.0f}k" if x >= 1_000 else f"{x:.0f}"


def plot_results(
    records: list[dict[str, Any]],
    *,
    output_path: Path,
    n_slides: int,
    target_spacing: float,
    tile_size: int,
    backend: str,
    dataset_label: str,
    slide_stats: dict[str, Any] | None = None,
) -> None:
    workers = [r["num_workers"] for r in records]
    tps     = [r["mean_tiles_per_second"] for r in records]
    err     = [r["std_tiles_per_second"]  for r in records]
    has_err = any(e > 0 for e in err)

    baseline     = tps[0]
    max_tps      = max(tps)
    max_workers  = workers[tps.index(max_tps)]
    # ------------------------------------------------------------------ style
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

    # subtle horizontal-only grid
    ax.yaxis.grid(True, color=_C_GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    # log₂ x-axis: worker counts are powers of 2 → evenly spaced
    ax.set_xscale("log", base=2)

    # y-axis focused on the actual data range
    y_min = min(tps) * 0.72   # floor slightly below lowest data point
    y_max = max_tps * 1.48

    # ------------------------------------------------------------------ error band
    if has_err:
        lower = [max(0.0, t - e) for t, e in zip(tps, err)]
        upper = [t + e for t, e in zip(tps, err)]
        ax.fill_between(workers, lower, upper, color=_C_BAND, alpha=0.35, zorder=3)

    # ------------------------------------------------------------------ main line
    ax.plot(
        workers, tps,
        color=_C_LINE, linewidth=2.4,
        solid_capstyle="round", solid_joinstyle="round",
        zorder=4,
    )

    # markers – filled dots with white centre ring
    for w, t in zip(workers, tps):
        ax.scatter(w, t, s=65, color=_C_LINE, zorder=6)
        ax.scatter(w, t, s=20, color="white",  zorder=7)

    # ------------------------------------------------------------------ value labels
    # on log₂ x, worker ticks are evenly spaced → simple offset above each point
    v_offset = y_max * 0.04
    for w, t in zip(workers, tps):
        ax.text(
            w, t + v_offset, f"{t / 1_000:.1f}k",
            ha="center", va="bottom",
            fontsize=8.5, color=_C_TEXT, fontweight="semibold",
        )

    # ------------------------------------------------------------------ axes
    ax.set_xlabel("Number of workers", fontsize=11, labelpad=10, color=_C_TEXT)
    ax.set_ylabel("Throughput  (tiles / s)", fontsize=11, labelpad=10, color=_C_TEXT)
    ax.set_xticks(workers)
    ax.set_xticklabels([str(w) for w in workers], fontsize=10, color=_C_TEXT)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
    ax.tick_params(axis="y", labelsize=10, colors=_C_TEXT)
    ax.tick_params(axis="x", colors=_C_TEXT)
    # small padding on log x-axis
    log_pad = 0.12
    ax.set_xlim(workers[0] * 2 ** (-log_pad), workers[-1] * 2 ** log_pad)
    ax.set_ylim(y_min, y_max)

    # ------------------------------------------------------------------ title block
    config_line = (
        f"{tile_size} px tiles · {target_spacing} µm/px · {backend} backend"
        f"  ·  {n_slides} {dataset_label} slides"
    )
    stats_parts: list[str] = []
    if slide_stats:
        if "median_width_px" in slide_stats and "median_height_px" in slide_stats:
            sw = int(slide_stats["median_width_px"])
            sh = int(slide_stats["median_height_px"])
            stats_parts.append(f"median {sw:,} × {sh:,} px at {target_spacing} µm/px")
        if "avg_tiles_per_slide" in slide_stats:
            stats_parts.append(f"avg {slide_stats['avg_tiles_per_slide']:,.0f} tiles/slide")
        if "median_file_size_mb" in slide_stats:
            stats_parts.append(f"median {slide_stats['median_file_size_mb']:.0f} MB/slide")

    fig.text(0.13, 0.97, "hs2p · Tiling Throughput",
             ha="left", va="top", fontsize=14, fontweight="bold", color=_C_TEXT)
    fig.text(0.13, 0.925, config_line,
             ha="left", va="top", fontsize=8.5, color=_C_MUTED)
    if stats_parts:
        fig.text(0.13, 0.895, "  ·  ".join(stats_parts),
                 ha="left", va="top", fontsize=8, color=_C_MUTED, alpha=0.85)

    # ------------------------------------------------------------------ save
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.13, right=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcdefaults()
    print(f"\nChart saved → {output_path}")


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def write_projection(
    records: list[dict[str, Any]],
    *,
    dataset_size: int,
    avg_tiles_per_slide: float,
    output_path: Path,
    dataset_label: str,
) -> None:
    """Write estimated full-dataset tiling times for each worker count."""
    total_tiles = dataset_size * avg_tiles_per_slide
    lines: list[str] = [
        f"Full-dataset projection  —  {dataset_label}",
        f"  Dataset size          : {dataset_size:,} slides",
        f"  Avg tiles / slide     : {avg_tiles_per_slide:,.0f}",
        f"  Estimated total tiles : {total_tiles:,.0f}",
        "",
        f"{'Workers':>10}  {'Throughput':>16}  {'Estimated time':>16}",
        "-" * 48,
    ]
    for r in records:
        tps = r["mean_tiles_per_second"]
        if tps <= 0:
            continue
        seconds = total_tiles / tps
        hours, rem = divmod(seconds, 3600)
        minutes = rem / 60
        lines.append(
            f"{r['num_workers']:>10}  {tps:>12,.0f} t/s  "
            f"{int(hours):>4}h {minutes:04.1f}m"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Projection saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # ── chart-only mode ───────────────────────────────────────────────────
    if args.chart_only is not None:
        print(f"Regenerating chart from {args.chart_only} ...")
        records = load_results_csv(args.chart_only)
        plot_results(
            records,
            output_path=args.chart_only.parent / "throughput.png",
            n_slides=records[0]["slides_processed"],
            target_spacing=args.target_spacing,
            tile_size=args.tile_size,
            backend=args.backend,
            dataset_label=args.dataset_label,
        )
        return 0

    # ── load slides ───────────────────────────────────────────────────────
    if args.csv is None:
        print("ERROR: provide --csv or --chart-only.", file=sys.stderr)
        return 1

    print(f"Loading slides from {args.csv} ...")
    all_slides = load_slides_from_csv(args.csv)
    print(f"Found {len(all_slides):,} slides in CSV.")

    if not all_slides:
        print("ERROR: CSV is empty.", file=sys.stderr)
        return 1

    n_target = args.n_slides if args.n_slides > 0 else len(all_slides)
    balanced = build_balanced_sample(all_slides, n_slides=n_target, seed=args.seed)
    print(f"Sampled {len(balanced)} balanced slides (seed={args.seed}).")
    sizes_mb = [s["size_bytes"] / 1e6 for s in balanced if s["size_bytes"] > 0]
    if sizes_mb:
        print(
            f"  balanced size range: {min(sizes_mb):.0f} MB – {max(sizes_mb):.0f} MB"
            f"  (median {float(np.median(sizes_mb)):.0f} MB)"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── copy slides locally ───────────────────────────────────────────────
    if args.copy_locally:
        total_mb = sum(s["size_bytes"] for s in balanced) / 1e6
        print(
            f"\nCopying {len(balanced)} slides to {args.local_dir}"
            f" (~{total_mb:.0f} MB) ...",
            flush=True,
        )
        local_lookup = {
            slide["sample_id"]: slide
            for slide in copy_slides_locally(balanced, args.local_dir)
        }
        balanced = [local_lookup.get(slide["sample_id"], slide) for slide in balanced]
        print("Copy complete.", flush=True)

    # ── collect slide metadata ────────────────────────────────────────────
    slide_stats: dict[str, Any] = {}
    if args.collect_slide_metadata:
        print("\nCollecting slide metadata ...", flush=True)
        slide_stats = collect_slide_stats(
            balanced, args.target_spacing, backend=args.backend
        )
        if "median_width_px" in slide_stats:
            print(
                f"  median slide size at {args.target_spacing}µm/px: "
                f"{int(slide_stats['median_width_px']):,} × {int(slide_stats['median_height_px']):,} px"
            )
        else:
            print("  (slide spacing metadata unavailable, skipping size stat)")
    sizes_mb_local = [s["size_bytes"] / 1e6 for s in balanced if s["size_bytes"] > 0]
    if sizes_mb_local:
        slide_stats["median_file_size_mb"] = float(np.median(sizes_mb_local))

    # ── benchmark ────────────────────────────────────────────────────────
    print(f"\nRunning benchmark: workers={sorted(args.workers)}, repeat={args.repeat}")
    print(f"Python: {args.python}")
    records = benchmark(
        slides=balanced,
        workers=sorted(args.workers),
        repeat=args.repeat,
        python_executable=args.python,
        backend=args.backend,
        target_spacing=args.target_spacing,
        tile_size=args.tile_size,
        output_root=args.output_dir,
    )

    # ── save & report ─────────────────────────────────────────────────────
    results_path = args.output_dir / "results.csv"
    save_csv(records, results_path)
    print(f"\nResults saved → {results_path}")

    print("\nSummary")
    print("-------")
    for r in records:
        print(
            f"  workers={r['num_workers']:>3}  "
            f"{r['mean_tiles_per_second']:>10,.0f} tiles/s"
            f"  (elapsed {r['mean_elapsed_seconds']:.1f}s)"
            + (f"  ⚠ {r['failed_slides']} failed" if r["failed_slides"] else "")
        )

    # ── projection ────────────────────────────────────────────────────────
    avg_tps = records[0]["total_tiles"] / max(records[0]["slides_processed"], 1)
    if args.dataset_size > 0:
        write_projection(
            records,
            dataset_size=args.dataset_size,
            avg_tiles_per_slide=avg_tps,
            output_path=args.output_dir / "projection.txt",
            dataset_label=args.dataset_label,
        )

    # ── chart ─────────────────────────────────────────────────────────────
    chart_slide_stats = dict(slide_stats)
    chart_slide_stats["avg_tiles_per_slide"] = records[0]["total_tiles"] / max(
        records[0]["slides_processed"], 1
    )
    plot_results(
        records,
        output_path=args.output_dir / "throughput.png",
        n_slides=len(balanced),
        target_spacing=args.target_spacing,
        tile_size=args.tile_size,
        backend=args.backend,
        dataset_label=args.dataset_label,
        slide_stats=chart_slide_stats,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
