#!/usr/bin/env python3
"""
Benchmark hs2p tiling throughput (tiles/s) as a function of num_workers.

Reads slides from a CSV file (sample_id, image_path, optional mask_path,
optional spacing_at_level_0), optionally subsamples --n-slides slides
stratified by file size, then runs the hs2p CLI for each worker count and
produces a publication-quality chart.

Usage
-----
    python scripts/benchmark_throughput.py \
        --csv /data/pathology/.../histai_mixed.csv \
        --n-slides 100 \
        --workers 4 8 16 32 \
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
        default=100,
        help="Number of slides to sample from the CSV (stratified by file size). "
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
        default="openslide",
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
        "--total-slides",
        type=int,
        default=52691,
        help="Total slide count for the projection annotation on the chart.",
    )
    parser.add_argument(
        "--dataset-label",
        default="HistAI mixed",
        help="Dataset name shown in the chart subtitle and legend.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("/tmp/benchmark-slides"),
        help="Local directory to copy slides into before benchmarking, "
        "eliminating network I/O from timing measurements.",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip copying slides to --local-dir (use if already copied).",
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
) -> dict[str, Any]:
    """Read basic slide metadata via openslide and return summary statistics.

    Computes the slide area in megapixels at the target spacing, which gives
    viewers a concrete sense of the data scale.  Returns an empty dict if
    openslide is not available or no spacing metadata can be read.
    """
    try:
        import openslide
    except ImportError:
        return {}

    mpx_at_target: list[float] = []
    for slide in slides:
        try:
            wsi = openslide.OpenSlide(str(slide["image_path"]))
            w, h = wsi.dimensions
            mpp_x = float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_X) or 0)
            wsi.close()
            if mpp_x > 0 and target_spacing > 0:
                scale = mpp_x / target_spacing
                mpx_at_target.append(w * scale * h * scale / 1e6)
        except Exception:
            continue

    if not mpx_at_target:
        return {}
    return {"median_mpx_at_target": float(np.median(mpx_at_target))}


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
                "tissue_threshold": 0.01,
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
                "ref_tile_size": 16,
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
        result = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - t0

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
_BG = "#0d1117"
_AXES_BG = "#161b22"
_GRID = "#21262d"
_TEXT = "#e6edf3"
_TEXT_MUTED = "#8b949e"
_LINE = "#58a6ff"
_IDEAL = "#3fb950"
_ANNOT_BG = "#21262d"
_ANNOT_BORDER = "#30363d"
_ACCENT = "#f78166"


def _format_kilo(x: float, _pos: Any) -> str:
    if x >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{x:.0f}"


def plot_results(
    records: list[dict[str, Any]],
    *,
    output_path: Path,
    n_slides: int,
    target_spacing: float,
    tile_size: int,
    backend: str,
    total_slides: int,
    dataset_label: str,
    slide_stats: dict[str, Any] | None = None,
) -> None:
    workers = [r["num_workers"] for r in records]
    tps = [r["mean_tiles_per_second"] for r in records]
    err = [r["std_tiles_per_second"] for r in records]
    has_error = any(e > 0 for e in err)

    # ideal linear scaling anchored at the lowest worker count
    baseline = tps[0]
    baseline_workers = workers[0]
    ideal = [baseline * w / baseline_workers for w in workers]

    # projection: observed avg tiles/slide extrapolated to full dataset
    avg_tiles_per_slide = records[0]["total_tiles"] / max(records[0]["slides_processed"], 1)
    max_tps = max(tps)
    max_workers = workers[tps.index(max_tps)]
    projected_hours = avg_tiles_per_slide * total_slides / max_tps / 3600

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_AXES_BG)

    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.tick_params(colors=_TEXT, which="both")
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)

    ax.grid(True, color=_GRID, linewidth=0.8, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # ideal linear scaling reference
    ax.plot(
        workers,
        ideal,
        color=_IDEAL,
        linewidth=1.5,
        linestyle="--",
        alpha=0.55,
        label="Ideal linear scaling",
        zorder=2,
    )

    # main throughput line
    ax.plot(
        workers,
        tps,
        color=_LINE,
        linewidth=3,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=4,
        label=f"hs2p  ({n_slides} {dataset_label} slides)",
    )
    if has_error:
        lower = [max(0.0, t - e) for t, e in zip(tps, err)]
        upper = [t + e for t, e in zip(tps, err)]
        ax.fill_between(workers, lower, upper, color=_LINE, alpha=0.18, zorder=3)

    # markers + value labels
    for w, t in zip(workers, tps):
        ax.scatter(w, t, color=_LINE, s=90, zorder=5, edgecolors=_BG, linewidths=1.5)
        ax.annotate(
            f"{t:,.0f}",
            xy=(w, t),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=_TEXT,
        )

    # speedup annotation (min → max workers)
    if len(workers) > 1:
        speedup = max_tps / baseline
        mid_x = (workers[0] + max_workers) / 2
        mid_y = (baseline + max_tps) / 2
        ax.annotate(
            f"{speedup:.1f}×",
            xy=(max_workers, max_tps),
            xytext=(mid_x, mid_y * 0.82),
            textcoords="data",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=_ACCENT,
            arrowprops=dict(
                arrowstyle="-",
                color=_ACCENT,
                lw=1.2,
                linestyle="dotted",
            ),
        )

    # projection annotation box
    ax.text(
        0.985,
        0.05,
        f"Projected · {dataset_label} (~{total_slides:,} slides)\n"
        f"At {max_workers} workers:  {projected_hours:.1f} hours",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=_TEXT,
        linespacing=1.7,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor=_ANNOT_BG,
            edgecolor=_ANNOT_BORDER,
            linewidth=1.2,
        ),
    )

    # ------------------------------------------------------------------ axes
    ax.set_xlabel("Number of workers", fontsize=14, labelpad=10)
    ax.set_ylabel("Throughput  (tiles / second)", fontsize=14, labelpad=10)
    ax.set_xticks(workers)
    ax.set_xticklabels([str(w) for w in workers], fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_kilo))
    ax.tick_params(axis="y", labelsize=12)
    x_pad = (workers[-1] - workers[0]) * 0.06
    ax.set_xlim(workers[0] - x_pad, workers[-1] + x_pad)
    ax.set_ylim(0, max(max(ideal), max(tps)) * 1.22)

    # ------------------------------------------------------------------ title
    fig.text(
        0.08, 0.96,
        "hs2p · Tiling Throughput",
        ha="left", va="top",
        fontsize=18, fontweight="bold", color=_TEXT,
    )
    fig.text(
        0.08, 0.91,
        f"{tile_size}px tiles · {target_spacing}µm/px · {backend} backend"
        f"  ·  {n_slides} {dataset_label} slides",
        ha="left", va="top",
        fontsize=11, color=_TEXT_MUTED,
    )

    # slide stats line (shown only when metadata is available)
    if slide_stats:
        stats_parts: list[str] = []
        if "median_mpx_at_target" in slide_stats:
            mpx = slide_stats["median_mpx_at_target"]
            stats_parts.append(f"median {mpx:,.0f} Mpx at {target_spacing}µm/px")
        if "avg_tiles_per_slide" in slide_stats:
            stats_parts.append(f"avg {slide_stats['avg_tiles_per_slide']:,.0f} tiles/slide")
        if "median_file_size_mb" in slide_stats:
            stats_parts.append(f"median {slide_stats['median_file_size_mb']:.0f} MB/slide")
        if stats_parts:
            fig.text(
                0.08, 0.875,
                "  ·  ".join(stats_parts),
                ha="left", va="top",
                fontsize=10, color=_TEXT_MUTED,
                alpha=0.8,
            )

    # ------------------------------------------------------------------ legend
    legend = ax.legend(
        fontsize=11,
        loc="upper left",
        framealpha=0.0,
        labelcolor=_TEXT,
        handlelength=2.2,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.5)

    # ------------------------------------------------------------------ footer
    fig.text(
        0.08, 0.025,
        "github.com/clemsgrs/hs2p",
        ha="left", va="bottom",
        fontsize=9.5, color=_TEXT_MUTED, style="italic",
    )

    # ------------------------------------------------------------------ save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"\nChart saved → {output_path}")


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
            total_slides=args.total_slides,
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
    sampled = stratified_sample(all_slides, n_target, seed=args.seed)
    print(f"Sampled {len(sampled)} slides (stratified by file size, seed={args.seed}).")
    sizes_mb = [s["size_bytes"] / 1e6 for s in sampled if s["size_bytes"] > 0]
    if sizes_mb:
        print(
            f"  size range: {min(sizes_mb):.0f} MB – {max(sizes_mb):.0f} MB"
            f"  (median {float(np.median(sizes_mb)):.0f} MB)"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── copy slides locally ───────────────────────────────────────────────
    if args.skip_copy:
        print(f"\nSkipping copy (--skip-copy). Expecting slides in {args.local_dir}.")
        # Remap paths to local_dir using the same naming convention.
        remapped = []
        for s in sampled:
            dst_image = args.local_dir / f"{s['sample_id']}{s['image_path'].suffix}"
            dst_mask: Path | None = None
            if s["mask_path"] is not None:
                dst_mask = args.local_dir / f"{s['sample_id']}.mask{s['mask_path'].suffix}"
            remapped.append({**s, "image_path": dst_image, "mask_path": dst_mask})
        sampled = remapped
    else:
        total_mb = sum(s["size_bytes"] for s in sampled) / 1e6
        print(
            f"\nCopying {len(sampled)} slides to {args.local_dir}"
            f" (~{total_mb:.0f} MB) ...",
            flush=True,
        )
        sampled = copy_slides_locally(sampled, args.local_dir)
        print("Copy complete.", flush=True)

    # ── collect slide metadata ────────────────────────────────────────────
    print("\nCollecting slide metadata ...", flush=True)
    slide_stats: dict[str, Any] = collect_slide_stats(sampled, args.target_spacing)
    sizes_mb_local = [s["size_bytes"] / 1e6 for s in sampled if s["size_bytes"] > 0]
    if sizes_mb_local:
        slide_stats["median_file_size_mb"] = float(np.median(sizes_mb_local))
    if "median_mpx_at_target" in slide_stats:
        print(f"  median slide area at {args.target_spacing}µm/px: {slide_stats['median_mpx_at_target']:,.0f} Mpx")
    else:
        print("  (slide spacing metadata unavailable, skipping Mpx stat)")

    # ── benchmark ────────────────────────────────────────────────────────
    print(f"\nRunning benchmark: workers={sorted(args.workers)}, repeat={args.repeat}")
    print(f"Python: {args.python}")
    records = benchmark(
        slides=sampled,
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
            f"  workers={r['num_workers']:>2}  "
            f"{r['mean_tiles_per_second']:>10,.0f} tiles/s"
            f"  (elapsed {r['mean_elapsed_seconds']:.1f}s)"
            + (f"  ⚠ {r['failed_slides']} failed" if r["failed_slides"] else "")
        )

    # ── chart ─────────────────────────────────────────────────────────────
    avg_tiles = records[0]["total_tiles"] / max(records[0]["slides_processed"], 1)
    slide_stats["avg_tiles_per_slide"] = avg_tiles
    plot_results(
        records,
        output_path=args.output_dir / "throughput.png",
        n_slides=len(sampled),
        target_spacing=args.target_spacing,
        tile_size=args.tile_size,
        backend=args.backend,
        total_slides=args.total_slides,
        dataset_label=args.dataset_label,
        slide_stats=slide_stats,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
