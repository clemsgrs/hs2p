#!/usr/bin/env python3
"""
Benchmark hs2p tiling throughput (tiles/s) as a function of num_workers.

Reads slides from a CSV file (sample_id, image_path, optional mask_path),
optionally subsamples --n-slides slides stratified by file size, runs
tile_slides() for each worker count, then saves a CSV of raw results and
produces a publication-quality chart.

Usage
-----
    python scripts/benchmark_throughput.py \
        --csv /data/pathology/projects/clement/notebooks/hs2p/histai_mixed.csv \
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
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from hs2p import (  # noqa: E402
    FilterConfig,
    SegmentationConfig,
    SlideSpec,
    TilingConfig,
    tile_slides,
)


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
        help="CSV with columns: sample_id, image_path, (optional) mask_path.",
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
        help="WSI reading backend passed to TilingConfig.",
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
# Benchmark
# ---------------------------------------------------------------------------


def build_configs(
    *,
    backend: str,
    target_spacing: float,
    tile_size: int,
) -> tuple[TilingConfig, SegmentationConfig, FilterConfig]:
    tiling = TilingConfig(
        backend=backend,
        target_spacing_um=target_spacing,
        target_tile_size_px=tile_size,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.01,
        drop_holes=False,
        use_padding=True,
    )
    return tiling, SegmentationConfig(), FilterConfig()


def run_one(
    *,
    whole_slides: list[SlideSpec],
    tiling: TilingConfig,
    segmentation: SegmentationConfig,
    filtering: FilterConfig,
    num_workers: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run tile_slides once and return timing + tile count metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    artifacts = tile_slides(
        whole_slides,
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        output_dir=output_dir,
        num_workers=num_workers,
        resume=False,
    )
    elapsed = time.perf_counter() - t0

    total_tiles = sum(a.num_tiles for a in artifacts)
    successful = sum(1 for a in artifacts if a.num_tiles > 0)
    return {
        "num_workers": num_workers,
        "elapsed_seconds": elapsed,
        "total_tiles": total_tiles,
        "tiles_per_second": total_tiles / elapsed if elapsed > 0 else 0.0,
        "slides_processed": len(artifacts),
        "slides_with_tiles": successful,
    }


def benchmark(
    *,
    slides: list[dict[str, Any]],
    workers: list[int],
    repeat: int,
    backend: str,
    target_spacing: float,
    tile_size: int,
    output_root: Path,
) -> list[dict[str, Any]]:
    tiling, segmentation, filtering = build_configs(
        backend=backend,
        target_spacing=target_spacing,
        tile_size=tile_size,
    )
    whole_slides = [
        SlideSpec(
            sample_id=s["sample_id"],
            image_path=s["image_path"],
            mask_path=s["mask_path"],
            spacing_at_level_0=s.get("spacing_at_level_0"),
        )
        for s in slides
    ]

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
            run_dir = output_root / "tiling" / f"workers-{nw:02d}" / f"rep-{rep:02d}"
            result = run_one(
                whole_slides=whole_slides,
                tiling=tiling,
                segmentation=segmentation,
                filtering=filtering,
                num_workers=nw,
                output_dir=run_dir,
            )
            rep_results.append(result)
            print(
                f"    → {result['total_tiles']:,} tiles in "
                f"{result['elapsed_seconds']:.1f}s "
                f"({result['tiles_per_second']:,.0f} tiles/s)",
                flush=True,
            )

        mean_tps = float(np.mean([r["tiles_per_second"] for r in rep_results]))
        mean_elapsed = float(np.mean([r["elapsed_seconds"] for r in rep_results]))
        std_tps = float(np.std([r["tiles_per_second"] for r in rep_results])) if repeat > 1 else 0.0

        records.append(
            {
                "num_workers": nw,
                "repeat": repeat,
                "mean_tiles_per_second": round(mean_tps, 2),
                "std_tiles_per_second": round(std_tps, 2),
                "mean_elapsed_seconds": round(mean_elapsed, 3),
                "total_tiles": rep_results[0]["total_tiles"],
                "slides_processed": rep_results[0]["slides_processed"],
                "slides_with_tiles": rep_results[0]["slides_with_tiles"],
            }
        )

    return records


# ---------------------------------------------------------------------------
# CSV I/O
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
) -> None:
    workers = [r["num_workers"] for r in records]
    tps = [r["mean_tiles_per_second"] for r in records]
    err = [r["std_tiles_per_second"] for r in records]
    has_error = any(e > 0 for e in err)

    # ideal linear scaling anchored at the lowest worker count
    baseline = tps[0]
    baseline_workers = workers[0]
    ideal = [baseline * w / baseline_workers for w in workers]

    # projection from observed avg tiles/slide extrapolated to full dataset
    avg_tiles_per_slide = records[0]["total_tiles"] / max(records[0]["slides_processed"], 1)
    max_tps = max(tps)
    max_workers = workers[tps.index(max_tps)]
    projected_total_tiles = avg_tiles_per_slide * total_slides
    projected_hours = projected_total_tiles / max_tps / 3600

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_AXES_BG)

    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)

    ax.tick_params(colors=_TEXT, which="both")
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)

    # grid
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
    proj_text = (
        f"Projected · {dataset_label} (~{total_slides:,} slides)\n"
        f"At {max_workers} workers:  {projected_hours:.1f} hours"
    )
    ax.text(
        0.985,
        0.05,
        proj_text,
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

    # ── benchmark ────────────────────────────────────────────────────────
    print(f"\nRunning benchmark: workers={sorted(args.workers)}, repeat={args.repeat}")
    records = benchmark(
        slides=sampled,
        workers=sorted(args.workers),
        repeat=args.repeat,
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
        )

    # ── chart ─────────────────────────────────────────────────────────────
    plot_results(
        records,
        output_path=args.output_dir / "throughput.png",
        n_slides=len(sampled),
        target_spacing=args.target_spacing,
        tile_size=args.tile_size,
        backend=args.backend,
        total_slides=args.total_slides,
        dataset_label=args.dataset_label,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
