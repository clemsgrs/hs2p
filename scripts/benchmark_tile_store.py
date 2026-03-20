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
        "--num-workers",
        type=int,
        default=4,
        help="Worker count passed to tile reader.",
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
    from PIL import Image

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

                img = Image.fromarray(tile_arr).convert("RGB")
                if result.read_tile_size_px != result.target_tile_size_px:
                    img = img.resize(
                        (result.target_tile_size_px, result.target_tile_size_px),
                        Image.LANCZOS,
                    )
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=jpeg_quality)
                buf.seek(0)
                t2 = time.perf_counter()
                encode_s += t2 - t1

                info = tarfile.TarInfo(name=f"{tile_count:06d}.jpg")
                info.size = buf.getbuffer().nbytes
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

    read_s = [float(r["read_s"]) for r in rows]
    encode_s = [float(r["encode_s"]) for r in rows]
    write_s = [float(r["write_s"]) for r in rows]
    total_s = [float(r["total_s"]) for r in rows]
    read_pct = [float(r["read_pct"]) for r in rows]
    encode_pct = [float(r["encode_pct"]) for r in rows]
    write_pct = [float(r["write_pct"]) for r in rows]
    tps = [float(r["tiles_per_second"]) for r in rows]

    return [
        {
            "tiles": int(rows[0]["tiles"]),
            "jpeg_quality": int(rows[0]["jpeg_quality"]),
            "num_workers": int(rows[0]["num_workers"]),
            "mean_read_s": _mean(read_s),
            "mean_encode_s": _mean(encode_s),
            "mean_write_s": _mean(write_s),
            "mean_total_s": _mean(total_s),
            "mean_read_pct": round(statistics.mean(read_pct), 2),
            "mean_encode_pct": round(statistics.mean(encode_pct), 2),
            "mean_write_pct": round(statistics.mean(write_pct), 2),
            "mean_tiles_per_second": round(statistics.mean(tps), 2),
            "std_tiles_per_second": round(_pstdev(tps), 2),
        },
    ]


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
    result = load_single_slide_result_from_config(
        config_file=args.config_file,
        num_workers=int(args.num_workers),
    )
    result = limit_tiling_result(result, max_tiles=int(args.max_tiles))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = int(args.warmup) + int(args.repeat)

    timed_rows: list[dict[str, Any]] = []
    run_counter = 0
    with BenchmarkProgressReporter(total_runs=total_runs) as reporter:
        reporter.print_banner(
            result=result,
            modes=["tile_store"],
            repeat=int(args.repeat),
            warmup=int(args.warmup),
        )
        for warmup_idx in range(int(args.warmup)):
            run_counter += 1
            reporter.start_run(
                run_counter=run_counter,
                phase="warmup",
                mode="tile_store",
                iteration_index=warmup_idx,
                iteration_total=int(args.warmup),
                total_read_calls=int(result.num_tiles),
                total_tiles=int(result.num_tiles),
            )
            benchmark_tile_store(
                result=result,
                jpeg_quality=int(args.jpeg_quality),
                num_workers=int(args.num_workers),
                output_dir=args.output_dir,
                progress_callback=reporter.advance,
            )
            reporter.finish_run()

        for repeat_index in range(int(args.repeat)):
            run_counter += 1
            reporter.start_run(
                run_counter=run_counter,
                phase="timed",
                mode="tile_store",
                iteration_index=repeat_index,
                iteration_total=int(args.repeat),
                total_read_calls=int(result.num_tiles),
                total_tiles=int(result.num_tiles),
            )
            metrics = benchmark_tile_store(
                result=result,
                jpeg_quality=int(args.jpeg_quality),
                num_workers=int(args.num_workers),
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
                num_workers=int(args.num_workers),
                read_s=metrics["read_s"],
                encode_s=metrics["encode_s"],
                write_s=metrics["write_s"],
                total_s=metrics["total_s"],
                jpeg_bytes=metrics["jpeg_bytes"],
            )
            reporter.console.print(
                (
                    f"tile_store             rep={repeat_index + 1} "
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
