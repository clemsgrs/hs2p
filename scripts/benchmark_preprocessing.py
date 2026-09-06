#!/usr/bin/env python3
"""Repeatable CPU preprocessing benchmarks; see docs/benchmark.md."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hs2p.api import FilterConfig, SegmentationConfig, SlideSpec, TilingConfig, tile_slide  # noqa: E402
from hs2p.tiling.generate import generate_tiles  # noqa: E402
from hs2p.tiling.mask import load_precomputed_tissue_mask  # noqa: E402
from hs2p.tiling.result import ContourResult  # noqa: E402

CASES = ("contours_sparse", "contours_dense", "mask_decode", "fixture")


def build_case(name: str):
    if name.startswith("contours_"):
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        contours = []
        rectangles = (
            [(x, y, 96) for x in range(64, 2048, 256) for y in range(64, 2048, 256)]
            if name == "contours_sparse" else [(64, 64, 1920)]
        )
        for x, y, size in rectangles:
            mask[y:y + size, x:x + size] = 255
            contours.append(np.array(
                [[[x, y]], [[x + size - 1, y]], [[x + size - 1, y + size - 1]], [[x, y + size - 1]]],
                dtype=np.int32,
            ) * 16)
        contour_result = ContourResult(contours, [[] for _ in contours], mask)

        def run():
            result = generate_tiles(
                (32768, 32768), contour_result, requested_tile_size_px=128,
                requested_spacing_um=0.5, base_spacing_um=0.5,
                level_downsamples=[1.0, 16.0], min_tissue_fraction=0.1, num_workers=1,
            )
            return result.x, result.y, result.tissue_fractions

        return run
    if name == "mask_decode":
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        mask[::2, :] = 7
        slide = SimpleNamespace(spacing=0.5, level_downsamples=[1.0], level_dimensions=[(2048, 2048)])
        reader = SimpleNamespace(**vars(slide), read_region=lambda *_: mask, close=lambda: None)

        def run():
            with patch("hs2p.tiling.mask.open_slide", return_value=reader):
                result, _, _ = load_precomputed_tissue_mask(
                    mask_path="benchmark-mask.tif", slide=slide, seg_level=0,
                    tissue_value=7, mask_backend="openslide",
                )
            return (result,)

        return run
    if name == "fixture":
        def run():
            result = tile_slide(
                SlideSpec(sample_id="test-wsi", image_path=ROOT / "tests/fixtures/input/test-wsi.tif",
                          mask_path=ROOT / "tests/fixtures/input/test-mask.tif"),
                tiling=TilingConfig(requested_spacing_um=0.5, requested_tile_size_px=224,
                                    tolerance=0.07, overlap=0.0, min_coverage={"tissue": 0.1}, backend="openslide", mask_backend="openslide"),
                segmentation=SegmentationConfig(method="hsv", downsample=64),
                filtering=FilterConfig(ref_tile_size=224, a_t=4, a_h=2), num_workers=1,
            )
            return result.x, result.y, result.tissue_fractions

        return run
    raise ValueError(name)


def signature(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return {"sha256": digest.hexdigest(), "elements": int(arrays[0].size)}


def measure(name, repeat, warmup, profile_dir):
    # Input construction, hashing, profiling and allocation tracing are outside timed samples.
    run = build_case(name)
    for _ in range(warmup):
        run()
    samples = []
    expected = None
    for _ in range(repeat):
        start = time.perf_counter()
        arrays = run()
        samples.append(time.perf_counter() - start)
        current = signature(arrays)
        if expected is not None and current != expected:
            raise RuntimeError(f"Non-deterministic output in {name}")
        expected = current
        del arrays
    tracemalloc.start()
    arrays = run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if signature(arrays) != expected:
        raise RuntimeError(f"Allocation run changed output in {name}")
    del arrays
    if profile_dir:
        profile_dir.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.runcall(run)
        profiler.dump_stats(str(profile_dir / f"{name}.prof"))
    return {"case": name, "seconds": samples, "median_seconds": statistics.median(samples),
            "peak_traced_bytes": peak, "output": expected}


def git_metadata(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_signature():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "hs2p").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES[:-1]))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be positive and warmup non-negative")
    if args.output.exists():
        parser.error("output already exists; use a fresh path for each measurement")
    cv2.setNumThreads(1)
    report = {
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                        "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
                        "source_sha256": source_signature(),
                        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "revision": git_metadata("rev-parse", "HEAD"),
                        "working_tree": git_metadata("status", "--short")},
        "repeat": args.repeat, "warmup": args.warmup, "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        print(f"Measuring {name} ({args.repeat} samples)...", file=sys.stderr, flush=True)
        result = measure(name, args.repeat, args.warmup, args.profile_dir)
        report["results"].append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
