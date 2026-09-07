# Performance audit — 2026-09-06

Baseline: `d4ea0c3ae1e2b9c8d106cf9c31324ec2f1828ae5`. The original production
package was exported with `git archive` to `/tmp`, then run with the same new
preprocessing benchmark. Existing untracked experiments were preserved.

## Changes and scope

1. **Contour-local coverage.** `generate_tiles` previously rasterized and integrated
the whole segmentation canvas for every contour. It now rasterizes only each
contour's bounding rectangle. Projection still happens on the original mask grid
before subtracting the crop origin, preserving fractional downsampling, holes,
boundaries, and the full tile-area denominator. Dense whole-slide contours still
need nearly the whole canvas; fragmented tissue benefits most. No new cache or
shared mutable state was added, and integral precision is unchanged.
2. **Byte-mask label validation.** Loading a uint8 mask previously widened the raster
to int64 and sorted all pixels to find labels. A 256-bin OpenCV histogram now finds
the same sorted label set. Other dtypes retain the existing path. The optimization
covers tissue and annotation masks, including strided channel views, while
preserving label rejection and diagnostic ordering.

## Measurements

All paired output fingerprints match exactly, including all 459 real-fixture tiles.
Values below are observed medians and peak traced allocation (decimal MB):

| Workload | Before → after time | Before → after peak allocation |
| --- | --- | --- |
| 64 sparse contours, 9,216 tiles | 39.413 s → 0.0159 s | 71.56 → 1.18 MB (98.4% less) |
| One dense contour, 57,600 tiles | 0.1383 s → 0.1157 s | 77.10 → 68.47 MB (11.2% less) |
| 4 Mpx byte-mask loading, in-memory reader | 0.1327 s → 0.0705 s | 75.52 → 37.90 MB (49.8% less) |
| Real TIFF fixture, `tile_slide` | 0.00922 s → 0.01048 s | 3.64 → 2.25 MB (38.0% less) |

The sparse case uses five samples; the smaller cases use an eleven-sample
confirmation pair because the first pair was noisy. All samples from both pairs
are retained in [the raw measurement report](../benchmarks/results/performance-audit.json).
Each run used one warmup, one worker, and one OpenCV thread. Input construction,
output hashing, allocation tracing and optional profiling were outside timed
samples. Traced allocation excludes prebuilt inputs and untracked native memory.

This was a busy shared Intel Xeon Gold 6342 host (96 logical CPUs), Linux x86-64,
Python 3.10.12, NumPy 2.2.6, OpenCV 4.13.0, Pillow 12.1.1, OpenSlide Python 1.4.3 /
native 4.0.1, and wholeslidedata 0.0.15. Timings were highly variable: sparse
baseline samples ranged from 21.26 to 287.91 seconds, versus 0.0130–0.0279 seconds
afterward. A universal speedup multiplier would be misleading. The deterministic
work reduction and allocation reduction corroborate a substantial sparse-case win.

For dense contours, confirmation timing ranges overlap (0.079–0.238 s before,
0.086–0.199 s after), so no reliable latency improvement is claimed. Byte-mask
loading improved in the longer confirmation (0.133 → 0.071 s median), although
its first pair was flat; the memory reduction is the stronger result. The real
fixture has a small mask and little contour work: its timing ranges also overlap,
and **no end-to-end speed improvement is claimed**. The 1.3 ms median increase
is retained in the table rather than hidden. Larger, fragmented real slides are
needed to quantify dataset-level throughput.

Reproduce the five-sample synthetic pair with the [benchmark commands](benchmark.md);
use `--cases contours_dense mask_decode fixture --repeat 11` for the confirmation
pair. Export the recorded baseline with
`git archive d4ea0c3ae1e2b9c8d106cf9c31324ec2f1828ae5`; for this audit, the
same benchmark script was copied into its `scripts/` directory. Run that copy with
the same interpreter to import the baseline package without modifying your
working checkout. Archive exports have no `.git`, so their revision field is null;
the baseline commit is recorded above and in the combined report.

## Verification

Both resource regressions were written and observed failing before implementation.
The contour test originally integrated 8,192 pixels for two islands totaling 8
pixels. It now processes 8 pixels, with one and two workers. Tests specify exact
coordinates and coverage for anisotropic scaling, holes, padding, and empty edge
crops. A separate deterministic comparison against the saved original implementation
matched coordinates and float32 fractions exactly in 405 geometry combinations.

The default suite passed 506 tests (one skip); targeted integration checks passed
four tests with two ASAP-dependent golden regressions skipped. The benchmark
script suite passed 21 tests. Real OpenSlide fixture read and production TAR-store
smokes each processed 64 tiles. Those concurrent smoke runs are correctness checks,
not performance evidence. Python compilation and `git diff --check` passed.
Flake8 was unavailable locally and its installation was blocked by network DNS;
no lint-pass claim is made.

## Maintained measurement workflow

See [benchmark commands and boundaries](benchmark.md). The audit reuses the existing
read/store runners and shared helpers, adds one CPU runner for the previously
unmeasured preprocessing stages, fixes stale imports and benchmark mocks, and
removes the obsolete test for a deleted throughput runner. Portable fixture
JSON/CSV replace CI dependencies on ignored configs and an untracked plotter.
CI now uploads raw preprocessing JSON and read/store CSVs without noisy speed gates.

## Remaining investigations

The inspected core is a local WSI pipeline, with no database query workload to tune.
Read grouping, supertiles, worker controls, and read/encode/write phase timing already
exist; this audit preserves them. Possible follow-ups include contour-hierarchy
scans on highly fragmented masks and QC reads of candidates later rejected by
annotation coverage. These have no before/after evidence here and were not changed.

GPU decode/encoder comparisons require CuCIM/CUDA and TurboJPEG, which were not
available. ASAP is needed for the checked-in golden-coordinate regressions. A
production-scale throughput claim also needs a representative multi-scanner slide
cohort and a specified local/network filesystem and cache policy. The maintained
read/store commands accept an external single-slide CSV and worker sweeps; repeat
across that cohort without reusing output directories. Synthetic masks demonstrate
the selected algorithmic waste, not end-to-end dataset throughput.
