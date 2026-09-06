# Performance verification

Run commands from the repository root. Use the same Python environment, machine,
slide, backend, configuration, tile limit, and worker count for both revisions.
On Ubuntu/Debian, install the native dependency before the Python packages.
The WSD read modes use `wholeslidedata==0.0.15`, which pins `rtree==1.0.0`
and requires `libspatialindex_c` even for image-only reads:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends -y libspatialindex-dev
```

Then install the CPU dependencies in a virtual environment:

```bash
python -m pip install -e '.[openslide,testing]' 'wholeslidedata==0.0.15' 'numpy<2'
```

The checked-in `benchmarks/fixture.json` and `benchmarks/fixture.csv` use the real
TIFF fixtures with OpenSlide and PIL JPEG encoding. They need no GPU or ASAP.
The CSV paths are relative to the repository root. JSON is accepted by the normal
OmegaConf configuration loader. Coordinate selection is computed before timing;
the benchmark loader does not reuse coordinate artifacts.

## Preprocessing CPU and memory

The deterministic preprocessing runner covers sparse/dense contour selection,
mask decoding, and the real fixture. It complements the existing read/store
benchmarks, which exclude coordinate preparation:

```bash
python scripts/benchmark_preprocessing.py \
  --cases contours_sparse contours_dense mask_decode fixture --repeat 5 \
  --output output/perf-before/preprocessing.json
```

Repeat with a fresh `perf-after` output. Keep case inputs identical; inspect the
recorded output fingerprints as well as timing and memory measurements. The runner
refuses to overwrite an existing report and saves each completed case. Add
`--profile-dir output/perf-profile` to collect profiles for investigation; compare
unprofiled runs for headline timings.

The synthetic cases use a 2048×2048 mask on a 32768×32768 slide grid:
`contours_sparse` has 64 separated 96×96 tissue islands (9,216 output tiles),
while `contours_dense` has one 1920×1920 region (57,600 tiles). They exercise
production `generate_tiles` with one worker. `mask_decode` loads a 4-million-pixel
byte mask through an in-memory reader: it measures validation and binarization,
**not codec or disk I/O**. `fixture` calls `tile_slide` with explicit OpenSlide slide
and mask backends, including opening and decoding the real TIFFs, contour
selection, and coordinate generation; it excludes artifact writing.

Each case performs one warmup by default, records all uninstrumented wall-clock
samples and their median, then makes a separate allocation-traced call. The
`peak_traced_bytes` measurement includes NumPy allocations visible to tracemalloc;
it excludes prebuilt inputs and native decoder/OpenCV allocations that Python
cannot trace. It is not process RSS. Profiles also come from a separate call.
JSON reports include package/runtime details, source and runner hashes, and output
hashes. Compare output hashes exactly across revisions. Use a quiet machine and
inspect the sample spread; shared-host scheduling and warm caches affect timings.
The CPU runner requires no additional benchmark package.

See [the audit results](performance-audit.md) for measured changes and limitations.

## Read and extraction throughput

Use a **new output directory for each revision**. The read runner resumes existing
modes and will skip them; reusing a directory does not produce new measurements.

```bash
python scripts/benchmark_tile_read.py \
  --config-file benchmarks/fixture.json --output-dir output/perf-before/read \
  --modes regular_wsd supertiles_wsd --num-workers 1 --warmup 1 --repeat 5
python scripts/benchmark_tile_store.py \
  --config-file benchmarks/fixture.json --output-dir output/perf-before/store \
  --workers 1 --jpeg-backend pil --warmup 1 --repeat 5
```

Repeat with `perf-after` after changing revisions. Both scripts write
`benchmark_runs.csv` (individual samples) and `benchmark_summary.csv` (mean and
standard deviation). Compare the same mode and worker count, including tile
counts and read checksums before interpreting throughput. Checksums are a coarse
sanity check; correctness tests remain necessary. `--max-tiles 64` gives a quick
smoke run; use all fixture tiles or a representative external slide for claims.

The read runner times region reads and tile consumption, excluding slide opening,
coordinate selection, and read planning. Its WSD modes use wholeslidedata, so they
compare read strategies rather than measuring every production reader path.
The store runner times the production `extract_tiles_to_tar` call, including
planning, reading, JPEG encoding and TAR writing, and reports read/encode/write
phase timings. It writes a temporary TAR per repetition and removes it afterward;
its output directory therefore determines the measured filesystem. Progress and
phase instrumentation are included. Warmups exercise filesystem/decoder caches;
these commands do not measure cold-cache storage performance.

For backend or concurrency studies, copy the fixture JSON, change the slide CSV
and explicit backend, and sweep `--workers 1 2 4`. CuCIM GPU modes require CUDA
hardware and the corresponding extra; TurboJPEG requires its Python and native
libraries. Do not compare measurements from different machines as a speedup.

## Correctness and CI

```bash
python -m pytest -q --no-cov -m script tests/test_benchmarking.py tests/test_benchmark_tile_store.py
python -m pytest -q --no-cov tests
python -m pytest -q --no-cov -m integration tests/test_fixture_artifacts_regression.py tests/test_tile_count_heuristic_regression.py tests/test_tiling_preview_mask_overlay.py
```

The last command contains ASAP-dependent golden-coordinate checks; inspect skips
and use the existing project Docker image when ASAP is needed. The fixture benchmark
workflow runs the preprocessing and portable CPU read/store commands and uploads raw JSON/CSVs. It is a
smoke/performance artifact workflow, without a noisy cross-run speed threshold.

Shared helpers remain in `benchmark_tile_read_support.py`,
`benchmark_tile_store_support.py`, and `benchmark_tile_utils.py`. The obsolete test
for the removed `benchmark_throughput.py` runner has been removed. Local untracked
experiment and plotting scripts are not dependencies of the repeatable workflow.

## Earlier throughput experiments

Previous single-slide experiments reported 62.64 tiles/s for regular WSD/PIL,
82.23 for WSD supertiles, 85.60 for CuCIM batch reads, 193.23 for CuCIM supertiles
with PIL, and 217.56 with TurboJPEG, all at four workers. These are historical
observations, not portable baselines: their original environment and raw results
are not checked in. Use the commands above for a current comparison.
