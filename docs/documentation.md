# Test Suite Structure

## Default fast suite

The default `pytest` run is the fast, backend-free regression lane for the Python-first codebase. It is centered on:

- `tests/test_tiling_api.py` for the public tiling API contract
- `tests/test_cli_smoke.py` for current-schema CLI smoke coverage
- mocked/internal algorithm tests that exercise tiling math and mask semantics without real WSI backends

`pyproject.toml` marks the default run with `-m "not integration"`, so routine local runs and CI focus on stable API and CLI coverage.

## Optional integration suite

Backend-dependent fixture tests are marked `@pytest.mark.integration` and excluded from the default run. These cover:

- `tests/test_legacy_coordinates_regression.py`
- `tests/test_real_fixture_smoke_regression.py`

They should run only in environments that provide the required WSI backend plus the real fixture assets.

## Fixture refresh script

`scripts/generate_test_fixture_tiling_artifacts.py` generates current-format tiling artifacts for `tests/fixtures/input/test-wsi.tif` and `tests/fixtures/input/test-mask.tif` using the public tiling API and the same fixture config used by the integration regression.

The default output directory is `tests/fixtures/gt-current/`, and the script writes:

- `{sample_id}.tiles.npz`
- `{sample_id}.tiles.meta.json`

This is intended to support the eventual transition away from the legacy `.npy` golden by giving the integration suite a readable, current-schema artifact source.

## Current guidance

- Prefer testing documented public APIs and supported CLI schemas over deprecated internal calling patterns.
- Keep low-level `hs2p.wsi` tests only when they protect unique algorithmic behavior that the API-level tests do not cover.
- Avoid adding new tests that depend on tuple-unpacking `CoordinateExtractionResult`, legacy CSV schemas, or legacy `.npy` output shapes as public contracts.

## Benchmark helper

`scripts/benchmark_tiling_cli.py` provides a simple branch-vs-branch wall-clock benchmark for the tiling CLI.

- It accepts one slide manifest with `wsi_path,mask_path` and optional `sample_id`.
- It derives unique `sample_id` values when needed, writes current-schema and legacy-schema CSVs automatically, and rewrites the config keys that changed between `main` and the refactor (`spacing`/`tile_size`/`min_tissue_percentage` versus `target_spacing_um`/`target_tile_size_px`/`tissue_threshold`).
- It runs the current checkout in place, checks out the legacy ref in a temporary git worktree, and writes per-run timings plus aggregated summaries to `benchmark_runs.csv` and `benchmark_summary.csv`.
- It also compares the first current run and first legacy run slide-by-slide, normalizing current `.tiles.npz`/`.tiles.meta.json` artifacts and legacy `.npy` artifacts to the same coordinate/metadata view, then writes `output_comparison.csv`.
- The benchmark exits non-zero if wall-clock runs fail or if the compared current/legacy outputs disagree on coordinates or shared extraction metadata (`num_tiles`, target spacing/tile size, read level, read tile size, `tile_size_lv0`).
- Example:
  `python scripts/benchmark_tiling_cli.py --slides-csv /path/to/slides.csv --config-file /path/to/config.yaml --output-dir /tmp/hs2p-benchmark --legacy-ref main --repeat 3`
- Use `--disable-visualize` when you want the comparison to focus on tiling throughput rather than preview rendering.

## Active planning

- [tasks/performance-improvement-plan.md](/Users/clems/Code/hs2p/tasks/performance-improvement-plan.md) stages performance work so the team can land low-risk throughput wins first, then tackle slide-level parallelism and larger coordinate-pipeline refactors with regression coverage in place.
- Phase 1 low-risk fixes are now in place: `tile_slides()` computes the batch config hash once, sampling always uses `1` inner worker per slide and rejects the undocumented `cfg.speed.inner_workers` override, sampling mask previews reuse the extraction path instead of reopening overlays, `segment_tissue()` skips the RGBA round-trip, and coordinate sorting is numeric rather than string-lexicographic.
- Phase 2 I/O reductions are now in place: `filter_coordinates()` reuses the already-opened mask and slices from one aligned in-memory mask array, tiling preview rendering crops from the loaded slide canvas instead of re-reading tiles, and `tile_slides()` defers preview writes through a single background worker so preview I/O can overlap with the next slide's compute step.
- Phase 3 slide-level concurrency is now in place: `tile_slides()` plans resume/precomputed/immediate-failure work up front, computes the remaining slides through `mp.Pool.imap_unordered()` when `num_workers > 1` and multiple slides need compute, lets workers write `.tiles.npz` / `.tiles.meta.json` atomically before returning lightweight metadata, drains out-of-order responses back into input order for artifact/process-list stability, and splits the worker budget adaptively so small compute batches can use more than one inner worker without oversubscribing the machine.
- The internal cleanup tranche is now complete in code: contour processing keeps NumPy arrays through worker accumulation instead of bouncing through Python lists/tuples, `HasEnoughTissue.check_coordinates()` uses an integral-image path with vectorized outputs, and `CoordinateExtractionResult` stores canonical `x`/`y` arrays while exposing `.coordinates` as a lazy compatibility view rather than duplicating the data in memory.
- The current mask/filter hot paths are also reduced further: sampling now scores tile coverage with one per-label integral image over the aligned mask instead of one Python crop per tile, and `filter_black_and_white_tiles()` batches exact read-level fetches through bounded supertiles instead of one backend read per candidate tile.
- Zero-tile extraction is now a supported low-level contract: tissue-free slides return empty coordinate arrays with stable read metadata instead of crashing when no contours are detected.
- Batch tiling now threads one precomputed config hash through per-slide compute instead of recomputing the same frozen-config hash for every slide, and `filter_coordinates()` annotations now match the actual caller contract (`mask_path: Path | None`, coordinate lists).
- The metadata-only slide descriptor in the public API is now named `SlideSpec`, which better distinguishes it from the opened backend object `WholeSlideImage`; `WholeSlide` remains as a compatibility alias.
- Sampling artifacts now record the active annotation threshold in `tissue_threshold` and include `independant_sampling` in their config hash, so the saved metadata matches the coordinate-generation path.
- `scripts/benchmark_throughput.py` now builds both balanced and skewed workloads from one slide manifest, records the workload label alongside elapsed time, tiles/s, slide counts, and failure counts, and writes separate charts for the balanced and skewed runs so batch-dispatch changes can be compared under both even and imbalanced slide mixes.
