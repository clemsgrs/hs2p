# Performance Improvement Plan

_Derived from [tasks/performance-review.md](/Users/clems/Code/hs2p/tasks/performance-review.md), drafted 2026-03-12._

## Goal

Improve end-to-end tiling and sampling throughput without destabilizing the public API or changing the project's resolution-agnostic behavior.

## Strategy

Prioritize work in this order:

1. Fix obvious oversubscription and redundant I/O that deliver immediate wall-time wins with low behavioral risk.
2. Add focused regression and micro-benchmark coverage before larger concurrency or data-shape changes.
3. Defer invasive `CoordinateExtractionResult` / contour-pipeline refactors until the cheaper wins are measured and locked in.

## Phase 0 - Baseline and guardrails

- [ ] Add benchmark-oriented regression coverage for the current hot paths using mocked slides/masks where feasible.
- [ ] Capture a small repeatable baseline for:
  - `tile_slides()` over multiple mocked slides
  - `sampling.main()` worker fan-out
  - `filter_coordinates()` mask-coverage filtering
- [ ] Document the benchmark command(s) and expected comparison workflow in the test/dev docs.

## Phase 1 - Low-risk high-impact fixes

- [x] Hoist `compute_config_hash()` out of the `tile_slides()` slide loop (`P11`).
- [x] Remove the `segment_tissue()` RGBA round-trip and keep conversion on the existing RGB array (`P10`).
- [x] Fix `sampling.main()` oversubscription by separating outer process count from per-slide inner workers and standardizing sampling on `inner_workers=1` (`P2`).
- [x] Stop `sampling.process_slide()` from re-opening slide and mask files just to write the overlay preview; route preview generation through the existing extraction path (`P5`).
- [x] Replace the string-based coordinate ordering in `sort_coordinates_with_tissue()` with numeric sorting (`np.lexsort` or equivalent) and add a regression test proving `10` sorts after `9` (`P6`).

## Phase 2 - Mask and preview I/O reduction

- [x] Refactor `filter_coordinates()` to reuse the mask already opened by `WholeSlideImage` instead of opening the mask twice (`P3`).
- [x] Hoist invariant mask-spacing computations out of the per-coordinate loop in `filter_coordinates()` (`P3`).
- [x] Replace per-tile mask reads in `filter_coordinates()` with slicing from one in-memory mask thumbnail / slide-level array (`P3`).
- [x] Update preview rendering to crop from the already-loaded canvas instead of calling `get_tile()` per coordinate (`P4`).
- [x] Move preview writing off the main tiling critical path only after functional parity is covered by tests (`P4`).

## Phase 3 - Concurrency redesign for `tile_slides()`

- [x] Add an API-level regression test for multi-slide ordering, result serialization, and process-list writing before changing execution order (`P1`).
- [x] Introduce a serializable per-slide worker payload for `tile_slides()` so slide-level work can run via `mp.Pool` safely (`P1`).
- [x] Keep process-list writing as a single end-of-run aggregation step to preserve current artifact semantics (`P1`).
- [x] Re-measure interaction between slide-level processes and any inner thread pools to avoid recreating the `sampling.py` oversubscription problem (`P1`).

## Deferred until earlier phases are measured

- [x] Revisit removing `coordinates` from `CoordinateExtractionResult` only after confirming the public API and downstream tests can tolerate the shape change (`P7`).
- [x] Revisit keeping contour-processing data as NumPy arrays end-to-end once the lower-risk I/O wins are complete (`P8`).
- [x] Evaluate an integral-image implementation for `HasEnoughTissue.check_coordinates()` after measuring whether it still matters once mask I/O is removed (`P9`).

## Verification

- [x] For each phase, add or update tests before implementation where feasible, then prove the phase with a focused pytest target and one before/after benchmark comparison.
- [x] Run the default fast test suite after each completed phase.
- [ ] Run the relevant integration coverage for tiling/sampling behavior before merging any concurrency change.

## Success criteria

- [x] `sampling.main()` no longer multiplies outer process count by the same inner worker count and now rejects the undocumented inner-worker override.
- [x] Mask filtering no longer performs one backend tile read per candidate coordinate.
- [x] Preview generation no longer performs redundant tile reads from a slide canvas that is already loaded.
- [x] `tile_slides()` uses slide-level multiprocessing for multi-slide compute batches while preserving artifact/process-list ordering and avoiding nested-worker oversubscription by default.
- [x] `CoordinateExtractionResult` now uses `x`/`y` as canonical storage while keeping `.coordinates` as a lazy compatibility view instead of duplicating the data in memory.
- [ ] Multi-slide tiling throughput improves measurably on the benchmark fixture without output-schema regressions.
