# Documentation Notes

## 2026-04-17 — CI test invocation cleanup

- Updated `.github/workflows/pr-test.yaml` so the integration regression step only runs `tests/test_fixture_artifacts_regression.py`, matching the current test tree.

## 2026-04-17 — Mask coverage config schema fix

- Normalized `tiling.masks.pixel_mapping`, `tiling.masks.colors`, and `tiling.masks.min_coverage` in `hs2p/configs/default.yaml` to use mapping syntax instead of one-item lists so OmegaConf can merge user overrides correctly.
- Updated `hs2p/configs/resolvers.py` so OmegaConf `DictConfig` nodes are accepted as mappings when resolving sampling config.

## 2026-04-17 — Preview contour naming cleanup

- Renamed the preview color setting to `tiling.preview.tissue_contour_color` in the default config and docs.
- Documented `tiling.preview.mask_overlay_alpha` as applying only to the filled annotation overlay path.

## 2026-04-17 — Unified tiling interface (breaking changes)

- Merged `hs2p.cli.tiling` and `hs2p.cli.sampling` into a single entrypoint: `python -m hs2p --config-file …`.
- `hs2p/cli/` package deleted; use `python -m hs2p` in all scripts.
- `tiling.sampling_params` config block renamed to `tiling.masks` (`pixel_mapping`, `color_mapping`, `min_coverage`).
- `tiling.params.tissue_threshold` folded into `tiling.masks.min_coverage["tissue"]`.
- Tar path rule: `{sample_id}.tiles.tar` when annotation is `"tissue"` (default), otherwise `{sample_id}.{annotation}.tiles.tar`.
- Process list gains `annotation` column (always `"tissue"` for the binary default).
- Internal modules reorganized into `hs2p/tiling/` package: `result.py`, `contours.py`, `coverage.py`, `generate.py`, `mask.py`, `io.py`, `single.py`, `tar.py`, `orchestration.py`. `hs2p/preprocessing.py` and `hs2p/api.py` are now thin re-export shims.

## 2026-04-16

- Batch tiling now uses a spawn-based multiprocessing context for all SAM2 work and GPU-decode work so CUDA initializes in fresh child interpreters instead of forked workers.
- CPU-only and non-SAM2 tiling paths still use the existing multiprocessing behavior.
- SAM2 predictor INFO lines from `sam2_image_predictor.py` are filtered out in the SAM2 segmentation path so the CLI stays quiet during inference.
- SAM2 tissue segmentation now pads rectangular thumbnails to a square before inference and crops back after prediction, which preserves aspect ratio on extreme slides.
- Tissue-mask previews now render contour boundaries instead of a filled mask overlay: outer tissue borders use evergreen `#255E3B` and hole contours use coral `#F26B3A`.
- The batch tiling preview path reuses the precomputed contour hierarchy from preprocessing when it is available.
- Mask preview files with `.jpg`/`.jpeg` suffixes are converted to RGB before saving so RGBA overlays no longer fail at write time.

## 2026-04-15

- Split tissue preprocessing into an explicit mask-resolution step and a separate tiling step in `hs2p/preprocessing.py`.
- Added a shared `ResolvedTissueMask` boundary so precomputed masks and on-the-fly segmentation feed the same tiling pipeline.
- Made `tile_slides(...)` resolve masks for the whole batch before starting tiling work, while keeping `tile_slide(...)` on the single-slide compatibility path.
- Added a dedicated Rich/text progress phase for batch tissue resolution so CLI runs show mask-prep progress before tiling starts.
- Switched the SAM2 thumbnail path to a fixed internal `8.0 µm/px` target resolved through the existing spacing-level selection helper, instead of the old power-based thumbnail rule.

## 2026-04-12

- Added package markers for `tests/` and `scripts/` so test modules can import shared helpers with explicit package-qualified paths.
- Updated backend-selection test imports to use `tests.test_progress` after the test suite became package-aware.

## 2026-04-14

- Centralized tissue-mask generation in `hs2p/segmentation.py` so both `preprocess_slide()` and `WSI.segment_tissue()` use the same implementation.
- Replaced the old `use_hsv` / `use_otsu` segmentation booleans with explicit `SegmentationConfig.method`.
- Added SAM2-oriented segmentation config fields and optional dependency wiring for checkpoint/config-path driven inference.

## 2026-04-01

- Preview saving now lives under `tiling.preview.save` instead of the old top-level `save_previews` flag.
- The preview toggle continues to control both mask overlays and tiling previews.
- Cut over the public mask field to `mask_path` everywhere in the CLI and process-list surfaces.
- Tiling now interprets `mask_path` as a tissue mask.
- Sampling now interprets `mask_path` as an annotation mask and rejects rows that omit it.
- Legacy `tissue_mask_path` and `annotation_mask_path` CSV/process-list inputs now fail validation instead of being translated.
