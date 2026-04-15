# Documentation Notes

## 2026-04-15

- Split tissue preprocessing into an explicit mask-resolution step and a separate tiling step in `hs2p/preprocessing.py`.
- Added a shared `ResolvedTissueMask` boundary so precomputed masks and on-the-fly segmentation feed the same tiling pipeline.
- Made `tile_slides(...)` resolve masks for the whole batch before starting tiling work, while keeping `tile_slide(...)` on the single-slide compatibility path.
- Added a dedicated Rich/text progress phase for batch tissue resolution so CLI runs show mask-prep progress before tiling starts.

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
