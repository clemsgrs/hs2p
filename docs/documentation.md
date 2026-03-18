# Documentation Notes

## 2026-03-18

- `scripts/generate_tissue_mask.py` now resolves requested spacings using the same physical-spacing rule as the main library: when no native pyramid level matches within tolerance, it reads the finest level at or below the requested spacing and resamples directly to the exact target spacing.
- This fixes coarse target spacings on non-power-of-two pyramids, where the previous resolver incorrectly failed unless the target spacing could be reached by repeated `2x` downsampling from the chosen read level.
- The spacing-resolution error now reports the requested target spacing, the available pyramid spacings, and actionable next steps when the request would require upsampling beyond the finest level.
- Mask-aware tiling and annotation sampling now share one internal coordinate-extraction engine, with explicit internal request fields for selection strategy (`merged_default_tiling`, `joint_sampling`, `independent_sampling`) and output mode (`single_output`, `per_annotation`).
- Masked default tiling now hashes the canonical tissue/background sampling preset, so stale masked artifacts created before this refactor are intentionally invalidated on resume or `read_tiles_from`.
- Sampling config resolution is now strict and centralized: partial sampling configs no longer receive hidden defaults, `background` is required for explicit sampling configs, and preview color mappings are validated in one place.
- Sampling process tracking now records per-annotation results, including effective config hashes and artifact paths, and resume validation checks per-annotation artifacts against the same effective config model used when writing them.
- Runtime config dataclasses now live under `hs2p.configs.models` and are re-exported from `hs2p.configs`, `hs2p.api`, and `hs2p`, so the WSI layer and public API consume the same concrete `TilingConfig`, `SegmentationConfig`, `FilterConfig`, and `PreviewConfig` types.
- Raw CLI/OmegaConf config is now normalized through `hs2p.configs.resolvers`, and the tiling/sampling entrypoints consume only resolved runtime objects instead of manually rebuilding configs inline.
- The WSI implementation now uses the public tiling field names directly (`target_spacing_um`, `target_tile_size_px`, `tissue_threshold`) with no parallel `*Parameters` aliases or legacy tiling-property vocabulary.
- `ResolvedSamplingSpec` is now the only runtime sampling model across the API, sampling workflow, and WSI execution path; the old `SamplingParameters` shim and dual-signature `sampling_params=` paths have been removed.
- Preview naming is now consistent across config, APIs, paths, and tests: `PreviewConfig`, `save_previews`, `tiling.preview`, `mask_preview_path`, and `write_coordinate_preview()`.
- Random seeding is now owned only by `setup()`, and the low-level tissue checker now consumes a single `ResolvedTileGeometry` object instead of a loose bag of correlated geometry scalars.
