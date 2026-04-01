# Documentation Notes

## 2026-04-01

- Preview saving now lives under `tiling.preview.save` instead of the old top-level `save_previews` flag.
- The preview toggle continues to control both mask overlays and tiling previews.
- Cut over the public mask field to `mask_path` everywhere in the CLI and process-list surfaces.
- Tiling now interprets `mask_path` as a tissue mask.
- Sampling now interprets `mask_path` as an annotation mask and rejects rows that omit it.
- Legacy `tissue_mask_path` and `annotation_mask_path` CSV/process-list inputs now fail validation instead of being translated.
