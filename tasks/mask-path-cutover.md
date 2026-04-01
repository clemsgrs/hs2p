# mask_path Cutover

- [x] Inspect current tiling, sampling, artifact, and CSV-loading paths
- [x] Make `mask_path` the only public CSV/process-list field
- [x] Rename public coordinate wrappers to use `mask_path`
- [x] Reject legacy `tissue_mask_path` and `annotation_mask_path` inputs
- [x] Update tests and fixtures for the new contract
- [x] Update README and docs to describe one public mask column
- [x] Run focused regression tests
