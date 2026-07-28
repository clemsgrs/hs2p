# Artifact Reference

`hs2p` writes explicit named coordinate artifacts for both tiling and sampling.

## Artifact locations

- Tiling writes one artifact pair per slide under `tiles/`
- Per-annotation sampling writes one artifact pair under `tiles/<annotation>/`.
  The conventional `tissue` annotation stays flat under `tiles/`.
- Structural merged sampling writes one flat artifact pair under `tiles/`; `merged` is
  reserved as an output identity and cannot be used as an annotation name.

Each successful output produces:

- `{sample_id}.coordinates.npz`
- `{sample_id}.coordinates.meta.json`

Optional tile tar export writes:

- `{sample_id}.tiles.tar`
- `{sample_id}.tiles.manifest.csv`

## `.coordinates.npz`

The NPZ contains the canonical geometry arrays:

- `tile_index`
  - contiguous tile ids from `0` to `n_tiles - 1`
- `x`
  - shape `(N,)`
  - level-0 tile origin x-coordinates
- `y`
  - shape `(N,)`
  - level-0 tile origin y-coordinates
- `tissue_fractions`
  - per-tile tissue or annotation coverage values aligned with `x` and `y`

Tile order is deterministic: numeric `x` first, then numeric `y` within each shared `x`.

## `.coordinates.meta.json`

The metadata file is structured into:

- `provenance`
- `slide`
- `tiling`
- `segmentation`
- `filtering`
- `artifact`

### `provenance`

- `sample_id`
- `image_path`
- `mask_path`
- `backend`
- `requested_backend`
- `spacing_at_level_0`
  - the explicit level-0 spacing override, or `null` when no override was used

### `slide`

- `dimensions`
- `base_spacing_um`
- `level_downsamples`

### `tiling`

- `requested_tile_size_px`
- `requested_spacing_um`
- `read_level`
- `read_tile_size_px`
- `read_spacing_um`
- `tile_size_lv0`
- `tolerance`
- `step_px_lv0`
- `overlap`
- `min_tissue_fraction`
- `is_within_tolerance`
- `n_tiles`

When `is_within_tolerance` is true, `tile_size_lv0` and `step_px_lv0` reflect the actual read-level crop geometry, so a slide read at level 0 keeps the level-0 footprint aligned with the crop size rather than the nominal requested-spacing projection.

### `segmentation`

- `tissue_method`
- `seg_downsample`
- `seg_level`
- `seg_spacing_um`
- `sthresh`
- `sthresh_up`
- `mthresh`
- `close`
- `sam2_checkpoint_path`
- `sam2_config_path`
  - path-based SAM2 segmentation identity; both are `null` for non-SAM2 artifacts
- `mask_path`
- `ref_tile_size_px`
- `tissue_mask_tissue_value`
- `mask_level`
- `mask_spacing_um`

### `filtering`

- `a_t`
- `a_h`
- `filter_white`
- `filter_black`
- `white_threshold`
- `black_threshold`
- `fraction_threshold`
- `filter_grayspace`
- `grayspace_saturation_threshold`
- `grayspace_fraction_threshold`
- `filter_blur`
- `blur_threshold`
- `qc_spacing_um`

### `artifact`

- `coordinate_space`
- `tile_order`
- `annotation`
- `selection_strategy`
- `output_mode`

## `process_list.csv`

### Tiling manifest

- `sample_id`
- `annotation`
- `image_path`
- `mask_path`
- `requested_backend`
- `backend`
- `tiling_status`
- `num_tiles`
- `coordinates_npz_path`
- `coordinates_meta_path`
- `tiles_tar_path`
- `mask_preview_path`
- `tiling_preview_path`
- `error`
- `traceback`

Each attempted slide is recorded as `success` or `failed`. Failure rows retain
the concise reason in `error` and the detailed diagnostic traceback in
`traceback`; they are failed slides, not skipped slides.

### Sampling manifest

- `sample_id`
- `annotation`
- `image_path`
- `mask_path`
- `requested_backend`
- `backend`
- `sampling_status`
- `num_tiles`
- `coordinates_npz_path`
- `coordinates_meta_path`
- `error`
- `traceback`

## Resume and validation

Existing artifacts are validated against their structured metadata:

- slide identity
- mask path
- explicit level-0 spacing override presence and value
- backend
- requested spacing and tile size
- overlap and minimum tissue fraction
- segmentation and filtering settings
- SAM2 checkpoint and model-config paths for SAM2 segmentation
- sampling selection/output metadata when relevant

Source identities are intentionally path-only. Artifact validation does not hash, stat, or
reopen a slide, mask, SAM2 checkpoint, or SAM2 model config solely to detect an in-place
replacement at the same path. Keeping the contents behind those paths stable is the user's
responsibility. After replacing a source in place, remove the reusable coordinate artifact or
write the replacement under a new path so incompatible coordinates are not reused.
