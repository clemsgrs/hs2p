# Artifact Reference

`hs2p` writes explicit named coordinate artifacts for both tiling and sampling.

## Artifact locations

- Tiling writes one artifact pair per slide under `tiles/`
- Sampling writes one artifact pair per annotation under `tiles/<annotation>/`

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
- `coordinates`
  - shape `(N, 2)`
  - level-0 tile origins in `[x, y]` order
- `tissue_fractions`
  - per-tile tissue or annotation coverage values aligned with `coordinates`

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

### `slide`

- `dimensions`
- `base_spacing_um`
- `level_downsamples`

### `tiling`

- `requested_tile_size_px`
- `requested_spacing_um`
- `read_level`
- `effective_tile_size_px`
- `effective_spacing_um`
- `tile_size_lv0`
- `tolerance`
- `step_px_lv0`
- `overlap`
- `min_tissue_fraction`
- `is_within_tolerance`
- `n_tiles`

### `segmentation`

- `tissue_method`
- `seg_downsample`
- `seg_level`
- `seg_spacing_um`
- `sthresh`
- `sthresh_up`
- `mthresh`
- `close`
- `use_otsu`
- `use_hsv`
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
- `image_path`
- `mask_path`
- `tiling_status`
- `num_tiles`
- `coordinates_npz_path`
- `coordinates_meta_path`
- `tiles_tar_path`
- `error`
- `traceback`

### Sampling manifest

- `sample_id`
- `annotation`
- `image_path`
- `mask_path`
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
- backend
- requested spacing and tile size
- overlap and minimum tissue fraction
- segmentation and filtering settings
- sampling selection/output metadata when relevant
