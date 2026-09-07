# Artifact Reference

Tiling and annotation sampling share the same coordinate format and process list.

## Artifact locations

- Tiling writes one coordinate output per slide under `tiles/`.
- Per-annotation sampling writes one output under `tiles/<annotation>/`.
  The conventional `tissue` annotation stays flat under `tiles/`.
- Merged sampling writes one flat output under `tiles/`; `merged` is
  reserved as an output identity and cannot be used as an annotation name.

Each successful non-empty output produces:

- `{sample_id}.coordinates.npz`
- `{sample_id}.coordinates.meta.json`

An output with zero tiles is still successful. It writes only the metadata JSON,
with `n_tiles: 0`; `coordinates_npz_path` is `None` in the API and blank in the
process list. `load_tiling_result(None, metadata_path)` restores its empty arrays.

Optional tile TAR export writes:

- `{sample_id}.tiles.tar`
- `{sample_id}.tiles.manifest.csv`

The manifest contains `tile_index`, `x`, and `y`, mapping level-0 tile origins to
JPEG members named `000000.jpg`, `000001.jpg`, and so on. The returned artifact and
process-list row record `tiles_tar_path`; the manifest uses the same stem with
`.manifest.csv`. Batch annotation sampling does not currently support TAR export.

## `.coordinates.npz`

All NPZ arrays have shape `(N,)`, where `N` is the tile count.

| Array | Stored dtype | Meaning |
| --- | --- | --- |
| `tile_index` | `int32` | Contiguous tile IDs from `0` to `N - 1`. |
| `x` | `int64` | Level-0 tile-origin x-coordinates in pixels. |
| `y` | `int64` | Level-0 tile-origin y-coordinates in pixels. |
| `tissue_fractions` | `float32` | Tissue or annotation coverage aligned with `x` and `y`. |

Tile order is deterministic: numeric `x` first, then numeric `y` within each shared `x`.

## `.coordinates.meta.json`

The JSON has six sections. The fields below describe the current schema; loading
rejects missing or unexpected keys.

### `provenance`

- `sample_id`
- `image_path`
- `mask_path`
- `backend`
- `requested_backend`
- `mask_backend`
- `requested_mask_backend`
- `spacing_at_level_0`
  - the explicit level-0 spacing override, or `null` when no override was used

Backend fields distinguish requested readers from resolved readers. Both mask
backend values are `null` when no source mask was used.

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
- `requested_seg_downsample`
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

`requested_seg_downsample` records the configuration value; `seg_downsample` records
the actual pyramid or SAM2-thumbnail downsample used.

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
  - `"level0_px"`
- `tile_order`
  - `"x_then_y"`
- `annotation`
- `selection_strategy`
- `output_mode`

## `process_list.csv`

Both tissue tiling and annotation sampling use these columns:

- `sample_id`
- `annotation`
- `output_mode`
- `image_path`
- `mask_path`
- `requested_backend`
- `backend`
- `requested_mask_backend`
- `mask_backend`
- `tiling_status`
- `num_tiles`
- `coordinates_npz_path`
- `coordinates_meta_path`
- `tiles_tar_path`
- `mask_preview_path`
- `tiling_preview_path`
- `error`
- `traceback`

`tiling_status` is `success` or `failed`, including for annotation sampling.
Successful per-annotation outputs have separate rows. `annotation` is `tissue`
for ordinary tiling, the class name for per-annotation sampling, or `merged` for
merged output. Failure rows retain the reason in `error` and the diagnostic
traceback in `traceback`.

## Resume and validation

For tissue tiling, `resume` treats a compatible successful process-list row as completed processing.
`read_coordinates_from` reuses coordinate computation only, so downstream TAR and tiling
preview outputs enabled for the current invocation are still materialized.

Existing artifacts are validated against their structured metadata:

- slide identity
- mask path
- explicit level-0 spacing override presence and value
- resolved slide backend and, when a source mask exists, resolved mask backend
- requested spacing and tile size
- overlap and minimum tissue fraction
- segmentation and filtering settings
- SAM2 checkpoint and model-config paths for SAM2 segmentation
- sampling selection/output metadata when relevant

Requested backend names are provenance only: changing `auto` to the same resolved
reader does not invalidate the artifacts. Batch annotation sampling currently
rejects both `resume` and `read_coordinates_from`.

Older metadata or process lists missing the mask-backend fields are rejected.
Recompute those artifacts with the current schema before reusing them.

Source identities are intentionally path-only. Artifact validation does not hash, stat, or
reopen a slide, mask, SAM2 checkpoint, or SAM2 model config solely to detect an in-place
replacement at the same path. Keeping the contents behind those paths stable is the user's
responsibility. After replacing a source in place, remove the reusable coordinate artifact or
write the replacement under a new path so incompatible coordinates are not reused.

After a slide is recorded as successfully completed, resume also treats its downstream
`tiles_tar_path`, `mask_preview_path`, and `tiling_preview_path` values as provenance. It does
not check, reopen, regenerate, or repair those files, even when the resumed invocation requests
the corresponding outputs. Deleting or replacing a recorded downstream file after successful
completion is the user's responsibility.
