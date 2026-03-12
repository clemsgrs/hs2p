# Artifact Reference

HS2P writes explicit named artifacts for both the Python API and the CLI.

## Coordinate artifact locations

- Tiling writes one pair per slide under `coordinates/`
- Sampling writes one pair per annotation under `coordinates/<annotation>/`

Each successful output produces:

- `{sample_id}.tiles.npz`
- `{sample_id}.tiles.meta.json`

## `.tiles.npz`

- `tile_index`
  - Contiguous tile ids from `0` to `num_tiles - 1`
  - Defines the row order for the other arrays in the artifact
- `x`
  - Tile origin x-coordinate in level-0 slide pixels
- `y`
  - Tile origin y-coordinate in level-0 slide pixels
- `tissue_fraction`
  - Optional per-tile tissue coverage measured during extraction

## `.tiles.meta.json`

- `sample_id`
  - Sample identifier that produced the artifact and names the files
- `image_path`
  - Slide path used to generate or validate the coordinates
- `mask_path`
  - Mask path used during generation, or `null` when no mask was provided
- `backend`
  - Whole-slide backend used for extraction
- `target_spacing_um`
  - Requested tile resolution in microns per pixel
- `target_tile_size_px`
  - Requested tile width and height in pixels at the target spacing
- `read_level`
  - Pyramid level actually read from the slide
- `read_spacing_um`
  - Native spacing of the level that was read
- `read_tile_size_px`
  - Tile width and height at the read level before mapping back to level 0
- `tile_size_lv0`
  - Tile width and height expressed in level-0 pixels
- `overlap`
  - Requested overlap fraction between neighboring tiles
- `tissue_threshold`
  - Minimum tissue fraction used to keep tiles for this artifact
  - For tiling artifacts, this is the global tiling threshold; for sampling artifacts, it is the active annotation threshold
- `num_tiles`
  - Number of tiles stored in the artifact
- `config_hash`
  - Hash of the effective coordinate-generation config
  - Sampling artifacts include the active annotation plus sampling-mode details that affect which coordinates are written

## `process_list.csv`

Batch runs also write a manifest summarizing per-slide status.

Tiling columns:

- `sample_id`
- `image_path`
- `mask_path`
- `tiling_status`
- `num_tiles`
- `tiles_npz_path`
- `tiles_meta_path`
- `error`
- `traceback`

Sampling columns:

- `sample_id`
- `image_path`
- `mask_path`
- `sampling_status`
- `error`
- `traceback`
