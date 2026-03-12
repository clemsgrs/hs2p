# CLI Guide

HS2P provides two batch entrypoints:

- `python -m hs2p.tiling`
- `python -m hs2p.sampling`

Both consume the same input CSV schema and the same base tiling configuration.

## Input CSV

```csv
sample_id,image_path,mask_path
slide-1,/data/slide-1.tif,/data/slide-1-mask.tif
slide-2,/data/slide-2.tif,
```

- `sample_id`
  - Stable identifier used to name output artifacts
- `image_path`
  - Path to the whole-slide image
- `mask_path`
  - Optional tissue or annotation mask path

## Quick start

Start from [`hs2p/configs/default.yaml`](../hs2p/configs/default.yaml), then edit:

- `csv`
- `output_dir`
- `tiling.backend`
- `tiling.params.target_spacing_um`
- `tiling.params.target_tile_size_px`

Run tiling:

```bash
python -m hs2p.tiling --config-file /path/to/config.yaml
```

Run sampling:

```bash
python -m hs2p.sampling --config-file /path/to/config.yaml
```

## Current config areas

- `tiling.read_tiles_from`
  - Optional directory containing precomputed `{sample_id}.tiles.npz` and `{sample_id}.tiles.meta.json`
- `tiling.params`
  - Core tiling resolution, tile size, overlap, and tissue-threshold settings
- `tiling.seg_params`
  - Tissue segmentation settings
- `tiling.filter_params`
  - Contour and white/black filtering settings
- `tiling.visu_params`
  - Preview-rendering settings
- `tiling.sampling_params`
  - Annotation-specific sampling rules for `hs2p.sampling`
- `speed.num_workers`
  - Parallelism for slide processing

## Sampling-specific settings

`hs2p.sampling` uses the same base tiling setup as `hs2p.tiling`, plus:

- `tiling.sampling_params.independant_sampling`
  - Whether annotations are sampled independently or jointly
- `tiling.sampling_params.pixel_mapping`
  - Mapping from annotation names to mask pixel values
- `tiling.sampling_params.color_mapping`
  - Optional overlay colors used in previews
- `tiling.sampling_params.tissue_percentage`
  - Minimum annotation coverage required to keep a tile

## Resume and precomputed artifacts

- `resume: true` expects the current `process_list.csv` schema and current-format artifacts
- reused tiling artifacts are validated against `sample_id`, `config_hash`, `image_path`, and `mask_path`
- `tiling.read_tiles_from` is the supported way to reuse precomputed tiling outputs

For the exact output files and field meanings, see [artifacts.md](artifacts.md).
