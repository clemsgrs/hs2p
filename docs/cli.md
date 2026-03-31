# CLI Guide

HS2P provides two batch entrypoints:

- `python -m hs2p.tiling`
- `python -m hs2p.sampling`

They share the same base tiling/segmentation/filtering config model, but they do not share the same public CSV mask column anymore.

## Input CSV schemas

### Tiling

```csv
sample_id,image_path,tissue_mask_path
slide-1,/data/slide-1.tif,/data/slide-1-tissue-mask.tif
slide-2,/data/slide-2.tif,
...
```

### Sampling

```csv
sample_id,image_path,annotation_mask_path
slide-1,/data/slide-1.tif,/data/slide-1-annotations.tif
slide-2,/data/slide-2.tif,/data/slide-2-annotations.tif
...
```

### Optional spacing override

Works in either mode:

```csv
sample_id,image_path,tissue_mask_path,spacing_at_level_0
slide-1,/data/slide-1.tif,,0.25
slide-2,/data/slide-2.tif,/data/slide-2-tissue-mask.tif,
...
```

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

## Installation and backends

Base install:

```bash
pip install hs2p
```

Optional extras:

```bash
pip install "hs2p[openslide]"
pip install "hs2p[asap]"
pip install "hs2p[vips]"
pip install "hs2p[cucim]"
pip install "hs2p[all]"
```

`tiling.backend` supports:

- `auto`
- `cucim`
- `vips`
- `openslide`
- `asap`

`auto` prefers `cucim -> vips -> openslide -> asap`.

## Config areas

- `tiling.read_coordinates_from`
  - Reuse precomputed `{sample_id}.coordinates.*` artifacts
- `tiling.params`
  - spacing, tile size, overlap, tolerance, padding, and minimum tissue fraction
- `tiling.seg_params`
  - tissue segmentation settings
- `tiling.filter_params`
  - contour and optional white/black filtering settings
- `tiling.preview`
  - preview rendering settings
- `tiling.sampling_params`
  - annotation-specific sampling rules for `hs2p.sampling`
- `save_previews`
  - write preview images
- `save_tiles`
  - write `tiles/{sample_id}.tiles.tar`
- `speed.num_workers`
  - slide-level batch parallelism

## Sampling-specific config

`hs2p.sampling` adds:

- `tiling.sampling_params.independent_sampling`
- `tiling.sampling_params.pixel_mapping`
- `tiling.sampling_params.color_mapping`
- `tiling.sampling_params.tissue_percentage`

Sampling config resolution is strict:

- explicit configs must include `background`
- partial sampling configs are rejected
- color mappings are validated centrally

## Progress reporting

When stdout is interactive, both entrypoints use `rich` live progress:

- tiling shows discovered tile totals during the run
- sampling shows kept-tile totals during the run
- both finish with summary panels including output locations and `process_list.csv`

When stdout is non-interactive, `hs2p` falls back to concise plain-text progress and summary logs.

Detailed logs still go to `output_dir/logs/log.txt`.

## Resume and precomputed artifacts

- `resume: true` expects the current process-list schema
- reused artifacts are validated against structured metadata, not `config_hash`
- `tiling.read_coordinates_from` is the supported way to reuse precomputed coordinate artifacts

## Performance notes

### Segmentation downsample

`tiling.seg_params.downsample` controls the resolution used for tissue segmentation:

- larger values are faster and coarser
- smaller values improve edge precision but cost more time and memory

### Tile pixel QC

`tiling.filter_params.filter_white`, `filter_black`, `filter_grayspace`, and `filter_blur` are disabled by default.

When enabled, HS2P evaluates candidate tiles at `tiling.filter_params.qc_spacing_um`, which is typically coarser than the final extraction spacing. This is still slower than mask-only tiling, but cheaper than running pixel QC at the requested tile spacing.

### Tile tar export

When `save_tiles: true`, HS2P also writes `tiles/{sample_id}.tiles.tar`.

- non-CuCIM paths coalesce dense tile regions before slicing them back into tiles
- CuCIM paths use batched reads
- `gpu_decode=True` is opt-in in the Python API for CuCIM tar export

## Outputs

See [artifacts.md](artifacts.md) for the exact coordinate artifact schema and process-list columns.
