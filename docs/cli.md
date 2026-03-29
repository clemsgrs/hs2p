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
- `spacing_at_level_0` *(optional)*
  - Override for the slide's native spacing at pyramid level 0 (µm/px). Use this when the embedded metadata is missing or incorrect. All other pyramid-level spacings are rescaled proportionally from this value. Leave the column empty or omit it entirely to use the spacing reported by the file.

```csv
sample_id,image_path,mask_path,spacing_at_level_0
slide-1,/data/slide-1.tif,,0.25
slide-2,/data/slide-2.tif,/data/slide-2-mask.tif,
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

Optional CuCIM install for faster tar export with `save_tiles: true` and
`tiling.backend: cucim`:

```bash
pip install cucim-cu12
```

Use the CuCIM wheel that matches your CUDA runtime. Non-CuCIM backends continue to
use the generic reader export path.

Other optional reader extras:

```bash
pip install "hs2p[openslide]"
pip install "hs2p[asap]"
pip install "hs2p[vips]"
```

Supported `tiling.backend` values are `auto`, `cucim`, `vips`, `openslide`, and `asap`. `auto` currently prefers `cucim -> vips -> openslide -> asap`.

## Progress UX

When stdout is an interactive terminal, `hs2p` uses `rich` to show live progress for both CLI entrypoints.

- `hs2p.tiling`
  - one slide-level progress bar for batch tiling
  - discovered tile counts in the live task description
  - a final summary panel with slide totals, failures, zero-tile successes, output directory, and `process_list.csv`
- `hs2p.sampling`
  - one slide-level progress bar for batch sampling
  - cumulative kept-tile counts in the live task description
  - a final summary panel with slide totals, failures, per-annotation zero-tile counts, output directory, and `process_list.csv`

When stdout is redirected or otherwise non-interactive, `hs2p` falls back to plain-text stage updates and summaries.

Detailed logs still go to `output_dir/logs/log.txt`, which is the best place to look when a run fails.

## Current config areas

- `tiling.read_coordinates_from`
  - Optional directory containing precomputed `{sample_id}.coordinates.npz` and `{sample_id}.coordinates.meta.json`
- `tiling.params`
  - Core tiling resolution, tile size, overlap, and tissue-threshold settings
- `tiling.seg_params`
  - Tissue segmentation settings
- `tiling.filter_params`
  - Contour and white/black filtering settings
- `tiling.preview`
  - Preview-rendering settings
- `tiling.sampling_params`
  - Annotation-specific sampling rules for `hs2p.sampling`
- `save_previews`
  - Global switch for writing mask and tiling previews to disk
- `save_tiles`
  - Global switch for writing `tiles/{sample_id}.tiles.tar` alongside coordinate artifacts
- `speed.num_workers`
  - Parallelism for slide processing, and the per-slide worker budget reused by CuCIM batched tile extraction when `tiling.backend: cucim`

## Sampling-specific settings

`hs2p.sampling` uses the same base tiling setup as `hs2p.tiling`, plus:

- `tiling.sampling_params.independent_sampling`
  - Whether annotations are sampled independently or jointly
- `tiling.sampling_params.pixel_mapping`
  - Mapping from annotation names to mask pixel values
- `tiling.sampling_params.color_mapping`
  - Optional overlay colors used in previews
- `tiling.sampling_params.tissue_percentage`
  - Minimum annotation coverage required to keep a tile
- `speed.num_workers`
  - Controls how many slides `hs2p.sampling` processes in parallel. Each worker always uses one extraction thread, and the old `cfg.speed.inner_workers` override is no longer supported.

## Performance notes

### Segmentation downsample (`tiling.seg_params.downsample`)

Tissue segmentation runs once per slide on a downsampled thumbnail. The `downsample` value controls which pyramid level is used:

- **Larger value** (e.g. `64`, the default) → smaller thumbnail → faster read and less memory per worker. At 64× downsample a 256 px tile maps to a 4×4 patch in the mask, which is coarse but sufficient for interior tiles.
- **Smaller value** (e.g. `16` or `4`) → higher-resolution thumbnail → more precise tissue boundaries at tile edges, but the thumbnail read is proportionally larger and the in-memory mask grows as `1/downsample²`. Below `16`, segmentation quality rarely improves meaningfully while speed degrades noticeably.

The tissue percentage check itself (`check_coordinates`) is entirely in-memory — it does not read any pixel data from the slide; it operates on the mask computed during segmentation.

### Black/white tile filtering (`tiling.filter_params.filter_white` / `filter_black`)

These filters are **disabled by default** and should stay off unless your dataset contains a meaningful fraction of pen marks, blank regions, or background tiles that tissue segmentation does not catch.

When enabled, every candidate tile that passes the tissue mask check is read from the slide at full resolution and its pixel values inspected. This is the **only step in the tiling pipeline that reads actual tile pixel data**. For slides with large internal JPEG tiles (common in some scanner formats), each read triggers a full JPEG decode of the underlying tile block — which can be an order of magnitude slower than the rest of the pipeline per slide.

### GPU-accelerated tile decoding (`gpu_decode`)

When `save_tiles: true` and `tiling.backend: cucim`, you can enable GPU-accelerated batch decoding by passing `gpu_decode=True` to `extract_tiles_to_tar` or `tile_slides` in the Python API:

```python
from hs2p.api import tile_slides, extract_tiles_to_tar

# via tile_slides
tile_slides(..., save_tiles=True, gpu_decode=True)

# or directly
extract_tiles_to_tar(result, output_dir, gpu_decode=True)
```

When enabled, two things happen:
1. `ENABLE_CUSLIDE2=1` is set in the process environment before CuCIM is imported, activating NVIDIA's cuSlide2 GPU-accelerated SVS/TIFF reader.
2. `device="cuda"` is passed to `read_region`, so batch JPEG decoding runs on the GPU via nvImageCodec.

This can give a significant speedup (measured ~3.8× for batch decoding) on `.svs` and `.tif` files.

**Requirements:** `libnuma1` must be installed and `nvImageCodec` must be available (included with `cucim-cu12`). If the installed CuCIM version does not support `device="cuda"`, hs2p falls back silently to CPU decoding.

**Default:** `False` — opt in explicitly.

### Saved tile export (`save_tiles`)

When `save_tiles: true`, HS2P also writes a `tiles/{sample_id}.tiles.tar` archive with JPEG-encoded tile images.

- For non-CuCIM backends, tar extraction uses the generic reader path, and dense `8x8` and `4x4` tile blocks are coalesced into larger contiguous reads before slicing them back into tiles.
- For `tiling.backend: cucim`, tar extraction uses a CuCIM batch-read fast path and reuses the per-slide worker count from `speed.num_workers`.
- Installing CuCIM is optional. If `backend: cucim` is selected but CuCIM is not installed at tile-export time, HS2P raises an import error rather than silently switching backends.

## Resume and precomputed artifacts

- `resume: true` expects the current `process_list.csv` schema and current-format artifacts
- reused tiling artifacts are validated against `sample_id`, `config_hash`, `image_path`, and `mask_path`
- `tiling.read_coordinates_from` is the supported way to reuse precomputed tiling outputs

For the exact output files and field meanings, see [artifacts.md](artifacts.md).
