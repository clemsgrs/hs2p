# Tissue Mask Generation

For standalone binary tissue-mask generation outside the main tiling pipeline, use:

```bash
python scripts/generate_tissue_mask.py --help
```

This script produces pyramidal tissue masks that can later be consumed by:

- CLI tiling through the `mask_path` CSV column
- Python workflows through `SlideSpec(mask_path=...)`

## Installation

At minimum you need a whole-slide backend plus `tifffile`:

```bash
pip install "hs2p[asap]" tifffile
```

or install the full backend set:

```bash
pip install "hs2p[all]" tifffile
```

## Single-slide example

```bash
python scripts/generate_tissue_mask.py \
  --wsi /path/to/slide.tif \
  --output /path/to/tissue-mask-pyramid.tif \
  --spacing 4.0 \
  --tolerance 0.1
```

## Multi-slide example

```bash
python scripts/generate_tissue_mask.py \
  --wsi /path/to/slide_dir/*.tif \
  --output-dir /path/to/output_dir \
  --spacing 4.0 \
  --tolerance 0.1
```

## What it does

- reads the slide through the selected whole-slide backend
- computes a binary mask with `0 = background`, `1 = tissue`
- can use a coarse-to-fine ROI shortcut to reduce memory and compute
- writes a pyramidal TIFF mask at the requested spacing
- writes a summary CSV for the run

## Common options

- `--backend`
  - whole-slide backend, default `asap`
- `--output` / `--output-dir`
  - output path for single-slide or multi-slide mode
- `--num-workers`
  - parallelism for multi-slide processing
- `--spacing-at-level-0`
  - override when slide metadata is missing or wrong
- `--no-cache`
  - disable cache-based skipping
- `--disable-coarse-roi-shortcut`
  - force full-frame processing at the requested spacing
- `--coarse-spacing`, `--coarse-roi-margin-um`, `--processing-tile-size`
  - coarse-to-fine ROI tuning
- `--min-component-area-um2`, `--min-hole-area-um2`
  - morphology cleanup thresholds
- `--gaussian-sigma-um`, `--open-radius-um`, `--close-radius-um`
  - smoothing and morphology controls
- `--compression`, `--tile-size`
  - TIFF output controls

## Outputs

- `summary.csv`
  - written next to `--output` or inside `--output-dir`
- `cache_manifest.json`
  - cache manifest used for skip detection
