# Tissue Mask Generation

For standalone tissue-mask creation outside the main tiling pipeline, use:

```bash
python scripts/generate_tissue_mask.py --help
```

## Installation note

The script relies on `tifffile` for pyramidal TIFF output:

```bash
python -m pip install tifffile
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

- reads the WSI through the selected backend
- computes a binary tissue mask with `0=background` and `1=tissue`
- uses a coarse-to-fine ROI shortcut by default to reduce memory use
- writes a pyramidal TIFF mask at the requested spacing
- prints a final success/skip/failure summary

## Common options

- `--backend`
  - Whole-slide backend, default `asap` in this script; `asap` is the `wholeslidedata` compatibility adapter
- `--output` / `--output-dir`
  - Output path for single-slide or multi-slide mode
- `--num-workers`
  - Parallelism for multi-slide processing
- `--no-cache`
  - Disable cache-based skipping
- `--disable-coarse-roi-shortcut`
  - Force legacy full-frame loading at target spacing
- `--coarse-spacing`, `--coarse-roi-margin-um`, `--processing-tile-size`
  - Coarse-to-fine ROI tuning
- `--min-component-area-um2`, `--min-hole-area-um2`
  - Morphology cleanup thresholds
- `--gaussian-sigma-um`, `--open-radius-um`, `--close-radius-um`
  - Smoothing and morphology controls
- `--spacing-at-level-0`
  - Override level-0 spacing when metadata is incorrect
- `--compression`, `--tile-size`
  - TIFF output controls

## Outputs

- `summary.csv`
  - Run summary written next to `--output` or into `--output-dir`
- `cache_manifest.json`
  - Cache manifest used for skip inference in the same output location
