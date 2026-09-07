# Tissue Mask Generation

Use `scripts/generate_tissue_mask.py` to generate binary pyramidal TIFF masks
before tiling. Pass each output to the [CLI](cli.md) through the `mask_path` CSV
column or to the [Python API](api.md) through `SlideSpec(mask_path=...)`.

## Installation

The script lives in the repository and is not included in the pip package. Clone
the repository and run the commands below from its root:

```bash
git clone https://github.com/clemsgrs/hs2p.git
cd hs2p
```

The script uses WholeSlideData for every backend. On Ubuntu/Debian, install its
native spatial-index dependency, then install the Python dependencies in a virtual
environment. This example uses OpenSlide:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends -y libspatialindex-dev
python -m pip install -e '.[openslide,asap]' tifffile
python scripts/generate_tissue_mask.py --help
```

The `asap` extra supplies WholeSlideData; using the script's default `asap`
backend also requires the native ASAP library and its Python bindings.

## Single-slide example

```bash
python scripts/generate_tissue_mask.py \
  --wsi /path/to/slide.tif \
  --output /path/to/tissue-mask-pyramid.tif \
  --backend openslide \
  --spacing 4.0 \
  --tolerance 0.1
```

## Multi-slide example

```bash
python scripts/generate_tissue_mask.py \
  --wsi '/path/to/slide_dir/*.tif' \
  --output-dir /path/to/output_dir \
  --backend openslide \
  --spacing 4.0 \
  --tolerance 0.1
```

Multi-slide outputs are named `<slide_stem>.tif`; input slides must have unique
stems within a run.

The script applies HSV thresholding and morphology to produce `0 = background`
and `1 = tissue`. A coarse-to-fine ROI shortcut is enabled by default to reduce
memory and compute. `--spacing` sets the target mask resolution in µm/px; a native
spacing within `--tolerance` is reused, otherwise the image is downsampled to the
target. The TIFF records the resulting spacing. Use `--verbose` to inspect it.

## Common options

| Options | Purpose |
| --- | --- |
| `--backend` | WholeSlideData reader; default `asap`. |
| `--output` / `--output-dir` | Single output path or directory of outputs. |
| `--num-workers` | Parallel slide processing. |
| `--spacing-at-level-0` | Override missing or incorrect slide spacing metadata. |
| `--no-cache` | Force recomputation. |
| `--disable-coarse-roi-shortcut` | Process the full frame instead of coarse tissue ROIs. |
| `--coarse-spacing`, `--coarse-roi-margin-um`, `--processing-tile-size` | Tune coarse-to-fine ROI processing. |
| `--min-component-area-um2`, `--min-hole-area-um2` | Remove small tissue components and fill small holes. |
| `--gaussian-sigma-um`, `--open-radius-um`, `--close-radius-um` | Control smoothing and morphology. |
| `--compression`, `--tile-size` | Control TIFF encoding. |

## Outputs

Alongside the masks, the script writes `summary.csv` and `cache_manifest.json`
next to `--output` or inside `--output-dir`. The summary records each slide's
input path, output path, status, and failure traceback. The manifest lets repeated
runs skip unchanged inputs and outputs when the processing options match.
