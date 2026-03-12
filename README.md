# hs2p

[![PyPI version](https://img.shields.io/pypi/v/hs2p?label=pypi&logo=pypi&color=3776AB)](https://pypi.org/project/hs2p/)

`hs2p` is a Python package for tiling and sampling whole-slide images at physically defined resolution. It is designed for computational pathology workflows that need reproducible coordinates, explicit artifact files, and support for masks that do not necessarily match the slide pyramid.

It supports efficient slide tiling and tile sampling at any requested spacing, whether or not that spacing is natively present in the whole-slide image. If a mask is not provided, HS2P can segment tissue directly from the slide; if you want to precompute tissue masks, a standalone script is available.

HS2P supports two main workflows:

- a Python API for library-style integration
- a CLI for batch preprocessing from a CSV and YAML config

## Installation

```bash
pip install hs2p
```

## Workflows

### Tiling

<img src="illustrations/extraction_illu.png" alt="HS2P tiling workflow" width="1000" />

Tiling computes a reproducible grid of tile coordinates for each slide and saves them as named artifacts with extraction metadata.

### Sampling

<img src="illustrations/sampling_illu.png" alt="HS2P sampling workflow" width="1000" />

Sampling filters or partitions tile coordinates by annotation coverage so you can keep only tiles relevant to a tissue class or label.

## Python API

The public API is centered on `WholeSlide`, config dataclasses, and explicit save/load functions for named artifacts.

If `mask_path` is omitted, HS2P segments tissue from the slide before extracting coordinates.

Minimal tiling example:

```python
from pathlib import Path

from hs2p import (
    FilterConfig,
    SegmentationConfig,
    TilingConfig,
    WholeSlide,
    save_tiling_result,
    tile_slide,
)

result = tile_slide(
    WholeSlide(
        sample_id="slide-1",
        image_path=Path("/data/slide-1.tif"),
        mask_path=Path("/data/slide-1-mask.tif"),
    ),
    tiling=TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
        backend="asap",
        drop_holes=False,
        use_padding=True,
    ),
    segmentation=SegmentationConfig(downsample=64, sthresh=8, sthresh_up=255, mthresh=7, close=4, use_otsu=False, use_hsv=True),
    filtering=FilterConfig(ref_tile_size=224, a_t=4, a_h=2, max_n_holes=8, filter_white=False, filter_black=False, white_threshold=220, black_threshold=25, fraction_threshold=0.9),
    num_workers=1,
)

artifacts = save_tiling_result(result, output_dir=Path("output"))
print(artifacts.tiles_npz_path)
print(artifacts.tiles_meta_path)
```

More API details: [docs/api.md](docs/api.md)

## CLI

Both CLI entrypoints use the same input CSV schema:

```csv
sample_id,image_path,mask_path
slide-1,/data/slide-1.tif,/data/slide-1-mask.tif
slide-2,/data/slide-2.tif,
```

For a first run, start from [hs2p/configs/default.yaml](hs2p/configs/default.yaml) and edit only the essentials:

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

For sampling, add `tiling.sampling_params.pixel_mapping` and `tiling.sampling_params.tissue_percentage` for the annotations you want to keep.

More CLI details: [docs/cli.md](docs/cli.md)

## Outputs

HS2P writes explicit named artifacts rather than anonymous coordinate dumps.

- Tiling writes `coordinates/{sample_id}.tiles.npz` and `coordinates/{sample_id}.tiles.meta.json`
- Sampling writes the same pair under `coordinates/<annotation>/`
- Batch runs also write `process_list.csv`

Artifact field reference: [docs/artifacts.md](docs/artifacts.md)

## Documentation

- [Documentation index](docs/README.md)
- [Python API guide](docs/api.md)
- [CLI guide](docs/cli.md)
- [Artifact format reference](docs/artifacts.md)
- [Tissue mask generation script](docs/tissue-mask-generation.md)
- [Testing and fixture notes](docs/documentation.md)

## Docker

If you prefer running HS2P in a container, a published Docker image is available:

```bash
docker pull waticlems/hs2p:latest
docker run --rm -it -v /path/to/your/data:/data waticlems/hs2p:latest
```
