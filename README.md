# hs2p

<p>
    <a href="https://pypi.org/project/hs2p"><img src="https://img.shields.io/pypi/v/hs2p.svg" alt="PyPI version"></a>
    <a href="https://pypi.org/project/hs2p"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
    <a href="https://github.com/psf/black"><img alt="empty" src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
    <a href="https://github.com/PyCQA/pylint"><img alt="empty" src=https://img.shields.io/github/stars/clemsgrs/hs2p?style=social></a>
    <a href="https://huggingface.co/spaces/waticlems/hs2p-demo"><img alt="HuggingFace Space" src="https://img.shields.io/badge/🤗%20demo-hs2p-blue"></a>
</p>

`hs2p` is a Python package for fast, scalable whole-slide tiling. You can request tiles at any spacing, whether or not that spacing is natively present in the image pyramid. It is designed for computational pathology workflows that need reproducible coordinates.

We support two main workflows:

- a Python API for library-style integration
- a CLI for batch preprocessing

## Demo

Try hs2p interactively: **[hs2p-demo on HuggingFace Spaces](https://huggingface.co/spaces/waticlems/hs2p-demo)**  
You can adjust tiling parameters (spacing, tile size, tissue threshold, overlap) and instantly see a tiling preview and tissue mask overlay.  
You can also upload your own pyramidal WSI (up to 1 GB).

## Installation

```bash
pip install hs2p
```

Backend-specific extras:

```bash
pip install "hs2p[openslide]"
pip install "hs2p[asap]"
pip install "hs2p[vips]"
pip install "hs2p[cucim]"
```

`hs2p[cucim]` pulls in `cucim-cu12`, `cupy-cuda12x`, and `nvidia-nvimgcodec-cu12` for optional batched GPU JPEG decoding during tar export when you opt in with `gpu_decode=True`. Use the cuCIM wheel that matches your CUDA runtime. The base install keeps slide-reading backends optional.

Supported backend names are `auto`, `cucim`, `vips`, `openslide`, and `asap`. `auto` currently tries `cucim -> vips -> openslide -> asap`, and `asap` is the compatibility wrapper around `wholeslidedata`.

## Workflows

### Tiling

Tiling computes a reproducible grid of tile coordinates for each slide and saves them as named artifacts with extraction metadata, ready for downstream use.  
When a precomputed tissue mask is not provided, `hs2p` segments tissue on-the-fly. If you want to precompute tissue masks, a [standalone script](docs/tissue-mask-generation.md) is available.

<img src="assets/tiling.png" alt="hs2p tiling workflow" width="1000" />

### Sampling

Sampling filters or partitions tile coordinates by annotation coverage so you can keep only tiles relevant to a tissue class or label.

<img src="illustrations/sampling_illu.png" alt="hs2p sampling workflow" width="1000" />

## Python API

`hs2p` supports pre-extracted tissue masks. If you don't have such tissue masks, you can either:

- use our standalone [tissue segmentation script](docs/tissue-mask-generation.md) (Recommended)
- tune the SegmentationConfig parameters and let `hs2p` segments tissue on the fly

Minimal tiling example:

```python
from pathlib import Path

from hs2p import (
    SlideSpec,
    TilingConfig,
    overlay_mask_on_slide,
    save_tiling_result,
    tile_slide,
    write_tiling_preview,
)

result = tile_slide(
    SlideSpec(
        sample_id="slide-1",
        image_path=Path("/data/wsi/slide-1.tif"),
        mask_path=Path("/data/mask/slide-1.tif"),
    ),
    tiling=TilingConfig(
        backend="openslide",
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
    ),
)

artifacts = save_tiling_result(result, output_dir=Path("output"))

print(artifacts.coordinates_npz_path)   # output/tiles/slide-1.coordinates.npz ; more info in docs/artifacts.md
print(artifacts.coordinates_meta_path)  # output/tiles/slide-1.coordinates.meta.json ; more info in docs/artifacts.md

tiling_preview_path = write_tiling_preview(
    result=result,
    output_dir=Path("output"),
    downsample=32,
)
print(tiling_preview_path)  # output/preview/tiling/slide-1.jpg ; low resolution preview of tiling result, good for QC

mask_overlay = overlay_mask_on_slide(
    wsi_path=result.image_path,
    annotation_mask_path=Path("/data/mask/slide-1.tif"),
    downsample=32,
    backend=result.backend,
)
mask_overlay.save("output/preview/mask/slide-1.jpg")
```

`result` is a [`TilingResult`](hs2p/api.py#L144) for one slide. It gives downstream pipelines the tile coordinates plus the metadata needed to relate those coordinates back to the slide pyramid and persist them as reusable named artifacts.

More API details: [docs/api.md](docs/api.md)

## CLI

The CLI is intended for fast batch processing of multiple slides with the same config. Both CLI entrypoints expect the same input `csv` schema:

```csv
sample_id,image_path,mask_path
slide-1,/data/wsi/slide-1.tif,/data/mask/slide-1.tif
slide-2,/data/wsi/slide-2.tif,
```

For a first run, start from [hs2p/configs/default.yaml](hs2p/configs/default.yaml) and edit only the essentials:

- `csv`
- `output_dir`
- `tiling.backend`
- `tiling.params.target_spacing_um`
- `tiling.params.target_tile_size_px`

Optional:

- `save_tiles`
  - also write `tiles/{sample_id}.tiles.tar` archives; with `tiling.backend="cucim"` this uses batched CuCIM reads during tar extraction, and other backends use the generic reader path with dense `8x8` / `4x4` regions coalesced before slicing them back into tiles

Run tiling:

```bash
python -m hs2p.tiling --config-file /path/to/config.yaml
```

Run sampling:

```bash
python -m hs2p.sampling --config-file /path/to/config.yaml
```

For sampling, add `tiling.sampling_params.pixel_mapping` and `tiling.sampling_params.tissue_percentage` for the annotations you want to keep.

### Progress UX

When stdout is an interactive terminal, both CLI entrypoints show live `rich` progress with:

- slide-level batch progress
- elapsed and remaining time
- live tile counts for tiling discovery or sampling retention
- final summary panels with output and `process_list.csv` locations

When stdout is redirected or otherwise non-interactive, `hs2p` falls back to concise plain-text stage updates.

If a run fails, check `output_dir/logs/log.txt` for the full log stream.

More CLI details: [docs/cli.md](docs/cli.md)

## Outputs

`hs2p` writes explicit named artifacts rather than anonymous coordinate dumps.

- Tiling writes `tiles/{sample_id}.coordinates.npz` and `tiles/{sample_id}.coordinates.meta.json`
- Sampling writes the same pair under `tiles/<annotation>/`
- Batch runs also write `process_list.csv`
- Saved coordinate arrays use a deterministic column-major order: numeric `x` first, then numeric `y` within each shared `x`

Artifact field reference: [docs/artifacts.md](docs/artifacts.md)

## Docker

[![Docker Version](https://img.shields.io/docker/v/waticlems/hs2p?sort=semver&label=docker&logo=docker&color=2496ED)](https://hub.docker.com/r/waticlems/hs2p)

If you prefer running `hs2p` in a container, a published Docker image is available:

```bash
docker pull waticlems/hs2p:latest
docker run --rm -it -v /path/to/your/data:/data waticlems/hs2p:latest
```

## Documentation

- [Documentation index](docs/README.md)
- [Python API guide](docs/api.md)
- [CLI guide](docs/cli.md)
- [Artifact format reference](docs/artifacts.md)
- [Tissue mask generation script](docs/tissue-mask-generation.md)
