<h1 align="center">Histopathology Slide Pre-processing Pipeline</h1>

[![PyPI version](https://img.shields.io/pypi/v/hs2p?label=pypi&logo=pypi&color=3776AB)](https://pypi.org/project/hs2p/)
[![Docker Version](https://img.shields.io/docker/v/waticlems/hs2p?sort=semver&label=docker&logo=docker&color=2496ED)](https://hub.docker.com/r/waticlems/hs2p)


HS2P is an open-source project largely based on [CLAM](https://github.com/mahmoodlab/CLAM) tissue segmentation and patching code.

<p>
   <a href="https://github.com/psf/black"><img alt="empty" src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
   <a href="https://github.com/PyCQA/pylint"><img alt="empty" src=https://img.shields.io/github/stars/clemsgrs/hs2p?style=social></a>
</p>

## 🛠️ Installation

System requirements: Linux-based OS (e.g., Ubuntu 22.04) with Python 3.11+ and Docker installed.

We recommend running the script inside a container using the latest `hs2p` image from Docker Hub:

```shell
docker pull waticlems/hs2p:latest
docker run --rm -it \
    -v /path/to/your/data:/data \
    waticlems/hs2p:latest
```

Replace `/path/to/your/data` with your local data directory.

Alternatively, you can install `hs2p` via pip:

```shell
pip install hs2p
```

## Slide tiling

<img src="illustrations/extraction_illu.png" width="1000px" align="center" />

1. Create a `.csv` file containing paths to the desired slides. Optionally, you can provide paths to pre-computed tissue masks under the 'mask_path' column

    ```csv
    wsi_path,mask_path
    /path/to/slide1.tif,/path/to/mask1.tif
    /path/to/slide2.tif,/path/to/mask2.tif
    ...
    ```

2. Create a configuration file

    A good starting point is to look at the default configuration file under `hs2p/configs/default.yaml` where parameters are documented.

3. Kick off slide tiling

    ```shell
    python3 -m hs2p.tiling --config-file </path/to/config.yaml>
    ```

## Tile sampling

<img src="illustrations/sampling_illu.png" width="1000px" align="center" />

1. Create a `.csv` file containing paths to the desired slides & associated annotation masks:

    ```csv
    wsi_path,mask_path
    /path/to/slide1.tif,/path/to/mask1.tif
    /path/to/slide2.tif,/path/to/mask2.tif
    ...
    ```

2. Create a configuration file

    A good starting point is to look at the default configuration file under `hs2p/configs/default.yaml` where parameters are documented.

3. Kick off tile sampling

    ```shell
    python3 -m hs2p.sampling --config-file </path/to/config.yaml>
    ```

## Output structure

Both `tiling.py` and `sampling.py` produce a similar output structure in the specified output directory.

### Coordinates

The `coordinates/` folder contains a `.npy` file for each successfully processed slide.  
This file stores a numpy array of shape `(num_tiles, 7)` containing the following information for each tile:

1. **`x`**: x-coordinate of the tile at level 0
2. **`y`**: y-coordinate of the tile at level 0
3. **`tile_level`**: pyramid level at which the tile was extracted
4. **`tile_size_resized`**: size of the tile at the extraction level, which may differ from the requested tile size if the target spacing was not available
5. **`resize_factor`**: ratio between `tile_size_resized` and the requested tile size, useful for resizing when loading the tile
6. **`tile_size_lv0`**: tile size scaled to the slide's level 0
7. **`target_spacing`**: spacing at which the user requested the tile (in microns per pixel)

### Visualization (optional)

If `visualize` is set to `true`, a `visualization/` folder is created containing low-resolution images to verify the results:

- **`mask/`**: visualizations of the provided tissue (or annotation) mask
- **`tiling/`** (for `tiling.py`) or **`sampling/`** (for `sampling.py`): visualizations of the extracted or sampled tiles overlaid on the slide. For `sampling.py`, this includes subfolders for each category defined in the sampling parameters (e.g., tumor, stroma, etc.)

These visualizations are useful for double-checking that the tiling or sampling process ran as expected.

### Process summary

- **`process_list.csv`**: a summary file listing each processed slide, indicating whether processing was successful or failed. If a failure occurred, the traceback is provided to help diagnose the issue.
