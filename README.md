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

## Patch Extraction: Step-by-step guide

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

## Patch Sampling: Step-by-step guide

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
