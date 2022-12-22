<h1 align="center">Histopathology Slide Pre-processing Pipeline</h2>


HS2P is an open-source project largely based on [CLAM](https://github.com/mahmoodlab/CLAM) tissue segmentation and patching code. 

<p>
   <a href="https://github.com/psf/black"><img alt="empty" src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
   <a href="https://github.com/PyCQA/pylint"><img alt="empty" src=https://img.shields.io/github/stars/clemsgrs/hs2p?style=social></a>
</p>

<img src="illustration.png" width="1000px" align="center" />

## Requirements

install requirements via `pip3 install -r requirements.txt`

## Step-by-step guide

1. [Optional] Configure wandb

If you want to benefit from wandb logging, you need to follow these simple steps:
 - grab your wandb API key under your profile and export
 - run the following command in your terminal: `export WANDB_API_KEY=<your_personal_key>`
 - change wandb paramters in the configuration file under `config/` (set `enable` to `True`)

2. Create a .txt file containing paths to the desired slides:

```
path/to/slide_1.tif
slightly/different/path/to/slide_2.tif
...
```

3. Create a configuration file under `config/`.<br>
A good starting point is to use the default configuration file `config/default.yaml` where parameters are documented.

4. Run the following command to kick off the algorithm:

`python3 main.py --config-name <config_filename>`

5. Depending on which flags have been set to True, it will produce (part of) the following results:


```
hs2p/ 
├── output/<experiment_name>/
│     ├── masks/
│     │     ├── slide_1.jpg
│     │     ├── slide_2.jpg
│     │     └── ...
│     ├── patches/
│     │        ├── slide_1/
│     │        │   └── <patch_size>/
│     │        │       ├── slide_1.h5
│     │        │       └── jpg/
│     │        │          ├── x0_y0.jpg
│     │        │          ├── x1_y0.jpg
│     │        │          └── ...
│     │        ├── slide_2/
│     │        └── ...
│     ├── visualization/
│     │     └── <patch_size>/
│     │         ├── slide_1.jpg
│     │         ├── slide_2.jpg
│     │         └── ...
│     └── process_list.csv
```

Extracted patches will be saved as `x_y.jpg` where `x` and `y` represent the true location in the slide **at level 0**:
- if spacing at level 0 is `0.25` and you extract [256, 256] patches at spacing `0.25`, two consecutive patches will be distant from `256` pixels (either along `x` or `y` axis)
- if spacing at level 0 is `0.25` and you extract [256, 256] patches at spacing `0.5`, two consecutive patches will be distant from `512` pixels (either along `x` or `y` axis)

## Resuming experiment after crash / bug

If, for some reason, the experiment crashes, you should be able to resume from last processed slide simply by turning the `resume` parameter in your config file to `True`, keeping all other parameters unchanged.

## Troubleshooting

If the generated visualization are noisy, you'll need to change `libpixman` version. Running the following command should fix this issue:

```
sudo -S wget https://www.cairographics.org/releases/pixman-0.40.0.tar.gz
sudo -S tar -xf pixman-0.40.0.tar.gz
cd pixman-0.40.0
sudo -S ./configure
sudo -S make
sudo -S make install

export LD_PRELOAD=/usr/local/lib/libpixman-1.so.0.40.0
```

## TODO List

- [ ] improve documentation
- [ ] add support for deep-learning based tissue segmentation
- [ ] add support for black patch removal in latest contour processing function
- [ ] make patch saving to disk faster (using multiprocessing?)
