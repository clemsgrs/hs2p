<h1 align="center">Histopathology Slide Pre-processing Pipeline</h2>


HS2P is an open-source project largely based on [CLAM](https://github.com/mahmoodlab/CLAM) tissue segmentation and patching code. 

<p>
   <a href="https://github.com/psf/black"><img alt="empty" src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
   <a href="https://github.com/PyCQA/pylint"><img alt="empty" src=https://img.shields.io/github/stars/clemsgrs/hs2p?style=social></a>
</p>

## Step-by-step guide

1. [Optional] Configure wandb

If you want to benefit from wandb logging, you need to follow these simple steps:
 - grab your wandb API key under your profile and export
 - run the following command in your terminal: `export WANDB_API_KEY=<your_personal_key>`
 - change wandb paramters in the config file under `config/` (mainly `project` and `username`)


## TODO List

- [x] add support for tissue percentage comparison to keep / exclude patches
- [ ] add support for black patch removal in latest contour processing function
- [x] add possibility to save extracted patch to disk as .png images
- [ ] improve patch saving to disk using multiprocessing
- [ ] add support for deep-learning based tissue segmentation
