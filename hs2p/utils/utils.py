import os
import wandb
import random
import subprocess
import numpy as np
import pandas as pd

from typing import Optional
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from hs2p.api import WholeSlide


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def initialize_wandb(
    cfg: DictConfig,
    key: Optional[str] = "",
):
    command = f"wandb login {key}"
    subprocess.call(command, shell=True)
    if cfg.wandb.tags is None:
        tags = []
    else:
        tags = [str(t) for t in cfg.wandb.tags]
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.wandb.resume_id:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.username,
            name=cfg.wandb.exp_name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            config=config,
            tags=tags,
            id=cfg.wandb.resume_id,
            resume="must",
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.username,
            name=cfg.wandb.exp_name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            config=config,
            tags=tags,
        )
    return run


def load_csv(cfg):
    csv_path = Path(cfg.csv).resolve()
    df = pd.read_csv(csv_path)
    required = {"sample_id", "image_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )
    if df["sample_id"].duplicated().any():
        duplicates = sorted(df.loc[df["sample_id"].duplicated(), "sample_id"].astype(str).unique())
        raise ValueError(
            "Duplicate sample_id values are not allowed: " + ", ".join(duplicates)
        )
    mask_series = df["mask_path"] if "mask_path" in df.columns else pd.Series([None] * len(df))
    whole_slides = []
    for sample_id, image_path, mask_path in zip(
        df["sample_id"].astype(str).tolist(),
        df["image_path"].tolist(),
        mask_series.tolist(),
    ):
        whole_slides.append(
            WholeSlide(
                sample_id=sample_id,
                image_path=Path(image_path),
                mask_path=Path(mask_path) if mask_path is not None and not pd.isna(mask_path) else None,
            )
        )
    return whole_slides
