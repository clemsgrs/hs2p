import logging
import os
import datetime

from pathlib import Path
from omegaconf import OmegaConf

from hs2p.utils import initialize_wandb, fix_random_seeds, get_sha, setup_logging
from hs2p.configs import default_config

logger = logging.getLogger("hs2p")


def write_config(cfg, output_dir, name="config.yaml", skip_logging: bool = False):
    if not skip_logging:
        logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_file(config_file):
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(config_file)
    cfg = OmegaConf.merge(default_cfg, cfg)
    OmegaConf.resolve(cfg)
    return cfg


def get_cfg_from_args(args):
    if args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)
        args.opts += [f"output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    OmegaConf.resolve(cfg)
    return cfg


def setup(args):
    """
    Basic configuration setup.
    This function:
      - Loads the config from file and command-line options.
      - Sets up logging.
      - Fixes random seeds.
      - Creates the output directory.
    """
    cfg = get_cfg_from_args(args)

    if cfg.resume:
        run_id = cfg.resume_dirname
    elif not args.skip_datetime:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    else:
        run_id = ""

    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, run_id)
    output_dir.mkdir(exist_ok=cfg.resume or args.skip_datetime, parents=True)
    cfg.output_dir = str(output_dir)

    fix_random_seeds(0)
    setup_logging(output=cfg.output_dir, level=logging.INFO)
    logger.info("git:\n  {}\n".format(get_sha()))
    cfg_path = write_config(cfg, cfg.output_dir, skip_logging=args.skip_logging)
    if cfg.wandb.enable:
        wandb_run.save(cfg_path)
    return cfg
