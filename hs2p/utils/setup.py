import logging
import os
import datetime

from pathlib import Path
from omegaconf import OmegaConf

from hs2p.utils import initialize_wandb, fix_random_seeds, get_sha, setup_logging
from hs2p.configs import default_config
from hs2p.configs.keys import reject_unknown_config_keys
from hs2p.configs.models import RunSettings
from hs2p.configs.values import check_config_values

logger = logging.getLogger("hs2p")


def write_config(*, cfg, output_dir, name="config.yaml", skip_logging: bool = False):
    if not skip_logging:
        logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_file(config_file):
    return _load_validated(OmegaConf.load(config_file))


def get_cfg_from_args(args):
    if args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)
        args.opts += [f"output_dir={args.output_dir}"]
    return _load_validated(
        OmegaConf.load(args.config_file), OmegaConf.from_cli(args.opts)
    )


#: Config paths of the scalars the CLI reads outside the typed configs; ``RunSettings``
#: declares their types.
_RUN_SETTING_PATHS = (
    "resume",
    "save_tiles",
    "seed",
    "speed.num_workers",
    "speed.jpeg_backend",
)

_ABSENT = object()


def _load_validated(*layers):
    """Merge ``layers`` onto the defaults, rejecting unknown keys before and after
    resolution, and wrong-typed run settings after it.

    The first key check runs on the raw tree, so an unknown key is reported even when its
    value is an interpolation that cannot resolve. The second catches keys that only
    appear once a section-level interpolation (``speed: ${wandb.speed}``) resolves. Value
    types are checked on the resolved tree, where interpolations hold their values. The
    typed configs' sections are checked when the resolvers construct them.

    A run setting that is absent is not checked: an interpolated section replaces the
    default one, and the CLI keeps its own fallback for a key it leaves out
    (``speed.jpeg_backend``). An explicit ``null`` is present, and is checked.
    """
    cfg = OmegaConf.merge(OmegaConf.create(default_config), *layers)
    reject_unknown_config_keys(cfg)
    OmegaConf.resolve(cfg)
    reject_unknown_config_keys(cfg)
    present = {
        path: OmegaConf.select(cfg, path, default=_ABSENT)
        for path in _RUN_SETTING_PATHS
    }
    check_config_values(
        RunSettings,
        {path: value for path, value in present.items() if value is not _ABSENT},
    )
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

    fix_random_seeds(int(cfg.seed))
    setup_logging(output=cfg.output_dir, level=logging.INFO)
    logger.info("git:\n  {}\n".format(get_sha()))
    cfg_path = write_config(
        cfg=cfg, output_dir=cfg.output_dir, skip_logging=args.skip_logging
    )
    if cfg.wandb.enable:
        wandb_run.save(cfg_path)
    return cfg
