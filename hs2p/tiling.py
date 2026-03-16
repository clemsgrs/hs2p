import argparse
from pathlib import Path

import pandas as pd

from hs2p.api import (
    QCConfig,
    _build_cli_configs,
    tile_slides,
)
from hs2p.utils import setup, load_csv, fix_random_seeds


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("hs2p", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--skip-datetime", action="store_true", help="skip run id datetime prefix"
    )
    parser.add_argument(
        "--skip-logging", action="store_true", help="skip logging configuration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "opts",
        help='Modify config options at the end of the command using "path.key=value".',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main(args):
    cfg = setup(args)
    output_dir = Path(cfg.output_dir)
    fix_random_seeds(cfg.seed)
    whole_slides = load_csv(cfg)
    tiling, segmentation, filtering = _build_cli_configs(cfg)
    qc = QCConfig(
        save_mask_preview=bool(cfg.visualize),
        save_tiling_preview=bool(cfg.visualize),
        downsample=cfg.tiling.visu_params.downsample,
    )
    artifacts = tile_slides(
        whole_slides,
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        qc=qc,
        output_dir=output_dir,
        num_workers=cfg.speed.num_workers,
        resume=cfg.resume,
        read_tiles_from=(
            Path(cfg.tiling.read_tiles_from) if cfg.tiling.read_tiles_from else None
        ),
    )
    process_df = pd.read_csv(output_dir / "process_list.csv")
    failed_tiling = process_df[process_df["tiling_status"] == "failed"]
    no_tiles = process_df[
        (process_df["tiling_status"] == "success") & (process_df["num_tiles"] == 0)
    ]
    print("=+=" * 10)
    print(f"Total number of slides: {len(process_df)}")
    print(f"Failed tiling: {len(failed_tiling)}")
    print(f"No tiles after tiling step: {len(no_tiles)}")
    print(f"Slides with tiles: {sum(a.num_tiles > 0 for a in artifacts)}")
    print("=+=" * 10)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
