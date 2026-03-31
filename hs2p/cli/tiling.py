import argparse
from pathlib import Path

from hs2p.api import tile_slides
from hs2p.configs.resolvers import (
    resolve_filter_config,
    resolve_preview_config,
    resolve_read_coordinates_from,
    resolve_segmentation_config,
    resolve_tiling_config,
)
import hs2p.progress as progress
from hs2p.utils import setup, load_csv


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
    reporter = progress.create_cli_progress_reporter(
        output_dir=getattr(args, "output_dir", None)
    )
    with progress.activate_progress_reporter(reporter):
        try:
            cfg = setup(args)
            output_dir = Path(cfg.output_dir)
            whole_slides = load_csv(cfg, mask_column="tissue_mask_path")
            tiling = resolve_tiling_config(cfg)
            segmentation = resolve_segmentation_config(cfg)
            filtering = resolve_filter_config(cfg)
            preview = resolve_preview_config(cfg)
            read_coordinates_from = resolve_read_coordinates_from(cfg)
            jpeg_backend = str(getattr(cfg.speed, "jpeg_backend", "turbojpeg"))
            progress.emit_progress(
                "run.started",
                command="tiling",
                slide_count=len(whole_slides),
                backend=tiling.backend,
                target_spacing_um=tiling.target_spacing_um,
                target_tile_size_px=tiling.target_tile_size_px,
                output_dir=str(output_dir),
                num_workers=int(cfg.speed.num_workers),
                resume=bool(cfg.resume),
                read_coordinates_from=(
                    str(read_coordinates_from) if read_coordinates_from else None
                ),
            )
            artifacts = tile_slides(
                whole_slides,
                tiling=tiling,
                segmentation=segmentation,
                filtering=filtering,
                preview=preview,
                output_dir=output_dir,
                num_workers=cfg.speed.num_workers,
                resume=cfg.resume,
                read_coordinates_from=read_coordinates_from,
                save_tiles=bool(getattr(cfg, "save_tiles", False)),
                jpeg_backend=jpeg_backend,
            )
            progress.emit_progress(
                "run.finished",
                command="tiling",
                output_dir=str(output_dir),
                process_list_path=str(output_dir / "process_list.csv"),
                logs_dir=str(output_dir / "logs"),
            )
            return artifacts
        except Exception as exc:
            progress.emit_progress("run.failed", stage="tiling", error=str(exc))
            raise


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
