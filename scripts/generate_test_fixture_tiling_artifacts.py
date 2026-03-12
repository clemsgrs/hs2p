from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hs2p.api import (
    FilterConfig,
    SegmentationConfig,
    TilingConfig,
    WholeSlide,
    save_tiling_result,
    tile_slide,
)

DEFAULT_INPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "input"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "gt-current"


def build_fixture_configs(*, backend: str, tissue_threshold: float) -> tuple[TilingConfig, SegmentationConfig, FilterConfig]:
    return (
        TilingConfig(
            target_spacing_um=0.5,
            target_tile_size_px=224,
            tolerance=0.07,
            overlap=0.0,
            tissue_threshold=tissue_threshold,
            drop_holes=False,
            use_padding=True,
            backend=backend,
        ),
        SegmentationConfig(
            downsample=64,
            sthresh=8,
            sthresh_up=255,
            mthresh=7,
            close=4,
            use_otsu=False,
            use_hsv=True,
        ),
        FilterConfig(
            ref_tile_size=224,
            a_t=4,
            a_h=2,
            max_n_holes=8,
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
    )


def generate_fixture_artifacts(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_id: str = "test-wsi",
    backend: str = "asap",
    tissue_threshold: float = 0.1,
    num_workers: int = 1,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    wsi_path = input_dir / "test-wsi.tif"
    mask_path = input_dir / "test-mask.tif"

    missing = [path.name for path in (wsi_path, mask_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required fixture inputs in {input_dir}: {', '.join(missing)}"
        )

    tiling, segmentation, filtering = build_fixture_configs(
        backend=backend,
        tissue_threshold=tissue_threshold,
    )
    result = tile_slide(
        WholeSlide(sample_id=sample_id, image_path=wsi_path, mask_path=mask_path),
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        num_workers=num_workers,
    )
    return save_tiling_result(result, output_dir=output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate current-format tiling artifacts for the repository's real WSI test fixture."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing test-wsi.tif and test-mask.tif.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where {sample_id}.tiles.npz and {sample_id}.tiles.meta.json are written.",
    )
    parser.add_argument(
        "--sample-id",
        default="test-wsi",
        help="Sample identifier to embed in the generated artifacts.",
    )
    parser.add_argument(
        "--backend",
        default="asap",
        help="Whole-slide backend to use for fixture generation.",
    )
    parser.add_argument(
        "--tissue-threshold",
        type=float,
        default=0.1,
        help="Tissue threshold used when generating the tiling artifacts.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers passed to tile_slide().",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = generate_fixture_artifacts(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        backend=args.backend,
        tissue_threshold=args.tissue_threshold,
        num_workers=args.num_workers,
    )
    print(f"Wrote {artifacts.tiles_npz_path}")
    print(f"Wrote {artifacts.tiles_meta_path}")


if __name__ == "__main__":
    main()
