#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hs2p import (
    FilterConfig,
    SegmentationConfig,
    TilingConfig,
    WholeSlide,
    overlay_mask_on_slide,
    tile_slide,
    write_tiling_preview,
)


def _default_paths() -> tuple[Path, Path, Path]:
    fixtures_dir = REPO_ROOT / "tests" / "fixtures"
    return (
        fixtures_dir / "input" / "test-wsi.tif",
        fixtures_dir / "input" / "test-mask.tif",
        fixtures_dir / "gt",
    )


def _resolve_backend(requested: str, wsi_path: Path) -> str:
    if requested != "auto":
        return requested

    import wholeslidedata as wsd

    for backend in ("asap", "openslide"):
        try:
            wsd.WholeSlideImage(wsi_path, backend=backend)
            return backend
        except Exception:
            continue
    raise RuntimeError(
        f"Unable to open {wsi_path} with any supported backend (tried: asap, openslide)"
    )


def _build_parser() -> argparse.ArgumentParser:
    default_wsi_path, default_mask_path, default_output_dir = _default_paths()
    parser = argparse.ArgumentParser(
        description="Regenerate the checked-in fixture visualization images with the current hs2p code."
    )
    parser.add_argument("--wsi-path", type=Path, default=default_wsi_path)
    parser.add_argument("--mask-path", type=Path, default=default_mask_path)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--sample-id", default="test-wsi")
    parser.add_argument(
        "--backend",
        default="auto",
        help="Slide backend to use (for example: asap, openslide, or auto).",
    )
    parser.add_argument("--downsample", type=int, default=32)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    backend = _resolve_backend(args.backend, args.wsi_path)

    tiling = TilingConfig(
        backend=backend,
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
        drop_holes=False,
        use_padding=True,
    )
    segmentation = SegmentationConfig(
        downsample=64,
        sthresh=8,
        sthresh_up=255,
        mthresh=7,
        close=4,
        use_otsu=False,
        use_hsv=True,
    )
    filtering = FilterConfig(
        ref_tile_size=224,
        a_t=4,
        a_h=2,
        max_n_holes=8,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )

    result = tile_slide(
        WholeSlide(
            sample_id=args.sample_id,
            image_path=args.wsi_path,
            mask_path=args.mask_path,
        ),
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        num_workers=1,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        temp_tiling_path = write_tiling_preview(
            result=result,
            output_dir=temp_dir,
            downsample=args.downsample,
        )
        if temp_tiling_path is None:
            raise RuntimeError("No tiling preview was generated because the result has zero tiles")
        tiling_output_path = args.output_dir / "tiling-visu.jpg"
        temp_tiling_path.replace(tiling_output_path)

    mask_overlay = overlay_mask_on_slide(
        wsi_path=result.image_path,
        annotation_mask_path=args.mask_path,
        downsample=args.downsample,
        backend=backend,
    )
    mask_output_path = args.output_dir / "mask-visu.jpg"
    mask_overlay.save(mask_output_path)

    print(f"Backend: {backend}")
    print(f"Tiling preview: {tiling_output_path}")
    print(f"Mask preview: {mask_output_path}")


if __name__ == "__main__":
    main()
