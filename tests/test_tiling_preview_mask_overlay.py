"""Regression: annotation tiling previews must render against a real (openslide-backed)
slide, not crash with ``'WSI' object has no attribute 'read_level'``.

The mask backdrop for the tiling preview must be read via a *reader* (which exposes
``read_level``/``spacings``), not via a ``WSI`` (which does not expose ``read_level``).
A fake backend whose WSI happens to expose ``read_level`` hides this gap, so this test
drives the real fixture slide + a multi-label pyramidal mask through ``tile_slides`` with
``save_tiling_preview`` enabled, and asserts the JPEG is actually produced under
``preview/tiling/``. It fails on main (the crash) and passes after the fix.
"""

import importlib.util
from pathlib import Path

import numpy as np
import openslide
import pytest

from hs2p.api import (
    FilterConfig,
    PreviewConfig,
    SegmentationConfig,
    SlideSpec,
    TilingConfig,
    tile_slides,
)
from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy, SamplingSpec

pytestmark = pytest.mark.integration

PIXEL_MAPPING = {"background": 0, "tumor": 1, "stroma": 2}
COLOR_MAPPING = {
    "background": None,
    "tumor": [220, 40, 40],
    "stroma": [40, 80, 220],
}


def _load_mask_pyramid_writer():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "generate_tissue_mask.py"
    if not script_path.is_file():
        pytest.skip("generate_tissue_mask.py helper is not present")
    spec = importlib.util.spec_from_file_location("generate_tissue_mask", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_pyramidal_mask_tiff


def _build_multilabel_mask(binary_mask_path: Path, output_path: Path) -> None:
    """Derive a multi-label pyramidal mask from the binary fixture mask.

    Splits the tissue region of ``test-mask.tif`` (values {0,1}) into two labels:
    tumor (=1) on the left half, stroma (=2) on the right half, keeping the mask
    registered to ``test-wsi.tif``. Written as a backend-readable pyramidal TIFF.
    """
    write_pyramidal_mask_tiff = _load_mask_pyramid_writer()
    src = openslide.OpenSlide(str(binary_mask_path))
    level0_spacing = float(src.properties.get("openslide.mpp-x") or 4.032)
    levels: list[np.ndarray] = []
    for level in range(src.level_count):
        w, h = src.level_dimensions[level]
        tissue = np.asarray(src.read_region((0, 0), level, (w, h)))[..., 0] > 0
        labelled = np.zeros((h, w), dtype=np.uint8)
        mid = w // 2
        labelled[:, :mid][tissue[:, :mid]] = 1  # tumor (left half)
        labelled[:, mid:][tissue[:, mid:]] = 2  # stroma (right half)
        levels.append(labelled)
    write_pyramidal_mask_tiff(
        levels=levels,
        output_path=output_path,
        level0_spacing=level0_spacing,
        downsample_per_level=2.0,
        compression="deflate",
        tile_size=256,
    )


def _require_openslide(wsi_path: Path) -> None:
    try:
        openslide.OpenSlide(str(wsi_path))
    except Exception:
        pytest.skip("openslide cannot open the fixture slide")


def _configs() -> tuple[TilingConfig, SegmentationConfig, FilterConfig, PreviewConfig]:
    tiling = TilingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
        backend="openslide",
    )
    segmentation = SegmentationConfig(
        method="hsv",
        downsample=64,
        sthresh=8,
        sthresh_up=255,
        mthresh=7,
        close=4,
    )
    filtering = FilterConfig(
        ref_tile_size=224,
        a_t=1,
        a_h=1,
        filter_white=False,
        filter_black=False,
    )
    preview = PreviewConfig(save_mask_preview=True, save_tiling_preview=True, downsample=64)
    return tiling, segmentation, filtering, preview


def _sampling_spec() -> SamplingSpec:
    return SamplingSpec(
        pixel_mapping=PIXEL_MAPPING,
        color_mapping=COLOR_MAPPING,
        tissue_percentage={"background": None, "tumor": 0.1, "stroma": 0.1},
        active_annotations=("tumor", "stroma"),
    )


@pytest.mark.parametrize(
    "output_mode, selection_strategy",
    [
        (CoordinateOutputMode.PER_ANNOTATION, CoordinateSelectionStrategy.INDEPENDENT_SAMPLING),
        (CoordinateOutputMode.MERGED, CoordinateSelectionStrategy.JOINT_SAMPLING),
    ],
)
def test_annotation_tiling_preview_renders_on_real_slide(
    real_fixture_paths, tmp_path: Path, output_mode, selection_strategy
):
    wsi_path, binary_mask_path = real_fixture_paths
    _require_openslide(wsi_path)

    mask_path = tmp_path / "multilabel-mask.tif"
    _build_multilabel_mask(binary_mask_path, mask_path)

    tiling, segmentation, filtering, preview = _configs()
    output_dir = tmp_path / "out"

    artifacts = tile_slides(
        [SlideSpec(sample_id="test-wsi", image_path=wsi_path, mask_path=mask_path)],
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        preview=preview,
        output_dir=output_dir,
        num_workers=1,
        sampling=_sampling_spec(),
        selection_strategy=selection_strategy,
        output_mode=output_mode,
    )

    # A non-empty artifact must exist so a tiling preview is expected.
    nonempty = [a for a in artifacts if a.num_tiles > 0]
    assert nonempty, "expected at least one artifact with tiles to drive a tiling preview"

    tiling_preview_dir = output_dir / "preview" / "tiling"
    produced = sorted(tiling_preview_dir.rglob("*.jpg"))
    assert produced, (
        f"no tiling-preview JPEG was produced under {tiling_preview_dir} "
        f"(mode={output_mode})"
    )

    # Each non-empty artifact's recorded tiling preview path must point at a real JPEG.
    for artifact in nonempty:
        if artifact.tiling_preview_path is not None:
            assert artifact.tiling_preview_path.is_file()

    # Mask-preview behavior is unchanged: still rendered.
    mask_preview_dir = output_dir / "preview" / "mask"
    assert sorted(mask_preview_dir.rglob("*.jpg")), "mask preview should still be produced"
