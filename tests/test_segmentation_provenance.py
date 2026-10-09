"""Segmentation thresholds are recorded only when segmentation ran, and resume honours that.

A flat 256 px PNG slide at 0.5 um/px is tiled with 64 px tiles at its native spacing on the
CPU ``pil`` reader. Tissue tiling either segments the slide (``threshold`` method) or reads a
binary PNG tissue mask covering its top-left 128 px square; annotation sampling reads a
tumor/stroma label mask. A precomputed mask or an annotation mask is never segmented, so no
segmentation threshold shaped its tiles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from hs2p.api import (
    BatchPartialFailureWarning,
    FilterConfig,
    SegmentationConfig,
    SlideSpec,
    TilingConfig,
    load_tiling_result,
    tile_slides,
)
from hs2p.wsi.types import (
    CoordinateOutputMode,
    CoordinateSelectionStrategy,
    SamplingSpec,
)

SPACING_UM = 0.5
TILE_PX = 64
SAMPLE_ID = "slide"
THRESHOLD_KEYS = ("sthresh", "sthresh_up", "mthresh", "close")
MASK_ORIGINS = [(0, 0), (0, 64), (64, 0), (64, 64)]


def _write_slide(tmp_path: Path) -> Path:
    """White background with one saturated 128 px block at the origin, so threshold
    segmentation finds exactly the block the tissue mask declares."""
    pixels = np.full((256, 256, 3), 255, dtype=np.uint8)
    pixels[0:128, 0:128] = (200, 60, 120)
    path = tmp_path / "slide.png"
    Image.fromarray(pixels).save(path)
    return path


def _write_tissue_mask(tmp_path: Path) -> Path:
    labels = np.zeros((256, 256), dtype=np.uint8)
    labels[0:128, 0:128] = 1
    path = tmp_path / "tissue.png"
    Image.fromarray(labels, mode="L").save(path)
    return path


def _slide_spec(image_path: Path, mask_path: Path | None) -> SlideSpec:
    return SlideSpec(
        sample_id=SAMPLE_ID,
        image_path=image_path,
        mask_path=mask_path,
        spacing_at_level_0=SPACING_UM,
    )


def _tiling() -> TilingConfig:
    return TilingConfig(
        requested_spacing_um=SPACING_UM,
        requested_tile_size_px=TILE_PX,
        tolerance=0.05,
        overlap=0.0,
        min_coverage={"tissue": 0.5},
        backend="pil",
        mask_backend="pil",
    )


def _segmentation(*, sthresh: int) -> SegmentationConfig:
    return SegmentationConfig(method="threshold", downsample=1, sthresh=sthresh)


def _tile(slide: SlideSpec, output_dir: Path, *, sthresh: int, resume: bool = False):
    return tile_slides(
        [slide],
        tiling=_tiling(),
        segmentation=_segmentation(sthresh=sthresh),
        filtering=FilterConfig(a_t=0, a_h=0),
        output_dir=output_dir,
        resume=resume,
    )


def _process_row(output_dir: Path) -> dict:
    rows = pd.read_csv(
        output_dir / "process_list.csv", converters={"sample_id": str}
    ).to_dict(orient="records")
    assert len(rows) == 1
    return rows[0]


def _thresholds(meta_path: Path) -> dict:
    meta = json.loads(Path(meta_path).read_text())
    return {key: meta["segmentation"][key] for key in THRESHOLD_KEYS}


def _origins(npz_path: Path) -> list[tuple[int, int]]:
    coordinates = np.load(npz_path)
    return sorted(zip(coordinates["x"].tolist(), coordinates["y"].tolist()))


# ------------------------------------------------------------------- resume


def test_resuming_a_precomputed_mask_run_after_changing_sthresh_reuses_its_coordinates(
    tmp_path,
):
    slide = _slide_spec(_write_slide(tmp_path), _write_tissue_mask(tmp_path))
    output_dir = tmp_path / "out"
    fresh = _tile(slide, output_dir, sthresh=8)
    meta_bytes = fresh[0].coordinates_meta_path.read_bytes()

    resumed = _tile(slide, output_dir, sthresh=15, resume=True)

    row = _process_row(output_dir)
    assert row["tiling_status"] == "success", row.get("error")
    assert resumed == fresh
    assert _origins(resumed[0].coordinates_npz_path) == MASK_ORIGINS
    # Reused, not recomputed: the sidecar is the fresh run's, byte for byte.
    assert resumed[0].coordinates_meta_path.read_bytes() == meta_bytes
    assert _thresholds(resumed[0].coordinates_meta_path) == dict.fromkeys(THRESHOLD_KEYS)


def test_resuming_a_segmentation_run_after_changing_sthresh_still_rejects_the_artifact(
    tmp_path,
):
    slide = _slide_spec(_write_slide(tmp_path), None)
    output_dir = tmp_path / "out"
    fresh = _tile(slide, output_dir, sthresh=8)
    assert _thresholds(fresh[0].coordinates_meta_path) == {
        "sthresh": 8,
        "sthresh_up": 255,
        "mthresh": 7,
        "close": 4,
    }
    assert _origins(fresh[0].coordinates_npz_path) == MASK_ORIGINS

    with pytest.warns(BatchPartialFailureWarning, match="sthresh mismatch"):
        _tile(slide, output_dir, sthresh=15, resume=True)

    row = _process_row(output_dir)
    assert row["tiling_status"] == "failed"
    assert row["error"] == "precomputed tiles sthresh mismatch"


def _rewrite_as_earlier_version(meta_path: Path, thresholds: dict) -> None:
    """Rewrite a sidecar's thresholds as an earlier hs2p version stored them: ints, even
    for a precomputed mask that never segmented."""
    meta = json.loads(meta_path.read_text())
    meta["segmentation"].update(thresholds)
    meta_path.write_text(json.dumps(meta, indent=2))


def test_an_earlier_version_sidecar_holding_int_thresholds_still_loads_and_resumes(
    tmp_path,
):
    slide = _slide_spec(_write_slide(tmp_path), _write_tissue_mask(tmp_path))
    output_dir = tmp_path / "out"
    fresh = _tile(slide, output_dir, sthresh=8)
    legacy = {"sthresh": 8, "sthresh_up": 255, "mthresh": 7, "close": 4}
    _rewrite_as_earlier_version(fresh[0].coordinates_meta_path, legacy)

    loaded = load_tiling_result(
        coordinates_npz_path=fresh[0].coordinates_npz_path,
        coordinates_meta_path=fresh[0].coordinates_meta_path,
    )
    assert (
        loaded.seg_sthresh,
        loaded.seg_sthresh_up,
        loaded.seg_mthresh,
        loaded.seg_close,
    ) == (8, 255, 7, 4)

    resumed = _tile(slide, output_dir, sthresh=8, resume=True)

    assert _process_row(output_dir)["tiling_status"] == "success"
    assert resumed == fresh
    assert _thresholds(resumed[0].coordinates_meta_path) == legacy


# -------------------------------------------------------- annotation sampling


def _write_annotation_mask(tmp_path: Path) -> Path:
    labels = np.zeros((256, 256), dtype=np.uint8)
    labels[0:128, 0:128] = 1
    labels[128:192, 0:128] = 2
    path = tmp_path / "annotations.png"
    Image.fromarray(labels, mode="L").save(path)
    return path


def _sampling() -> SamplingSpec:
    return SamplingSpec(
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2},
        color_mapping=None,
        tissue_percentage={"background": None, "tumor": 0.5, "stroma": 0.5},
        active_annotations=("tumor", "stroma"),
    )


@pytest.mark.parametrize(
    "output_mode",
    [CoordinateOutputMode.PER_ANNOTATION, CoordinateOutputMode.MERGED],
)
@pytest.mark.parametrize(
    "strategy",
    [
        CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
        CoordinateSelectionStrategy.JOINT_SAMPLING,
    ],
)
def test_annotation_sampling_metadata_records_null_thresholds_whatever_seg_params_say(
    tmp_path, strategy, output_mode
):
    artifacts = tile_slides(
        [_slide_spec(_write_slide(tmp_path), _write_annotation_mask(tmp_path))],
        tiling=_tiling(),
        # Deliberately non-default thresholds: none of them may leak into the metadata.
        segmentation=SegmentationConfig(
            method="hsv", sthresh=15, sthresh_up=250, mthresh=5, close=2
        ),
        filtering=FilterConfig(a_t=0, a_h=0),
        output_dir=tmp_path / "out",
        sampling=_sampling(),
        selection_strategy=strategy,
        output_mode=output_mode,
    )

    assert artifacts
    for artifact in artifacts:
        assert _thresholds(artifact.coordinates_meta_path) == dict.fromkeys(
            THRESHOLD_KEYS
        )
