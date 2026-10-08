"""hs2p-owned probes of soma contracts that import resolution cannot prove.

Run by scripts/downstream_contracts/run.py inside soma's pytest process at the pinned
revision. They call soma's own code against the candidate hs2p; nothing here
re-implements soma. The historical tiling cache, supplied-coordinate validation and
artifact provenance are covered by soma's selected contract suites.
"""

from __future__ import annotations

from pathlib import Path

import hs2p
import numpy as np
import openslide
from hs2p import SlideSpec, TilingConfig, tile_slides
from PIL import Image
from slide2vec import progress as slide2vec_progress

from soma.dense.reader import read_image_at_spacing, read_image_region_at_spacing
from soma.extraction.reporters import _forward_tiling_progress_ctx

# The candidate checkout's own pyramidal fixture (hs2p is installed editable from it).
FIXTURE_WSI = Path(hs2p.__file__).resolve().parents[1] / "tests" / "fixtures" / "input" / "test-wsi.tif"


class _RecordingReporter:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None

    def write_log(self, message: str, *, stream=None) -> None:
        return None


def test_hs2p_tiling_progress_is_forwarded_into_the_active_slide2vec_reporter(tmp_path):
    """extraction.reporters: hs2p.progress.activate_progress_reporter + tiling.progress events."""
    image_path = tmp_path / "slide.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (128, 128), (160, 70, 110)).save(image_path)
    Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(mask_path)
    slide = SlideSpec(sample_id="slide", image_path=image_path, mask_path=mask_path, spacing_at_level_0=0.5)
    reporter = _RecordingReporter()

    with slide2vec_progress.activate_progress_reporter(reporter), _forward_tiling_progress_ctx():
        tile_slides(
            [slide],
            tiling=TilingConfig(
                backend="auto",
                requested_spacing_um=0.5,
                requested_tile_size_px=32,
                tolerance=0.05,
                overlap=0.0,
                min_coverage={"tissue": 0.0},
            ),
            output_dir=tmp_path / "out",
        )

    forwarded = [event for event in reporter.events if event.kind == "tiling.progress"]
    assert forwarded, [event.kind for event in reporter.events]
    last = forwarded[-1].payload
    assert {"total", "completed", "failed", "pending", "discovered_tiles"} <= set(last)
    assert (last["completed"], last["failed"], last["discovered_tiles"]) == (1, 0, 16)


def test_dense_reads_at_the_native_spacing_match_the_slide_pixels_exactly():
    """dense.reader: WSI.read_region_at_spacing / read_full_at_spacing keyword contract."""
    with openslide.OpenSlide(str(FIXTURE_WSI)) as slide:
        spacing = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        expected_region = np.asarray(slide.read_region((4096, 4096), 0, (64, 48)).convert("RGB"))
        expected_full = np.asarray(slide.read_region((0, 0), 3, slide.level_dimensions[3]).convert("RGB"))
        level3_spacing = spacing * slide.level_downsamples[3]

    region = read_image_region_at_spacing(
        FIXTURE_WSI, location=(4096, 4096), size=(64, 48), spacing_um=spacing, backend="openslide"
    )
    full = read_image_at_spacing(FIXTURE_WSI, spacing_um=level3_spacing, backend="openslide")

    np.testing.assert_array_equal(region, expected_region)
    np.testing.assert_array_equal(full, expected_full)
