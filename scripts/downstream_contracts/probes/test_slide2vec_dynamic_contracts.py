"""hs2p-owned probes of slide2vec contracts that import resolution cannot prove.

Run by scripts/downstream_contracts/run.py inside slide2vec's pytest process at the pinned
revision. They call slide2vec's own code against the candidate hs2p; nothing here
re-implements slide2vec. Grouped/supertile exact-pixel reads, annotation masks and
persisted artifacts are covered by slide2vec's tests/test_hs2p5_integration.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hs2p import SlideSpec
from PIL import Image

from slide2vec import progress as slide2vec_progress
from slide2vec.api import PreprocessingConfig
from slide2vec.runtime.hierarchical import resolve_hierarchical_geometry
from slide2vec.runtime.image_specs import spacing_readable_image_suffixes
from slide2vec.runtime.tiling_pipeline import prepare_tiled_slides


class _RecordingReporter:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None

    def write_log(self, message: str, *, stream=None) -> None:
        return None


def _flat_slide(tmp_path: Path) -> SlideSpec:
    image_path = tmp_path / "slide.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (128, 128), (160, 70, 110)).save(image_path)
    Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(mask_path)
    return SlideSpec(sample_id="slide", image_path=image_path, mask_path=mask_path, spacing_at_level_0=0.5)


def _preprocessing(**kwargs) -> PreprocessingConfig:
    params = dict(
        backend="auto",
        requested_spacing_um=0.5,
        requested_tile_size_px=32,
        segmentation={"downsample": 1},
        filtering={"a_t": 0, "a_h": 0},
        preview={"save_mask_preview": False, "save_tiling_preview": False},
    )
    params.update(kwargs)
    return PreprocessingConfig(**params)


def test_hs2p_tiling_progress_is_activated_and_forwarded_to_the_slide2vec_reporter(tmp_path):
    """progress_bridge: hs2p.progress.activate_progress_reporter + hs2p's event kinds/payloads."""
    reporter = _RecordingReporter()
    with slide2vec_progress.activate_progress_reporter(reporter):
        prepare_tiled_slides([_flat_slide(tmp_path)], _preprocessing(), output_dir=tmp_path / "out", num_workers=1)

    by_kind = {event.kind: event for event in reporter.events}
    # Only hs2p emits these two kinds; slide2vec merely forwards them.
    assert {"tissue.finished", "tiling.finished"} <= set(by_kind), sorted(by_kind)
    assert isinstance(by_kind["tiling.finished"], slide2vec_progress.ProgressEvent)
    finished = by_kind["tiling.finished"].payload
    assert (finished["completed"], finished["failed"], finished["discovered_tiles"]) == (1, 0, 16)


def test_slide2vec_discovers_the_path_suffixes_hs2p_backends_declare():
    """image_specs walks hs2p.wsi.backends for *_SUPPORTED_SUFFIXES declarations."""
    spacing_readable_image_suffixes.cache_clear()

    suffixes = spacing_readable_image_suffixes()

    assert {".svs", ".tif", ".tiff", ".png", ".jpg", ".jpeg"} <= suffixes


def test_hierarchical_geometry_comes_from_hs2p_spacing_read_plans(tmp_path):
    """hierarchical: plan_spacing_read(...).read_size_px and tile_size_lv0_from_plan."""
    _, (result,), _ = prepare_tiled_slides(
        [_flat_slide(tmp_path)], _preprocessing(), output_dir=tmp_path / "out", num_workers=1
    )
    regions = _preprocessing(requested_spacing_um=1.0, requested_region_size_px=64, region_tile_multiple=2)

    geometry = resolve_hierarchical_geometry(regions, result)

    # 1.0 um/px from a 0.5 um/px level 0: every 32 px tile reads 64 level-0 pixels.
    assert geometry == {
        "region_tile_multiple": 2,
        "tiles_per_region": 4,
        "requested_tile_size_px": 32,
        "read_tile_size_px": 64,
        "requested_region_size_px": 64,
        "read_region_size_px": 128,
        "tile_size_lv0": 64,
    }
