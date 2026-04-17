"""Tests for the unified process-list row schema with annotation column."""
import numpy as np
import pytest

from hs2p.api import TilingArtifacts, SlideSpec
from pathlib import Path

import hs2p.api as api_mod
import hs2p.tiling.orchestration as orchestration_mod


def _whole_slide(*, mask_path=None):
    return SlideSpec(
        sample_id="slide-1",
        image_path=Path("slide-1.svs"),
        mask_path=mask_path,
    )


def _artifact(*, annotation=None, tiles_tar_path=None):
    return TilingArtifacts(
        sample_id="slide-1",
        coordinates_npz_path=Path("tiles/slide-1.coordinates.npz"),
        coordinates_meta_path=Path("tiles/slide-1.coordinates.meta.json"),
        num_tiles=5,
        annotation=annotation,
        tiles_tar_path=tiles_tar_path,
    )


def test_success_row_annotation_defaults_to_tissue_when_none():
    row = orchestration_mod._build_success_process_row(
        whole_slide=_whole_slide(),
        artifact=_artifact(annotation=None),
    )
    assert row["annotation"] == "tissue"


def test_success_row_annotation_preserved_when_set():
    row = orchestration_mod._build_success_process_row(
        whole_slide=_whole_slide(),
        artifact=_artifact(annotation="tumor"),
    )
    assert row["annotation"] == "tumor"


def test_success_row_has_all_required_columns():
    row = orchestration_mod._build_success_process_row(
        whole_slide=_whole_slide(mask_path=Path("mask.png")),
        artifact=_artifact(annotation="tissue"),
    )
    expected_columns = {
        "sample_id", "annotation", "image_path", "mask_path",
        "requested_backend", "backend", "tiling_status", "num_tiles",
        "coordinates_npz_path", "coordinates_meta_path", "tiles_tar_path",
        "error", "traceback",
    }
    assert set(row.keys()) == expected_columns


def test_failure_row_annotation_defaults_to_tissue():
    row = orchestration_mod._build_failure_process_row(
        whole_slide=_whole_slide(),
        error="test error",
        traceback_text="traceback",
    )
    assert row["annotation"] == "tissue"


def test_failure_row_annotation_preserved_when_set():
    row = orchestration_mod._build_failure_process_row(
        whole_slide=_whole_slide(),
        error="test error",
        traceback_text="traceback",
        annotation="stroma",
    )
    assert row["annotation"] == "stroma"


def test_success_row_tiling_status_is_success():
    row = orchestration_mod._build_success_process_row(
        whole_slide=_whole_slide(),
        artifact=_artifact(),
    )
    assert row["tiling_status"] == "success"
    assert row["num_tiles"] == 5


def test_failure_row_tiling_status_is_failed():
    row = orchestration_mod._build_failure_process_row(
        whole_slide=_whole_slide(),
        error="oops",
        traceback_text="tb",
    )
    assert row["tiling_status"] == "failed"
    assert row["num_tiles"] == 0
