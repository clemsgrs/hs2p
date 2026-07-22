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


def test_merged_row_labeled_merged_not_tissue():
    """A merged MERGED artifact has annotation=None like binary tissue, but must not be
    recorded as 'tissue' — it carries output_mode=merged and the label 'merged'."""
    from hs2p.wsi.types import CoordinateOutputMode

    artifact = TilingArtifacts(
        sample_id="slide-1",
        coordinates_npz_path=Path("tiles/slide-1.coordinates.npz"),
        coordinates_meta_path=Path("tiles/slide-1.coordinates.meta.json"),
        num_tiles=5,
        annotation=None,
        output_mode=CoordinateOutputMode.MERGED,
    )
    row = orchestration_mod._build_success_process_row(
        whole_slide=_whole_slide(), artifact=artifact
    )
    assert row["annotation"] == "merged"
    assert row["output_mode"] == CoordinateOutputMode.MERGED


def test_binary_tissue_row_has_no_output_mode():
    row = orchestration_mod._build_success_process_row(
        whole_slide=_whole_slide(), artifact=_artifact(annotation=None)
    )
    assert row["annotation"] == "tissue"
    assert row["output_mode"] is None


def test_success_row_has_all_required_columns():
    row = orchestration_mod._build_success_process_row(
        whole_slide=_whole_slide(mask_path=Path("mask.png")),
        artifact=_artifact(annotation="tissue"),
    )
    expected_columns = {
        "sample_id", "annotation", "output_mode", "image_path", "mask_path",
        "requested_backend", "backend", "requested_mask_backend", "mask_backend",
        "tiling_status", "num_tiles",
        "coordinates_npz_path", "coordinates_meta_path", "tiles_tar_path",
        "mask_preview_path", "tiling_preview_path",
        "error", "traceback",
    }
    assert set(row.keys()) == expected_columns


def test_failure_row_has_all_required_columns():
    row = orchestration_mod._build_failure_process_row(
        whole_slide=_whole_slide(mask_path=Path("mask.png")),
        error="test error",
        traceback_text="traceback",
    )
    expected_columns = {
        "sample_id", "annotation", "output_mode", "image_path", "mask_path",
        "requested_backend", "backend", "requested_mask_backend", "mask_backend",
        "tiling_status", "num_tiles",
        "coordinates_npz_path", "coordinates_meta_path", "tiles_tar_path",
        "mask_preview_path", "tiling_preview_path",
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


# --- resume metadata merge ignores requested_backend (Finding 5) -------------------------


def _resume_row(**overrides):
    row = dict(
        sample_id="slide-1",
        tiling_status="success",
        num_tiles=5,
        coordinates_npz_path="tiles/slide-1.npz",
        coordinates_meta_path="tiles/slide-1.meta.json",
        tiles_tar_path=None,
        requested_backend="auto",
        backend="openslide",
        mask_backend=None,
        tiling_preview_path=None,
    )
    row.update(overrides)
    return row


def test_resume_merge_carries_external_columns_when_only_requested_backend_differs(tmp_path):
    """When only requested_backend differs but the resolved backend matches, the existing row's
    external columns and preview path must carry forward into the new row."""
    preview_file = tmp_path / "slide-1.tiling.jpg"
    preview_file.write_bytes(b"x")
    row = _resume_row(requested_backend="auto", tiling_preview_path=None)
    existing_row = _resume_row(
        requested_backend="openslide",  # differs from this run, same resolved backend
        tiling_preview_path=str(preview_file),
        feature_status="done",
        feature_path="features/slide-1.pt",
    )
    merged = orchestration_mod._merge_existing_resume_metadata(row, existing_row)
    assert merged["feature_status"] == "done"
    assert merged["feature_path"] == "features/slide-1.pt"
    assert Path(merged["tiling_preview_path"]) == preview_file


def test_resume_merge_skipped_when_resolved_backend_differs(tmp_path):
    """The resolved backend guard still holds: a genuinely different backend skips the merge."""
    preview_file = tmp_path / "slide-1.tiling.jpg"
    preview_file.write_bytes(b"x")
    row = _resume_row(requested_backend="auto", backend="openslide", tiling_preview_path=None)
    existing_row = _resume_row(
        requested_backend="cucim",
        backend="cucim",  # resolved backend genuinely differs
        tiling_preview_path=str(preview_file),
        feature_status="done",
        feature_path="features/slide-1.pt",
    )
    merged = orchestration_mod._merge_existing_resume_metadata(row, existing_row)
    assert "feature_status" not in merged
    assert "feature_path" not in merged
    assert merged["tiling_preview_path"] is None
