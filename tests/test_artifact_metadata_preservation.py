"""Artifact metadata and process-row labels survive fresh, resumed, and reused tiling.

A source-mask tissue batch tiles a flat PNG slide (CPU ``pil`` reader, explicit spacing)
against a binary PNG tissue mask. Whatever path produced the run, the returned
``TilingArtifacts`` must agree with the persisted metadata (``output_mode='merged'``,
``annotation=None``) while the ``process_list.csv`` row stays labelled ``'tissue'``.
"""

import csv
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from hs2p.api import FilterConfig, PreviewConfig, SlideSpec, TilingConfig, tile_slides
from hs2p.wsi.types import (
    CoordinateOutputMode,
    CoordinateSelectionStrategy,
    SamplingSpec,
)

SPACING_UM = 0.5
TILE_PX = 64
SAMPLE_ID = "flat"


def _write_inputs(tmp_path: Path, *, slide_px: int = 256, tissue_px: int = 128):
    """A ``slide_px`` square RGB slide with tissue in its top-left ``tissue_px`` square.

    At the slide's native spacing every tile is ``TILE_PX`` pixels, so the expected origins
    are exactly the tile grid covering that tissue square.
    """
    slide_path = tmp_path / "slide.png"
    rng = np.random.default_rng(0)
    Image.fromarray(
        rng.integers(90, 200, size=(slide_px, slide_px, 3), dtype=np.uint8)
    ).save(slide_path)
    labels = np.zeros((slide_px, slide_px), dtype=np.uint8)
    labels[:tissue_px, :tissue_px] = 1
    mask_path = tmp_path / "mask.png"
    Image.fromarray(labels, mode="L").save(mask_path)
    expected_origins = sorted(
        (x, y) for x in range(0, tissue_px, TILE_PX) for y in range(0, tissue_px, TILE_PX)
    )
    slide = SlideSpec(
        sample_id=SAMPLE_ID,
        image_path=slide_path,
        mask_path=mask_path,
        spacing_at_level_0=SPACING_UM,
    )
    return slide, expected_origins


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


def _run(slide: SlideSpec, output_dir: Path, **kwargs):
    return tile_slides(
        [slide],
        tiling=_tiling(),
        filtering=FilterConfig(a_t=0, a_h=0),
        output_dir=output_dir,
        **kwargs,
    )


def _single_row(output_dir: Path) -> dict:
    rows = pd.read_csv(
        output_dir / "process_list.csv", converters={"sample_id": str}
    ).to_dict(orient="records")
    assert len(rows) == 1
    assert rows[0]["tiling_status"] == "success", rows[0].get("error")
    return rows[0]


def _assert_tissue_result(artifacts, output_dir: Path, expected_origins):
    """The returned artifact, its persisted metadata/coordinates and its row agree."""
    assert len(artifacts) == 1
    artifact = artifacts[0]
    meta = json.loads(artifact.coordinates_meta_path.read_text())["artifact"]
    assert meta["output_mode"] == CoordinateOutputMode.MERGED
    assert meta["annotation"] is None
    assert meta["selection_strategy"] == CoordinateSelectionStrategy.MERGED_DEFAULT_TILING
    assert artifact.output_mode == meta["output_mode"]
    assert artifact.annotation == meta["annotation"]

    coordinates = np.load(artifact.coordinates_npz_path)
    origins = sorted(zip(coordinates["x"].tolist(), coordinates["y"].tolist()))
    assert origins == expected_origins
    assert artifact.num_tiles == len(expected_origins)

    row = _single_row(output_dir)
    assert row["annotation"] == "tissue"
    assert row["output_mode"] == CoordinateOutputMode.MERGED
    assert row["num_tiles"] == len(expected_origins)
    assert Path(row["coordinates_meta_path"]) == artifact.coordinates_meta_path
    return artifact, row


@pytest.mark.parametrize("tiling_preview", [False, True], ids=["no-preview", "tiling-preview"])
def test_fresh_source_mask_tissue_batch_keeps_merged_metadata_and_tissue_label(
    tmp_path: Path, tiling_preview: bool
):
    slide, expected_origins = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"
    preview = PreviewConfig(save_mask_preview=False, save_tiling_preview=tiling_preview)

    artifacts = _run(slide, output_dir, preview=preview)

    artifact, row = _assert_tissue_result(artifacts, output_dir, expected_origins)
    if tiling_preview:
        expected_preview = output_dir / "preview" / "tiling" / f"{SAMPLE_ID}.jpg"
        assert artifact.tiling_preview_path == expected_preview
        assert Path(row["tiling_preview_path"]) == expected_preview


def _assert_tile_export(artifact, row: dict, output_dir: Path, expected_origins) -> None:
    """The TAR and its manifest hold exactly the expected tiles, and both the artifact
    and its row point at that TAR."""
    tiles_dir = output_dir / "tiles"
    expected_tar = tiles_dir / f"{SAMPLE_ID}.tiles.tar"
    manifest = tiles_dir / f"{SAMPLE_ID}.tiles.manifest.csv"
    assert artifact.tiles_tar_path == expected_tar
    assert Path(row["tiles_tar_path"]) == expected_tar
    with tarfile.open(expected_tar) as archive:
        assert len(archive.getnames()) == len(expected_origins)
    with manifest.open(newline="") as handle:
        manifest_origins = sorted(
            (int(record["x"]), int(record["y"])) for record in csv.DictReader(handle)
        )
    assert manifest_origins == expected_origins


def test_fresh_source_mask_tissue_batch_with_tile_export_keeps_metadata(tmp_path: Path):
    slide, expected_origins = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"

    artifacts = _run(slide, output_dir, save_tiles=True, jpeg_backend="pil")

    artifact, row = _assert_tissue_result(artifacts, output_dir, expected_origins)
    _assert_tile_export(artifact, row, output_dir, expected_origins)


@pytest.mark.parametrize("save_tiles", [False, True], ids=["coordinates", "tile-export"])
def test_resumed_source_mask_tissue_batch_matches_the_fresh_run(
    tmp_path: Path, save_tiles: bool
):
    slide, expected_origins = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"
    export = {"save_tiles": True, "jpeg_backend": "pil"} if save_tiles else {}
    fresh = _run(slide, output_dir, **export)

    resumed = _run(slide, output_dir, resume=True, **export)

    assert resumed == fresh
    artifact, row = _assert_tissue_result(resumed, output_dir, expected_origins)
    if save_tiles:
        _assert_tile_export(artifact, row, output_dir, expected_origins)


@pytest.mark.parametrize(
    "outputs",
    [
        {},
        {"save_tiles": True, "jpeg_backend": "pil"},
        {"preview": PreviewConfig(save_mask_preview=False, save_tiling_preview=True)},
    ],
    ids=["coordinates", "tile-export", "tiling-preview"],
)
def test_reused_coordinates_keep_the_fresh_run_metadata_and_tissue_label(
    tmp_path: Path, outputs: dict
):
    slide, expected_origins = _write_inputs(tmp_path)
    fresh = _run(slide, tmp_path / "first")
    output_dir = tmp_path / "second"

    reused = _run(
        slide,
        output_dir,
        read_coordinates_from=tmp_path / "first" / "tiles",
        **outputs,
    )

    assert len(reused) == 1
    assert replace(reused[0], tiles_tar_path=None, tiling_preview_path=None) == fresh[0]
    artifact, row = _assert_tissue_result(reused, output_dir, expected_origins)
    if outputs.get("save_tiles"):
        _assert_tile_export(artifact, row, output_dir, expected_origins)
    if "preview" in outputs:
        expected_preview = output_dir / "preview" / "tiling" / f"{SAMPLE_ID}.jpg"
        assert expected_preview.is_file()
        assert artifact.tiling_preview_path == expected_preview
        assert Path(row["tiling_preview_path"]) == expected_preview


@pytest.mark.parametrize(
    ("output_mode", "expected_label"),
    [
        (CoordinateOutputMode.MERGED, "merged"),
        (CoordinateOutputMode.PER_ANNOTATION, "tumor"),
    ],
)
def test_joint_sampling_rows_keep_their_labels_through_preview_finalization(
    tmp_path: Path, output_mode: str, expected_label: str
):
    slide, expected_origins = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"

    artifacts = _run(
        slide,
        output_dir,
        preview=PreviewConfig(save_mask_preview=False, save_tiling_preview=True),
        sampling=SamplingSpec(
            pixel_mapping={"background": 0, "tumor": 1},
            color_mapping={"background": None, "tumor": [255, 0, 0]},
            tissue_percentage={"background": None, "tumor": 0.5},
            active_annotations=("tumor",),
        ),
        selection_strategy=CoordinateSelectionStrategy.JOINT_SAMPLING,
        output_mode=output_mode,
    )

    assert len(artifacts) == 1
    meta = json.loads(artifacts[0].coordinates_meta_path.read_text())["artifact"]
    assert artifacts[0].output_mode == meta["output_mode"] == output_mode
    assert artifacts[0].annotation == meta["annotation"]
    coordinates = np.load(artifacts[0].coordinates_npz_path)
    assert sorted(zip(coordinates["x"].tolist(), coordinates["y"].tolist())) == expected_origins
    row = _single_row(output_dir)
    assert row["annotation"] == expected_label
    assert row["output_mode"] == output_mode
    assert Path(row["tiling_preview_path"]).is_file()
