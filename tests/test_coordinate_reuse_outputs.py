import csv
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import hs2p
import hs2p.preprocessing as preprocessing
import hs2p.tiling.orchestration as orchestration
import hs2p.wsi.streaming.stream as stream
import hs2p.wsi.wsi as wsi
from fake_wsi_backend import FakeReaderFactory, make_mask_spec, make_slide_spec
from hs2p.api import (
    FilterConfig,
    PreviewConfig,
    SegmentationConfig,
    SlideSpec,
    TilingConfig,
    save_tiling_result,
    tile_slides,
)


def _saved_coordinates(
    tmp_path: Path,
    *,
    coords: list[tuple[int, int]],
) -> tuple[SlideSpec, Path]:
    slide = SlideSpec(sample_id="reuse-slide", image_path=Path("reuse-slide.svs"))
    coordinates = np.asarray(coords, dtype=np.int64).reshape((-1, 2))
    result = preprocessing.TilingResult(
        tiles=preprocessing.TileGeometry(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            tissue_fractions=np.full(len(coordinates), 1.0, dtype=np.float32),
            tile_index=np.arange(len(coordinates), dtype=np.int32),
            requested_tile_size_px=16,
            requested_spacing_um=1.0,
            read_level=0,
            read_tile_size_px=16,
            read_spacing_um=1.0,
            tile_size_lv0=16,
            is_within_tolerance=True,
            base_spacing_um=1.0,
            slide_dimensions=[32, 32],
            level_downsamples=[1.0, 2.0],
            overlap=0.0,
            min_tissue_fraction=0.5,
        ),
        sample_id=slide.sample_id,
        image_path=slide.image_path,
        backend="asap",
        requested_backend="asap",
        tolerance=0.05,
        step_px_lv0=16,
        tissue_method="hsv",
        requested_seg_downsample=64,
        seg_downsample=64,
        seg_level=0,
        seg_spacing_um=1.0,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=16,
        a_t=4,
        a_h=0,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )
    artifact = save_tiling_result(result, output_dir=tmp_path / "saved")
    return slide, artifact.coordinates_meta_path.parent


def _tiling_config() -> TilingConfig:
    return TilingConfig(
        requested_spacing_um=1.0,
        requested_tile_size_px=16,
        tolerance=0.05,
        overlap=0.0,
        min_coverage={"tissue": 0.5},
        backend="asap",
    )


def _segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(
        method="hsv",
        downsample=64,
        sthresh=8,
        sthresh_up=255,
        mthresh=7,
        close=4,
    )


def _filter_config() -> FilterConfig:
    return FilterConfig(
        ref_tile_size=16,
        a_t=4,
        a_h=0,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def _install_fake_slide_reader(monkeypatch) -> None:
    factory = FakeReaderFactory(
        slide_spec=make_slide_spec(),
        mask_spec=make_mask_spec(np.zeros((32, 32, 1), dtype=np.uint8)),
    )
    monkeypatch.setattr(stream, "open_slide", factory)
    monkeypatch.setattr(wsi, "open_slide", factory)


def test_reused_coordinates_materialize_requested_tar_manifest_and_preview(
    monkeypatch,
    tmp_path: Path,
):
    slide, coordinates_dir = _saved_coordinates(
        tmp_path,
        coords=[(0, 0), (16, 0)],
    )
    _install_fake_slide_reader(monkeypatch)
    monkeypatch.setattr(
        orchestration,
        "_compute_tiling_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("coordinate computation must be skipped")
        ),
    )

    output_dir = tmp_path / "run"
    artifacts = tile_slides(
        [slide],
        tiling=_tiling_config(),
        segmentation=_segmentation_config(),
        filtering=_filter_config(),
        preview=PreviewConfig(save_tiling_preview=True, downsample=2),
        output_dir=output_dir,
        read_coordinates_from=coordinates_dir,
        save_tiles=True,
        jpeg_backend="pil",
    )

    artifact = artifacts[0]
    expected_tar = output_dir / "tiles" / "reuse-slide.tiles.tar"
    expected_manifest = output_dir / "tiles" / "reuse-slide.tiles.manifest.csv"
    expected_preview = output_dir / "preview" / "tiling" / "reuse-slide.jpg"
    assert artifact.coordinates_npz_path == coordinates_dir / "reuse-slide.coordinates.npz"
    assert artifact.coordinates_meta_path == coordinates_dir / "reuse-slide.coordinates.meta.json"
    assert artifact.tiles_tar_path == expected_tar
    assert artifact.tiling_preview_path == expected_preview

    with tarfile.open(expected_tar) as archive:
        assert archive.getnames() == ["000000.jpg", "000001.jpg"]
    with expected_manifest.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"tile_index": "0", "x": "0", "y": "0"},
            {"tile_index": "1", "x": "16", "y": "0"},
        ]
    assert Image.open(expected_preview).size == (16, 16)

    row = pd.read_csv(output_dir / "process_list.csv").iloc[0]
    assert Path(row["coordinates_npz_path"]) == artifact.coordinates_npz_path
    assert Path(row["coordinates_meta_path"]) == artifact.coordinates_meta_path
    assert Path(row["tiles_tar_path"]) == expected_tar
    assert Path(row["tiling_preview_path"]) == expected_preview


def test_reused_zero_tile_coordinates_do_not_fabricate_a_preview(
    monkeypatch,
    tmp_path: Path,
):
    slide, coordinates_dir = _saved_coordinates(tmp_path, coords=[])
    monkeypatch.setattr(
        orchestration,
        "_compute_tiling_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("coordinate computation must be skipped")
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "write_coordinate_preview",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("zero-tile preview rendering must be skipped")
        ),
    )

    output_dir = tmp_path / "run"
    artifacts = tile_slides(
        [slide],
        tiling=_tiling_config(),
        segmentation=_segmentation_config(),
        filtering=_filter_config(),
        preview=PreviewConfig(save_tiling_preview=True, downsample=2),
        output_dir=output_dir,
        read_coordinates_from=coordinates_dir,
    )

    assert artifacts[0].num_tiles == 0
    assert artifacts[0].coordinates_npz_path is None
    assert artifacts[0].tiling_preview_path is None
    assert not (output_dir / "preview" / "tiling" / "reuse-slide.jpg").exists()
    row = pd.read_csv(output_dir / "process_list.csv").iloc[0]
    assert pd.isna(row["coordinates_npz_path"])
    assert pd.isna(row["tiling_preview_path"])


def test_reused_coordinates_do_not_materialize_disabled_outputs(
    monkeypatch,
    tmp_path: Path,
):
    slide, coordinates_dir = _saved_coordinates(tmp_path, coords=[(0, 0)])
    monkeypatch.setattr(
        orchestration,
        "_compute_tiling_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("coordinate computation must be skipped")
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "extract_tiles_to_tar",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("disabled tile saving must be skipped")
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "write_coordinate_preview",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("disabled preview rendering must be skipped")
        ),
    )

    output_dir = tmp_path / "run"
    artifacts = tile_slides(
        [slide],
        tiling=_tiling_config(),
        segmentation=_segmentation_config(),
        filtering=_filter_config(),
        output_dir=output_dir,
        read_coordinates_from=coordinates_dir,
    )

    assert artifacts[0].tiles_tar_path is None
    assert artifacts[0].tiling_preview_path is None
    assert not (output_dir / "tiles" / "reuse-slide.tiles.tar").exists()
    assert not (output_dir / "tiles" / "reuse-slide.tiles.manifest.csv").exists()
    assert not (output_dir / "preview" / "tiling" / "reuse-slide.jpg").exists()
    row = pd.read_csv(output_dir / "process_list.csv").iloc[0]
    assert pd.isna(row["tiles_tar_path"])
    assert pd.isna(row["tiling_preview_path"])


def test_reused_output_failure_preserves_batch_failure_contract(
    monkeypatch,
    tmp_path: Path,
):
    slide, coordinates_dir = _saved_coordinates(tmp_path, coords=[(0, 0)])
    monkeypatch.setattr(
        orchestration,
        "extract_tiles_to_tar",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()),
    )

    output_dir = tmp_path / "run"
    with pytest.warns(
        hs2p.BatchPartialFailureWarning,
        match="reuse-slide: RuntimeError",
    ):
        artifacts = tile_slides(
            [slide],
            tiling=_tiling_config(),
            segmentation=_segmentation_config(),
            filtering=_filter_config(),
            output_dir=output_dir,
            read_coordinates_from=coordinates_dir,
            save_tiles=True,
            jpeg_backend="pil",
        )

    assert artifacts == []
    row = pd.read_csv(output_dir / "process_list.csv").iloc[0]
    assert row["tiling_status"] == "failed"
    assert row["error"] == "RuntimeError"
    assert "RuntimeError" in row["traceback"]
