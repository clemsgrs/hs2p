from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hs2p.api import (
    FilterConfig,
    QCConfig,
    SegmentationConfig,
    TilingArtifacts,
    TilingConfig,
    TilingResult,
    WholeSlide,
    compute_config_hash,
    load_tiling_result,
    load_whole_slides_from_rows,
    save_tiling_result,
    tile_slide,
    tile_slides,
    validate_tiling_artifacts,
)
from hs2p.utils import load_csv
from hs2p.wsi import CoordinateExtractionResult


@pytest.fixture
def tiling_config() -> TilingConfig:
    return TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.1,
        tissue_threshold=0.2,
        drop_holes=False,
        use_padding=True,
        backend="asap",
    )


@pytest.fixture
def segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(
        downsample=64,
        sthresh=8,
        sthresh_up=255,
        mthresh=7,
        close=4,
        use_otsu=False,
        use_hsv=True,
    )


@pytest.fixture
def filter_config() -> FilterConfig:
    return FilterConfig(
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


def _fake_extraction() -> CoordinateExtractionResult:
    return CoordinateExtractionResult(
        coordinates=[(100, 200), (300, 400)],
        contour_indices=[0, 0],
        tissue_percentages=[0.25, 0.75],
        x_lv0=np.array([100, 300], dtype=np.int64),
        y_lv0=np.array([200, 400], dtype=np.int64),
        read_level=1,
        read_spacing_um=1.0,
        read_tile_size_px=448,
        resize_factor=2.0,
        tile_size_lv0=448,
    )


def _build_result(
    *,
    sample_id: str,
    image_path: str,
    mask_path: str | None = None,
    config_hash: str = "actual-hash",
) -> TilingResult:
    return TilingResult(
        sample_id=sample_id,
        image_path=Path(image_path),
        mask_path=Path(mask_path) if mask_path is not None else None,
        backend="asap",
        x_lv0=np.array([10], dtype=np.int64),
        y_lv0=np.array([20], dtype=np.int64),
        tile_index=np.array([0], dtype=np.int32),
        tissue_fraction=None,
        target_spacing_um=0.5,
        target_tile_size_px=224,
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        tile_size_lv0=224,
        overlap=0.0,
        tissue_threshold=0.1,
        num_tiles=1,
        config_hash=config_hash,
    )


def test_tile_slide_builds_default_sampling_params_for_masked_slides(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    captured = {}

    def _fake_extract_coordinates(**kwargs):
        captured["sampling_params"] = kwargs["sampling_params"]
        return _fake_extraction()

    monkeypatch.setattr("hs2p.api.extract_coordinates", _fake_extract_coordinates)

    tile_slide(
        WholeSlide(
            sample_id="slide-with-mask",
            image_path=Path("slide.svs"),
            mask_path=Path("slide-mask.png"),
        ),
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
    )

    sampling_params = captured["sampling_params"]
    assert sampling_params is not None
    assert sampling_params.pixel_mapping == {"background": 0, "tissue": 1}
    assert sampling_params.color_mapping == {"background": None, "tissue": None}
    assert sampling_params.tissue_percentage == {
        "background": None,
        "tissue": tiling_config.tissue_threshold,
    }


def test_tile_slide_returns_named_arrays(monkeypatch, tiling_config, segmentation_config, filter_config):
    monkeypatch.setattr("hs2p.api.extract_coordinates", lambda **_: _fake_extraction())

    result = tile_slide(
        WholeSlide(
            sample_id="slide-1",
            image_path=Path("slide-1.svs"),
            mask_path=Path("slide-1-mask.png"),
        ),
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        num_workers=2,
    )

    assert isinstance(result, TilingResult)
    np.testing.assert_array_equal(result.x_lv0, np.array([100, 300], dtype=np.int64))
    np.testing.assert_array_equal(result.y_lv0, np.array([200, 400], dtype=np.int64))
    np.testing.assert_array_equal(result.tile_index, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        result.tissue_fraction,
        np.array([0.25, 0.75], dtype=np.float32),
    )
    assert result.sample_id == "slide-1"
    assert result.image_path == Path("slide-1.svs")
    assert result.mask_path == Path("slide-1-mask.png")
    assert result.read_level == 1
    assert result.read_spacing_um == pytest.approx(1.0)
    assert result.read_tile_size_px == 448
    assert result.tile_size_lv0 == 448
    assert result.target_spacing_um == tiling_config.target_spacing_um
    assert result.target_tile_size_px == tiling_config.target_tile_size_px
    assert result.overlap == tiling_config.overlap
    assert result.tissue_threshold == tiling_config.tissue_threshold
    assert result.num_tiles == 2
    assert result.config_hash == compute_config_hash(
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
    )


def test_tile_slide_warns_when_preview_qc_is_requested(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    monkeypatch.setattr("hs2p.api.extract_coordinates", lambda **_: _fake_extraction())

    with pytest.warns(UserWarning, match="does not write preview artifacts"):
        tile_slide(
            WholeSlide(sample_id="slide-qc", image_path=Path("slide-qc.svs")),
            tiling=tiling_config,
            segmentation=segmentation_config,
            filtering=filter_config,
            qc=QCConfig(save_mask_preview=True, save_tiling_preview=True),
        )


def test_save_tiling_result_writes_expected_npz_and_json(tmp_path: Path):
    result = TilingResult(
        sample_id="slide-2",
        image_path=Path("slide-2.svs"),
        mask_path=None,
        backend="asap",
        x_lv0=np.array([10, 30], dtype=np.int64),
        y_lv0=np.array([20, 40], dtype=np.int64),
        tile_index=np.array([0, 1], dtype=np.int32),
        tissue_fraction=np.array([0.3, 0.7], dtype=np.float32),
        target_spacing_um=0.5,
        target_tile_size_px=224,
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        tile_size_lv0=224,
        overlap=0.0,
        tissue_threshold=0.1,
        num_tiles=2,
        config_hash="abc123",
    )

    artifacts = save_tiling_result(result, output_dir=tmp_path)

    assert artifacts == TilingArtifacts(
        sample_id="slide-2",
        tiles_npz_path=tmp_path / "coordinates" / "slide-2.tiles.npz",
        tiles_meta_path=tmp_path / "coordinates" / "slide-2.tiles.meta.json",
        num_tiles=2,
        mask_preview_path=None,
        tiling_preview_path=None,
    )

    tiles = np.load(artifacts.tiles_npz_path, allow_pickle=False)
    assert set(tiles.files) == {"tile_index", "x_lv0", "y_lv0", "tissue_fraction"}
    np.testing.assert_array_equal(tiles["tile_index"], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(tiles["x_lv0"], np.array([10, 30], dtype=np.int64))
    np.testing.assert_array_equal(tiles["y_lv0"], np.array([20, 40], dtype=np.int64))
    np.testing.assert_array_equal(
        tiles["tissue_fraction"],
        np.array([0.3, 0.7], dtype=np.float32),
    )

    meta = json.loads(artifacts.tiles_meta_path.read_text())
    assert set(meta) == {
        "backend",
        "config_hash",
        "image_path",
        "mask_path",
        "num_tiles",
        "overlap",
        "read_level",
        "read_spacing_um",
        "read_tile_size_px",
        "sample_id",
        "target_spacing_um",
        "target_tile_size_px",
        "tile_size_lv0",
        "tissue_threshold",
    }
    assert meta == {
        "sample_id": "slide-2",
        "image_path": "slide-2.svs",
        "mask_path": None,
        "backend": "asap",
        "target_spacing_um": 0.5,
        "target_tile_size_px": 224,
        "read_level": 0,
        "read_spacing_um": 0.5,
        "read_tile_size_px": 224,
        "tile_size_lv0": 224,
        "overlap": 0.0,
        "tissue_threshold": 0.1,
        "num_tiles": 2,
        "config_hash": "abc123",
    }


def test_save_and_load_tiling_result_round_trip(tmp_path: Path):
    result = TilingResult(
        sample_id="slide-roundtrip",
        image_path=Path("slide-roundtrip.svs"),
        mask_path=Path("slide-roundtrip-mask.png"),
        backend="asap",
        x_lv0=np.array([10, 30], dtype=np.int64),
        y_lv0=np.array([20, 40], dtype=np.int64),
        tile_index=np.array([0, 1], dtype=np.int32),
        tissue_fraction=np.array([0.3, 0.7], dtype=np.float32),
        target_spacing_um=0.5,
        target_tile_size_px=224,
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        tile_size_lv0=224,
        overlap=0.1,
        tissue_threshold=0.2,
        num_tiles=2,
        config_hash="roundtrip-hash",
    )

    artifacts = save_tiling_result(result, output_dir=tmp_path)
    loaded = load_tiling_result(artifacts.tiles_npz_path, artifacts.tiles_meta_path)

    assert loaded.sample_id == result.sample_id
    assert loaded.image_path == result.image_path
    assert loaded.mask_path == result.mask_path
    assert loaded.backend == result.backend
    assert loaded.read_level == result.read_level
    assert loaded.read_spacing_um == result.read_spacing_um
    assert loaded.read_tile_size_px == result.read_tile_size_px
    assert loaded.tile_size_lv0 == result.tile_size_lv0
    assert loaded.overlap == result.overlap
    assert loaded.tissue_threshold == result.tissue_threshold
    assert loaded.num_tiles == result.num_tiles
    assert loaded.config_hash == result.config_hash
    np.testing.assert_array_equal(loaded.x_lv0, result.x_lv0)
    np.testing.assert_array_equal(loaded.y_lv0, result.y_lv0)
    np.testing.assert_array_equal(loaded.tile_index, result.tile_index)
    np.testing.assert_array_equal(loaded.tissue_fraction, result.tissue_fraction)


def test_save_tiling_result_rejects_invalid_tile_index(tmp_path: Path):
    invalid = TilingResult(
        sample_id="broken-slide",
        image_path=Path("broken.svs"),
        mask_path=None,
        backend="asap",
        x_lv0=np.array([10], dtype=np.int64),
        y_lv0=np.array([20], dtype=np.int64),
        tile_index=np.array([3], dtype=np.int32),
        tissue_fraction=None,
        target_spacing_um=0.5,
        target_tile_size_px=224,
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        tile_size_lv0=224,
        overlap=0.0,
        tissue_threshold=0.1,
        num_tiles=1,
        config_hash="hash",
    )

    with pytest.raises(ValueError, match="tile_index"):
        save_tiling_result(invalid, output_dir=tmp_path)


def test_save_tiling_result_rejects_non_vector_arrays(tmp_path: Path):
    invalid = TilingResult(
        sample_id="broken-shape",
        image_path=Path("broken-shape.svs"),
        mask_path=None,
        backend="asap",
        x_lv0=np.array([[10, 11]], dtype=np.int64),
        y_lv0=np.array([20], dtype=np.int64),
        tile_index=np.array([0], dtype=np.int32),
        tissue_fraction=None,
        target_spacing_um=0.5,
        target_tile_size_px=224,
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        tile_size_lv0=224,
        overlap=0.0,
        tissue_threshold=0.1,
        num_tiles=1,
        config_hash="hash",
    )

    with pytest.raises(ValueError, match="x_lv0 must be a 1D array"):
        save_tiling_result(invalid, output_dir=tmp_path)


def test_tile_slide_rejects_tissue_fraction_shape_mismatch(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    def _bad_extraction(**kwargs):
        return CoordinateExtractionResult(
            coordinates=[(100, 200), (300, 400)],
            contour_indices=[0, 0],
            tissue_percentages=[0.25],
            x_lv0=np.array([100, 300], dtype=np.int64),
            y_lv0=np.array([200, 400], dtype=np.int64),
            read_level=1,
            read_spacing_um=1.0,
            read_tile_size_px=448,
            resize_factor=2.0,
            tile_size_lv0=448,
        )

    monkeypatch.setattr("hs2p.api.extract_coordinates", _bad_extraction)

    with pytest.raises(ValueError, match="tissue_percentages length mismatch"):
        tile_slide(
            WholeSlide(sample_id="slide-bad-tissue", image_path=Path("slide.svs")),
            tiling=tiling_config,
            segmentation=segmentation_config,
            filtering=filter_config,
        )


def test_validate_tiling_artifacts_rejects_mismatched_hash(tmp_path: Path):
    result = _build_result(sample_id="slide-3", image_path="slide-3.svs")
    artifacts = save_tiling_result(result, output_dir=tmp_path)

    with pytest.raises(ValueError, match="config_hash"):
        validate_tiling_artifacts(
            whole_slide=WholeSlide(sample_id="slide-3", image_path=Path("slide-3.svs")),
            tiles_npz_path=artifacts.tiles_npz_path,
            tiles_meta_path=artifacts.tiles_meta_path,
            expected_config_hash="different-hash",
        )


def test_validate_tiling_artifacts_rejects_mismatched_image_path(tmp_path: Path):
    result = _build_result(sample_id="slide-4", image_path="stored-slide.svs")
    artifacts = save_tiling_result(result, output_dir=tmp_path)

    with pytest.raises(ValueError, match="image_path mismatch"):
        validate_tiling_artifacts(
            whole_slide=WholeSlide(sample_id="slide-4", image_path=Path("requested-slide.svs")),
            tiles_npz_path=artifacts.tiles_npz_path,
            tiles_meta_path=artifacts.tiles_meta_path,
            expected_config_hash="actual-hash",
        )


def test_validate_tiling_artifacts_rejects_mismatched_mask_path(tmp_path: Path):
    result = _build_result(
        sample_id="slide-5",
        image_path="slide-5.svs",
        mask_path="stored-mask.png",
    )
    artifacts = save_tiling_result(result, output_dir=tmp_path)

    with pytest.raises(ValueError, match="mask_path mismatch"):
        validate_tiling_artifacts(
            whole_slide=WholeSlide(
                sample_id="slide-5",
                image_path=Path("slide-5.svs"),
                mask_path=Path("requested-mask.png"),
            ),
            tiles_npz_path=artifacts.tiles_npz_path,
            tiles_meta_path=artifacts.tiles_meta_path,
            expected_config_hash="actual-hash",
        )


def test_tile_slides_writes_process_list_and_can_reuse_precomputed_tiles(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    monkeypatch.setattr("hs2p.api.extract_coordinates", lambda **_: _fake_extraction())

    precomputed_root = tmp_path / "precomputed"
    source_result = tile_slide(
        WholeSlide(sample_id="slide-1", image_path=Path("slide-1.svs")),
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
    )
    precomputed_artifacts = save_tiling_result(source_result, output_dir=precomputed_root)

    def _unexpected_extract(**kwargs):
        raise AssertionError("tile extraction should not run when precomputed tiles are reused")

    monkeypatch.setattr("hs2p.api.extract_coordinates", _unexpected_extract)

    artifacts = tile_slides(
        [WholeSlide(sample_id="slide-1", image_path=Path("slide-1.svs"))],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        output_dir=tmp_path / "run",
        read_tiles_from=precomputed_artifacts.tiles_npz_path.parent,
        resume=False,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert isinstance(artifact, TilingArtifacts)
    assert artifact.tiles_npz_path == precomputed_artifacts.tiles_npz_path
    assert artifact.tiles_meta_path == precomputed_artifacts.tiles_meta_path
    assert artifact.num_tiles == 2

    process_df = pd.read_csv(tmp_path / "run" / "process_list.csv")
    assert list(process_df.columns) == [
        "sample_id",
        "image_path",
        "mask_path",
        "tiling_status",
        "num_tiles",
        "tiles_npz_path",
        "tiles_meta_path",
        "error",
        "traceback",
    ]
    row = process_df.to_dict(orient="records")[0]
    assert row["sample_id"] == "slide-1"
    assert row["image_path"] == "slide-1.svs"
    assert pd.isna(row["mask_path"])
    assert row["tiling_status"] == "success"
    assert row["num_tiles"] == 2
    assert row["tiles_npz_path"] == str(precomputed_artifacts.tiles_npz_path)
    assert row["tiles_meta_path"] == str(precomputed_artifacts.tiles_meta_path)
    assert pd.isna(row["error"])
    assert pd.isna(row["traceback"])


def test_tile_slides_omits_tiling_preview_path_when_no_tiles(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    monkeypatch.setattr(
        "hs2p.api.extract_coordinates",
        lambda **_: CoordinateExtractionResult(
            coordinates=[],
            contour_indices=[],
            tissue_percentages=[],
            x_lv0=np.array([], dtype=np.int64),
            y_lv0=np.array([], dtype=np.int64),
            read_level=0,
            read_spacing_um=0.5,
            read_tile_size_px=224,
            resize_factor=1.0,
            tile_size_lv0=224,
        ),
    )
    monkeypatch.setattr(
        "hs2p.api.visualize_coordinates",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("preview rendering should be skipped for zero tiles")),
    )

    artifacts = tile_slides(
        [WholeSlide(sample_id="slide-0", image_path=Path("slide-0.svs"))],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        qc=QCConfig(save_tiling_preview=True),
        output_dir=tmp_path,
    )

    assert len(artifacts) == 1
    assert artifacts[0].num_tiles == 0
    assert artifacts[0].tiling_preview_path is None


def test_tile_slides_writes_preview_paths_when_visualizations_are_saved(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    monkeypatch.setattr("hs2p.api.extract_coordinates", lambda **_: _fake_extraction())

    def _fake_visualize_coordinates(**kwargs):
        save_dir = Path(kwargs["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        sample_id = kwargs.get("sample_id", "preview")
        (save_dir / f"{sample_id}.jpg").write_bytes(b"preview")

    monkeypatch.setattr("hs2p.api.visualize_coordinates", _fake_visualize_coordinates)

    mask_dir = tmp_path / "visualization" / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    expected_mask_path = mask_dir / "slide-preview.jpg"

    def _fake_extract_with_mask_preview(**kwargs):
        mask_visu_path = kwargs["mask_visu_path"]
        if mask_visu_path is not None:
            mask_visu_path.write_bytes(b"mask-preview")
        return _fake_extraction()

    monkeypatch.setattr("hs2p.api.extract_coordinates", _fake_extract_with_mask_preview)

    artifacts = tile_slides(
        [WholeSlide(sample_id="slide-preview", image_path=Path("slide-preview.svs"))],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        qc=QCConfig(save_mask_preview=True, save_tiling_preview=True),
        output_dir=tmp_path,
    )

    assert len(artifacts) == 1
    assert artifacts[0].mask_preview_path == expected_mask_path
    assert artifacts[0].mask_preview_path.is_file()
    assert artifacts[0].tiling_preview_path == tmp_path / "visualization" / "tiling" / "slide-preview.jpg"
    assert artifacts[0].tiling_preview_path.is_file()


def test_tile_slides_resume_marks_stale_artifact_as_failed(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    result = _build_result(
        sample_id="slide-6",
        image_path="stored-slide.svs",
        config_hash=compute_config_hash(
            tiling=tiling_config,
            segmentation=segmentation_config,
            filtering=filter_config,
        ),
    )
    artifacts = save_tiling_result(result, output_dir=tmp_path / "run")
    pd.DataFrame(
        [
            {
                "sample_id": "slide-6",
                "image_path": "stored-slide.svs",
                "mask_path": np.nan,
                "tiling_status": "success",
                "num_tiles": 1,
                "tiles_npz_path": str(artifacts.tiles_npz_path),
                "tiles_meta_path": str(artifacts.tiles_meta_path),
                "error": np.nan,
                "traceback": np.nan,
            }
        ]
    ).to_csv(tmp_path / "run" / "process_list.csv", index=False)

    monkeypatch.setattr(
        "hs2p.api.extract_coordinates",
        lambda **_: (_ for _ in ()).throw(AssertionError("should not recompute stale resumed tiles")),
    )

    reused = tile_slides(
        [WholeSlide(sample_id="slide-6", image_path=Path("requested-slide.svs"))],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        output_dir=tmp_path / "run",
        resume=True,
    )

    assert reused == []
    process_df = pd.read_csv(tmp_path / "run" / "process_list.csv")
    row = process_df.to_dict(orient="records")[0]
    assert row["sample_id"] == "slide-6"
    assert row["tiling_status"] == "failed"
    assert row["num_tiles"] == 0
    assert pd.isna(row["tiles_npz_path"])
    assert pd.isna(row["tiles_meta_path"])
    assert "image_path mismatch" in row["error"]


def test_tile_slides_logs_failures_in_real_time(
    monkeypatch,
    capsys,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    def _raise_extract(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("hs2p.api.extract_coordinates", _raise_extract)

    artifacts = tile_slides(
        [WholeSlide(sample_id="slide-log", image_path=Path("slide-log.svs"))],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        output_dir=tmp_path,
    )

    assert artifacts == []
    captured = capsys.readouterr()
    assert "[tile_slides] FAILED slide-log: boom" in captured.out


def test_tile_slides_resume_rejects_unsupported_process_list_schema(
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "wsi_name": "slide-7",
                "wsi_path": "slide-7.svs",
                "tiling_status": "success",
            }
        ]
    ).to_csv(run_dir / "process_list.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns: .*sample_id"):
        tile_slides(
            [WholeSlide(sample_id="slide-7", image_path=Path("slide-7.svs"))],
            tiling=tiling_config,
            segmentation=segmentation_config,
            filtering=filter_config,
            output_dir=run_dir,
            resume=True,
        )


def test_load_csv_rejects_duplicate_sample_id(tmp_path: Path):
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text(
        "sample_id,image_path,mask_path\n"
        "slide-1,slide-1.svs,slide-1-mask.png\n"
        "slide-1,slide-2.svs,slide-2-mask.png\n"
    )
    cfg = SimpleNamespace(csv=str(csv_path))

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_csv(cfg)


def test_load_tiling_result_rejects_missing_npz_keys(tmp_path: Path):
    npz_path = tmp_path / "broken.tiles.npz"
    meta_path = tmp_path / "broken.tiles.meta.json"
    np.savez(npz_path, tile_index=np.array([0], dtype=np.int32), y_lv0=np.array([20], dtype=np.int64))
    meta_path.write_text(
        json.dumps(
            {
                "sample_id": "broken",
                "image_path": "broken.svs",
                "mask_path": None,
                "backend": "asap",
                "target_spacing_um": 0.5,
                "target_tile_size_px": 224,
                "read_level": 0,
                "read_spacing_um": 0.5,
                "read_tile_size_px": 224,
                "tile_size_lv0": 224,
                "overlap": 0.0,
                "tissue_threshold": 0.1,
                "num_tiles": 1,
                "config_hash": "hash",
            }
        )
    )

    with pytest.raises(ValueError, match="missing keys: x_lv0"):
        load_tiling_result(npz_path, meta_path)


def test_load_tiling_result_wraps_corrupt_npz_errors_with_path(tmp_path: Path):
    npz_path = tmp_path / "corrupt.tiles.npz"
    meta_path = tmp_path / "corrupt.tiles.meta.json"
    npz_path.write_bytes(b"not a valid npz")
    meta_path.write_text(
        json.dumps(
            {
                "sample_id": "corrupt",
                "image_path": "corrupt.svs",
                "mask_path": None,
                "backend": "asap",
                "target_spacing_um": 0.5,
                "target_tile_size_px": 224,
                "read_level": 0,
                "read_spacing_um": 0.5,
                "read_tile_size_px": 224,
                "tile_size_lv0": 224,
                "overlap": 0.0,
                "tissue_threshold": 0.1,
                "num_tiles": 1,
                "config_hash": "hash",
            }
        )
    )

    with pytest.raises(ValueError, match=r"Unable to load tiling npz artifact .*corrupt\.tiles\.npz"):
        load_tiling_result(npz_path, meta_path)


def test_load_tiling_result_rejects_missing_meta_keys(tmp_path: Path):
    npz_path = tmp_path / "broken-meta.tiles.npz"
    meta_path = tmp_path / "broken-meta.tiles.meta.json"
    np.savez(
        npz_path,
        tile_index=np.array([0], dtype=np.int32),
        x_lv0=np.array([10], dtype=np.int64),
        y_lv0=np.array([20], dtype=np.int64),
    )
    meta_path.write_text(
        json.dumps(
            {
                "sample_id": "broken",
                "image_path": "broken.svs",
                "mask_path": None,
                "backend": "asap",
                "target_spacing_um": 0.5,
                "target_tile_size_px": 224,
                "read_level": 0,
                "read_spacing_um": 0.5,
                "read_tile_size_px": 224,
                "tile_size_lv0": 224,
                "overlap": 0.0,
                "tissue_threshold": 0.1,
                "num_tiles": 1,
            }
        )
    )

    with pytest.raises(ValueError, match="missing keys: config_hash"):
        load_tiling_result(npz_path, meta_path)


def test_load_whole_slides_from_rows_parses_current_schema_rows():
    rows = [
        {
            "sample_id": "slide-1",
            "image_path": "slide-1.svs",
            "mask_path": "slide-1-mask.png",
        },
        {
            "sample_id": "slide-2",
            "image_path": "slide-2.svs",
            "mask_path": None,
        },
    ]

    slides = load_whole_slides_from_rows(rows)

    assert slides == [
        WholeSlide(
            sample_id="slide-1",
            image_path=Path("slide-1.svs"),
            mask_path=Path("slide-1-mask.png"),
        ),
        WholeSlide(
            sample_id="slide-2",
            image_path=Path("slide-2.svs"),
            mask_path=None,
        ),
    ]


def test_write_process_list_removes_temp_file_on_failure(monkeypatch, tmp_path: Path):
    created_temp_paths: list[Path] = []
    original_named_temporary_file = tempfile.NamedTemporaryFile

    def _tracking_named_temporary_file(*args, **kwargs):
        handle = original_named_temporary_file(*args, **kwargs)
        created_temp_paths.append(Path(handle.name))
        return handle

    def _raise_to_csv(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("hs2p.api.tempfile.NamedTemporaryFile", _tracking_named_temporary_file)
    monkeypatch.setattr("hs2p.api.pd.DataFrame.to_csv", _raise_to_csv)

    with pytest.raises(OSError, match="disk full"):
        tile_slides(
            [],
            tiling=TilingConfig(0.5, 224, 0.07, 0.1, 0.2, False, True, "asap"),
            segmentation=SegmentationConfig(64, 8, 255, 7, 4, False, True),
            filtering=FilterConfig(224, 4, 2, 8, False, False, 220, 25, 0.9),
            output_dir=tmp_path,
        )

    assert created_temp_paths
    assert all(not path.exists() for path in created_temp_paths)
