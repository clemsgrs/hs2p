import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hs2p.api import (
    FilterConfig,
    PreviewConfig,
    CoordinateOutputMode,
    CoordinateSelectionStrategy,
    SegmentationConfig,
    ResolvedSamplingSpec,
    SlideSpec,
    TilingArtifacts,
    TilingConfig,
    TilingResult,
    compute_config_hash,
    load_tiling_result,
    load_whole_slides_from_rows,
    save_tiling_result,
    tile_slide,
    tile_slides,
    validate_tiling_artifacts,
    write_tiling_preview,
)
from hs2p.configs import (
    FilterConfig as ConfigsFilterConfig,
    PreviewConfig as ConfigsPreviewConfig,
    SegmentationConfig as ConfigsSegmentationConfig,
    TilingConfig as ConfigsTilingConfig,
    default_config,
)
from hs2p.utils import load_csv
from hs2p.wsi import CoordinateExtractionResult
import hs2p.wsi.wsi as wsi_mod
import hs2p.api as api_mod


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
        x=np.array([100, 300], dtype=np.int64),
        y=np.array([200, 400], dtype=np.int64),
        read_level=1,
        read_spacing_um=1.0,
        read_tile_size_px=448,
        resize_factor=2.0,
        tile_size_lv0=448,
        read_step_px=448,
        step_px_lv0=448,
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
        x=np.array([10], dtype=np.int64),
        y=np.array([20], dtype=np.int64),
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
        read_step_px=224,
        step_px_lv0=224,
    )


def test_tile_slide_builds_default_sampling_spec_for_masked_slides(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    captured = {}

    def _fake_extract_coordinates(**kwargs):
        captured["sampling_spec"] = kwargs["sampling_spec"]
        return _fake_extraction()

    monkeypatch.setattr("hs2p.api.extract_coordinates", _fake_extract_coordinates)

    tile_slide(
        SlideSpec(
            sample_id="slide-with-mask",
            image_path=Path("slide.svs"),
            mask_path=Path("slide-mask.png"),
        ),
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
    )

    sampling_spec = captured["sampling_spec"]
    assert sampling_spec is not None
    assert isinstance(sampling_spec, ResolvedSamplingSpec)
    assert sampling_spec.pixel_mapping == {"background": 0, "tissue": 1}
    assert sampling_spec.color_mapping == {"background": None, "tissue": None}
    assert sampling_spec.tissue_percentage == {
        "background": None,
        "tissue": tiling_config.tissue_threshold,
    }
    assert sampling_spec.active_annotations == ("tissue",)


def test_tile_slide_returns_named_arrays(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    monkeypatch.setattr("hs2p.api.extract_coordinates", lambda **_: _fake_extraction())

    result = tile_slide(
        SlideSpec(
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
    np.testing.assert_array_equal(result.x, np.array([100, 300], dtype=np.int64))
    np.testing.assert_array_equal(result.y, np.array([200, 400], dtype=np.int64))
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
        extra={
            "sampling": {
                "output_mode": CoordinateOutputMode.SINGLE_OUTPUT,
                "selection_strategy": CoordinateSelectionStrategy.MERGED_DEFAULT_TILING,
                "resolved_sampling_spec": {
                    "pixel_mapping": {"background": 0, "tissue": 1},
                    "tissue_percentage": {
                        "background": None,
                        "tissue": tiling_config.tissue_threshold,
                    },
                    "color_mapping": {"background": None, "tissue": None},
                    "active_annotations": ["tissue"],
                },
            }
        },
    )


def test_tile_slide_unmasked_hash_stays_legacy(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    monkeypatch.setattr("hs2p.api.extract_coordinates", lambda **_: _fake_extraction())

    result = tile_slide(
        SlideSpec(
            sample_id="slide-1",
            image_path=Path("slide-1.svs"),
            mask_path=None,
        ),
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
    )

    assert result.config_hash == compute_config_hash(
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
    )


def test_masked_default_tiling_hash_changes_when_sampling_semantics_change(
    tiling_config, segmentation_config, filter_config
):
    default_spec = ResolvedSamplingSpec(
        pixel_mapping={"background": 0, "tissue": 1},
        color_mapping={"background": None, "tissue": None},
        tissue_percentage={"background": None, "tissue": tiling_config.tissue_threshold},
        active_annotations=("tissue",),
    )
    changed_spec = ResolvedSamplingSpec(
        pixel_mapping={"background": 0, "tumor": 2},
        color_mapping={"background": None, "tumor": None},
        tissue_percentage={"background": None, "tumor": 0.3},
        active_annotations=("tumor",),
    )

    default_hash = compute_config_hash(
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        extra={
            "sampling": {
                "output_mode": CoordinateOutputMode.SINGLE_OUTPUT,
                "selection_strategy": CoordinateSelectionStrategy.MERGED_DEFAULT_TILING,
                "resolved_sampling_spec": default_spec,
            }
        },
    )
    changed_hash = compute_config_hash(
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        extra={
            "sampling": {
                "output_mode": CoordinateOutputMode.SINGLE_OUTPUT,
                "selection_strategy": CoordinateSelectionStrategy.MERGED_DEFAULT_TILING,
                "resolved_sampling_spec": changed_spec,
            }
        },
    )

    assert default_hash != changed_hash


def test_tile_slide_warns_when_preview_qc_is_requested(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    monkeypatch.setattr("hs2p.api.extract_coordinates", lambda **_: _fake_extraction())

    with pytest.warns(
        UserWarning,
        match="write_tiling_preview\\(\\).*overlay_mask_on_slide\\(\\)",
    ):
        tile_slide(
            SlideSpec(sample_id="slide-qc", image_path=Path("slide-qc.svs")),
            tiling=tiling_config,
            segmentation=segmentation_config,
            filtering=filter_config,
            preview=PreviewConfig(save_mask_preview=True, save_tiling_preview=True),
        )


def test_save_tiling_result_writes_expected_npz_and_json(tmp_path: Path):
    result = TilingResult(
        sample_id="slide-2",
        image_path=Path("slide-2.svs"),
        mask_path=None,
        backend="asap",
        x=np.array([10, 30], dtype=np.int64),
        y=np.array([20, 40], dtype=np.int64),
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
        read_step_px=224,
        step_px_lv0=224,
    )

    artifacts = save_tiling_result(result, output_dir=tmp_path)

    assert artifacts == TilingArtifacts(
        sample_id="slide-2",
        coordinates_npz_path=tmp_path / "tiles" / "slide-2.coordinates.npz",
        coordinates_meta_path=tmp_path / "tiles" / "slide-2.coordinates.meta.json",
        num_tiles=2,
        mask_preview_path=None,
        tiling_preview_path=None,
    )

    tiles = np.load(artifacts.coordinates_npz_path, allow_pickle=False)
    assert set(tiles.files) == {"tile_index", "x", "y", "tissue_fraction"}
    np.testing.assert_array_equal(tiles["tile_index"], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(tiles["x"], np.array([10, 30], dtype=np.int64))
    np.testing.assert_array_equal(tiles["y"], np.array([20, 40], dtype=np.int64))
    np.testing.assert_array_equal(
        tiles["tissue_fraction"],
        np.array([0.3, 0.7], dtype=np.float32),
    )

    meta = json.loads(artifacts.coordinates_meta_path.read_text())
    assert set(meta) == {
        "backend",
        "config_hash",
        "image_path",
        "mask_path",
        "num_tiles",
        "overlap",
        "read_level",
        "read_spacing_um",
        "read_step_px",
        "read_tile_size_px",
        "sample_id",
        "step_px_lv0",
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
        "read_step_px": 224,
        "read_tile_size_px": 224,
        "tile_size_lv0": 224,
        "step_px_lv0": 224,
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
        x=np.array([10, 30], dtype=np.int64),
        y=np.array([20, 40], dtype=np.int64),
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
        read_step_px=202,
        step_px_lv0=202,
    )

    artifacts = save_tiling_result(result, output_dir=tmp_path)
    loaded = load_tiling_result(artifacts.coordinates_npz_path, artifacts.coordinates_meta_path)

    assert loaded.sample_id == result.sample_id
    assert loaded.image_path == result.image_path
    assert loaded.mask_path == result.mask_path
    assert loaded.backend == result.backend
    assert loaded.read_level == result.read_level
    assert loaded.read_spacing_um == result.read_spacing_um
    assert loaded.read_step_px == result.read_step_px
    assert loaded.read_tile_size_px == result.read_tile_size_px
    assert loaded.step_px_lv0 == result.step_px_lv0
    assert loaded.tile_size_lv0 == result.tile_size_lv0
    assert loaded.overlap == result.overlap
    assert loaded.tissue_threshold == result.tissue_threshold
    assert loaded.num_tiles == result.num_tiles
    assert loaded.config_hash == result.config_hash
    np.testing.assert_array_equal(loaded.x, result.x)
    np.testing.assert_array_equal(loaded.y, result.y)
    np.testing.assert_array_equal(loaded.tile_index, result.tile_index)
    np.testing.assert_array_equal(loaded.tissue_fraction, result.tissue_fraction)


def test_write_tiling_preview_writes_expected_preview(monkeypatch, tmp_path: Path):
    result = TilingResult(
        sample_id="slide-preview",
        image_path=Path("slide-preview.svs"),
        mask_path=None,
        backend="asap",
        x=np.array([10, 30], dtype=np.int64),
        y=np.array([20, 40], dtype=np.int64),
        tile_index=np.array([0, 1], dtype=np.int32),
        tissue_fraction=None,
        target_spacing_um=0.5,
        target_tile_size_px=224,
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        tile_size_lv0=224,
        overlap=0.0,
        tissue_threshold=0.1,
        num_tiles=2,
        config_hash="preview-hash",
    )

    def _fake_write_coordinate_preview(**kwargs):
        save_dir = Path(kwargs["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{kwargs['sample_id']}.jpg").write_bytes(b"preview")

    monkeypatch.setattr("hs2p.api.write_coordinate_preview", _fake_write_coordinate_preview)

    preview_path = write_tiling_preview(
        result=result,
        output_dir=tmp_path,
        downsample=16,
    )

    assert preview_path == (tmp_path / "preview" / "tiling" / "slide-preview.jpg")
    assert preview_path.is_file()


def test_tile_slides_defers_preview_writes_until_after_next_slide_compute(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    events: list[str] = []

    def _fake_compute_tiling_result(
        whole_slide,
        *,
        tiling,
        segmentation,
        filtering,
        mask_preview_path,
        num_workers,
        config_hash=None,
    ):
        del tiling, segmentation, filtering, mask_preview_path, num_workers, config_hash
        events.append(f"compute:{whole_slide.sample_id}")
        return _build_result(
            sample_id=whole_slide.sample_id,
            image_path=str(whole_slide.image_path),
            config_hash="hash",
        )

    def _fake_save_tiling_result(result, output_dir, tiles_dir=None):
        del tiles_dir
        tiles_dir = Path(output_dir) / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        npz_path = tiles_dir / f"{result.sample_id}.coordinates.npz"
        meta_path = tiles_dir / f"{result.sample_id}.coordinates.meta.json"
        npz_path.write_bytes(b"npz")
        meta_path.write_text("{}")
        return TilingArtifacts(
            sample_id=result.sample_id,
            coordinates_npz_path=npz_path,
            coordinates_meta_path=meta_path,
            num_tiles=result.num_tiles,
        )

    def _fake_write_tiling_preview(*, result, output_dir, downsample):
        del downsample
        events.append(f"preview:{result.sample_id}")
        save_dir = Path(output_dir) / "preview" / "tiling"
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{result.sample_id}.jpg"
        path.write_bytes(b"preview")
        return path

    class _FakeFuture:
        def __init__(self, fn, kwargs):
            self._fn = fn
            self._kwargs = kwargs

        def result(self):
            return self._fn(**self._kwargs)

    class _FakeThreadPoolExecutor:
        def __init__(self, max_workers):
            assert max_workers == 1

        def submit(self, fn, **kwargs):
            return _FakeFuture(fn, kwargs)

        def shutdown(self, wait=True):
            return None

    monkeypatch.setattr("hs2p.api._compute_tiling_result", _fake_compute_tiling_result)
    monkeypatch.setattr("hs2p.api.save_tiling_result", _fake_save_tiling_result)
    monkeypatch.setattr("hs2p.api.write_tiling_preview", _fake_write_tiling_preview)
    monkeypatch.setattr("hs2p.api.ThreadPoolExecutor", _FakeThreadPoolExecutor)

    artifacts = tile_slides(
        [
            SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs")),
            SlideSpec(sample_id="slide-2", image_path=Path("slide-2.svs")),
        ],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        preview=PreviewConfig(save_tiling_preview=True),
        output_dir=tmp_path,
    )

    assert [artifact.sample_id for artifact in artifacts] == ["slide-1", "slide-2"]
    assert events == [
        "compute:slide-1",
        "compute:slide-2",
        "preview:slide-1",
        "preview:slide-2",
    ]


def test_tile_slides_uses_slide_level_pool_and_preserves_input_order(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    seen = {"pool_processes": None, "inner_workers": []}

    def _fake_compute_and_save(request):
        seen["inner_workers"].append(request.num_workers)
        tiles_dir = Path(request.output_dir) / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        npz_path = tiles_dir / f"{request.whole_slide.sample_id}.coordinates.npz"
        meta_path = tiles_dir / f"{request.whole_slide.sample_id}.coordinates.meta.json"
        npz_path.write_bytes(b"npz")
        meta_path.write_text("{}")
        return SimpleNamespace(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=True,
            artifact=TilingArtifacts(
                sample_id=request.whole_slide.sample_id,
                coordinates_npz_path=npz_path,
                coordinates_meta_path=meta_path,
                num_tiles=1,
            ),
            mask_preview_path=None,
            error=None,
            traceback_text=None,
        )

    class _FakePool:
        def __init__(self, processes):
            seen["pool_processes"] = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, fn, args_list):
            requests = list(args_list)
            for args in reversed(requests):
                yield fn(args)

    monkeypatch.setattr(
        "hs2p.api._compute_and_save_tiling_artifacts_from_request",
        _fake_compute_and_save,
        raising=False,
    )
    monkeypatch.setattr(
        "hs2p.api.save_tiling_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("parent should not write tiling artifacts in pooled mode")
        ),
    )
    monkeypatch.setattr("hs2p.api.mp.Pool", _FakePool)

    artifacts = tile_slides(
        [
            SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs")),
            SlideSpec(sample_id="slide-2", image_path=Path("slide-2.svs")),
            SlideSpec(sample_id="slide-3", image_path=Path("slide-3.svs")),
        ],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        output_dir=tmp_path,
        num_workers=2,
    )

    assert seen["pool_processes"] == 2
    assert seen["inner_workers"] == [1, 1, 1]
    assert [artifact.sample_id for artifact in artifacts] == [
        "slide-1",
        "slide-2",
        "slide-3",
    ]
    process_df = pd.read_csv(tmp_path / "process_list.csv")
    assert process_df["sample_id"].tolist() == ["slide-1", "slide-2", "slide-3"]


def test_tile_slides_assigns_inner_workers_when_batch_is_small(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    seen = {"pool_processes": None, "inner_workers": []}

    def _fake_compute_and_save(request):
        seen["inner_workers"].append(request.num_workers)
        tiles_dir = Path(request.output_dir) / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        npz_path = tiles_dir / f"{request.whole_slide.sample_id}.coordinates.npz"
        meta_path = tiles_dir / f"{request.whole_slide.sample_id}.coordinates.meta.json"
        npz_path.write_bytes(b"npz")
        meta_path.write_text("{}")
        return SimpleNamespace(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=True,
            artifact=TilingArtifacts(
                sample_id=request.whole_slide.sample_id,
                coordinates_npz_path=npz_path,
                coordinates_meta_path=meta_path,
                num_tiles=1,
            ),
            mask_preview_path=None,
            error=None,
            traceback_text=None,
        )

    class _FakePool:
        def __init__(self, processes):
            seen["pool_processes"] = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, fn, args_list):
            for args in args_list:
                yield fn(args)

    monkeypatch.setattr(
        "hs2p.api._compute_and_save_tiling_artifacts_from_request",
        _fake_compute_and_save,
        raising=False,
    )
    monkeypatch.setattr("hs2p.api.mp.Pool", _FakePool)

    tile_slides(
        [
            SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs")),
            SlideSpec(sample_id="slide-2", image_path=Path("slide-2.svs")),
        ],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        output_dir=tmp_path,
        num_workers=8,
    )

    assert seen["pool_processes"] == 2
    assert seen["inner_workers"] == [4, 4]


def test_compute_request_passes_inner_workers_to_tile_extraction(
    monkeypatch, tmp_path: Path
):
    seen = {}

    def _fake_compute_tiling_result(*args, **kwargs):
        del args, kwargs
        return TilingResult(
            sample_id="slide-1",
            image_path=Path("slide-1.svs"),
            mask_path=None,
            backend="cucim",
            x=np.array([10], dtype=np.int64),
            y=np.array([20], dtype=np.int64),
            tile_index=np.array([0], dtype=np.int32),
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

    def _fake_extract_tiles_to_tar(result, output_dir, *, filter_params=None, num_workers=1, **kwargs):
        del output_dir, filter_params, kwargs
        seen["num_workers"] = num_workers
        return tmp_path / "tiles" / "slide-1.tiles.tar", result

    def _fake_save_tiling_result(result, output_dir, *, tiles_dir=None):
        del tiles_dir
        tiles_dir = Path(output_dir) / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        npz_path = tiles_dir / f"{result.sample_id}.coordinates.npz"
        meta_path = tiles_dir / f"{result.sample_id}.coordinates.meta.json"
        npz_path.write_bytes(b"npz")
        meta_path.write_text("{}")
        return TilingArtifacts(
            sample_id=result.sample_id,
            coordinates_npz_path=npz_path,
            coordinates_meta_path=meta_path,
            num_tiles=result.num_tiles,
        )

    request = api_mod._SlideComputeRequest(
        input_index=0,
        whole_slide=SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs")),
        tiling=TilingConfig(
            backend="cucim",
            target_spacing_um=0.5,
            target_tile_size_px=224,
            tolerance=0.07,
            overlap=0.0,
            tissue_threshold=0.1,
            drop_holes=False,
            use_padding=True,
        ),
        segmentation=SegmentationConfig(64, 8, 255, 7, 4, False, True),
        filtering=FilterConfig(224, 4, 2, 8, False, False, 220, 25, 0.9),
        config_hash="hash",
        mask_preview_path=None,
        output_dir=tmp_path,
        num_workers=6,
        save_tiles=True,
    )

    monkeypatch.setattr(api_mod, "_compute_tiling_result", _fake_compute_tiling_result)
    monkeypatch.setattr(api_mod, "extract_tiles_to_tar", _fake_extract_tiles_to_tar)
    monkeypatch.setattr(api_mod, "save_tiling_result", _fake_save_tiling_result)
    response = api_mod._compute_tiling_result_from_request(request)

    assert response.ok
    assert seen["num_workers"] == 6


def test_save_tiling_result_rejects_invalid_tile_index(tmp_path: Path):
    invalid = TilingResult(
        sample_id="broken-slide",
        image_path=Path("broken.svs"),
        mask_path=None,
        backend="asap",
        x=np.array([10], dtype=np.int64),
        y=np.array([20], dtype=np.int64),
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
        x=np.array([[10, 11]], dtype=np.int64),
        y=np.array([20], dtype=np.int64),
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

    with pytest.raises(ValueError, match="x must be a 1D array"):
        save_tiling_result(invalid, output_dir=tmp_path)


def test_save_tiling_result_cleans_up_partial_outputs_when_metadata_write_fails(
    monkeypatch, tmp_path: Path
):
    result = _build_result(sample_id="slide-clean", image_path="slide-clean.svs")
    tiles_dir = tmp_path / "tiles"

    def _raise_json(*args, **kwargs):
        raise RuntimeError("json failure")

    monkeypatch.setattr("hs2p.api.json.dumps", _raise_json)

    with pytest.raises(RuntimeError, match="json failure"):
        save_tiling_result(result, output_dir=tmp_path)

    assert not (tiles_dir / "slide-clean.coordinates.npz").exists()
    assert not (tiles_dir / "slide-clean.coordinates.meta.json").exists()
    assert list(tiles_dir.glob("*")) == []


def test_tile_slide_rejects_tissue_fraction_shape_mismatch(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    def _bad_extraction(**kwargs):
        return CoordinateExtractionResult(
            coordinates=[(100, 200), (300, 400)],
            contour_indices=[0, 0],
            tissue_percentages=[0.25],
            x=np.array([100, 300], dtype=np.int64),
            y=np.array([200, 400], dtype=np.int64),
            read_level=1,
            read_spacing_um=1.0,
            read_tile_size_px=448,
            read_step_px=448,
            resize_factor=2.0,
            tile_size_lv0=448,
            step_px_lv0=448,
        )

    monkeypatch.setattr("hs2p.api.extract_coordinates", _bad_extraction)

    with pytest.raises(ValueError, match="tissue_percentages length mismatch"):
        tile_slide(
            SlideSpec(sample_id="slide-bad-tissue", image_path=Path("slide.svs")),
            tiling=tiling_config,
            segmentation=segmentation_config,
            filtering=filter_config,
        )


def test_validate_tiling_artifacts_rejects_mismatched_hash(tmp_path: Path):
    result = _build_result(sample_id="slide-3", image_path="slide-3.svs")
    artifacts = save_tiling_result(result, output_dir=tmp_path)

    with pytest.raises(ValueError, match="config_hash"):
        validate_tiling_artifacts(
            whole_slide=SlideSpec(sample_id="slide-3", image_path=Path("slide-3.svs")),
            coordinates_npz_path=artifacts.coordinates_npz_path,
            coordinates_meta_path=artifacts.coordinates_meta_path,
            expected_config_hash="different-hash",
        )


def test_validate_tiling_artifacts_rejects_mismatched_image_path(tmp_path: Path):
    result = _build_result(sample_id="slide-4", image_path="stored-slide.svs")
    artifacts = save_tiling_result(result, output_dir=tmp_path)

    with pytest.raises(ValueError, match="image_path mismatch"):
        validate_tiling_artifacts(
            whole_slide=SlideSpec(
                sample_id="slide-4", image_path=Path("requested-slide.svs")
            ),
            coordinates_npz_path=artifacts.coordinates_npz_path,
            coordinates_meta_path=artifacts.coordinates_meta_path,
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
            whole_slide=SlideSpec(
                sample_id="slide-5",
                image_path=Path("slide-5.svs"),
                mask_path=Path("requested-mask.png"),
            ),
            coordinates_npz_path=artifacts.coordinates_npz_path,
            coordinates_meta_path=artifacts.coordinates_meta_path,
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
        SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs")),
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
    )
    precomputed_artifacts = save_tiling_result(
        source_result, output_dir=precomputed_root
    )

    def _unexpected_extract(**kwargs):
        raise AssertionError(
            "tile extraction should not run when precomputed tiles are reused"
        )

    monkeypatch.setattr("hs2p.api.extract_coordinates", _unexpected_extract)

    artifacts = tile_slides(
        [SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs"))],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        output_dir=tmp_path / "run",
        read_coordinates_from=precomputed_artifacts.coordinates_npz_path.parent,
        resume=False,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert isinstance(artifact, TilingArtifacts)
    assert artifact.coordinates_npz_path == precomputed_artifacts.coordinates_npz_path
    assert artifact.coordinates_meta_path == precomputed_artifacts.coordinates_meta_path
    assert artifact.num_tiles == 2

    process_df = pd.read_csv(tmp_path / "run" / "process_list.csv")
    assert list(process_df.columns) == [
        "sample_id",
        "image_path",
        "mask_path",
        "tiling_status",
        "num_tiles",
        "coordinates_npz_path",
        "coordinates_meta_path",
        "tiles_tar_path",
        "error",
        "traceback",
    ]
    row = process_df.to_dict(orient="records")[0]
    assert row["sample_id"] == "slide-1"
    assert row["image_path"] == "slide-1.svs"
    assert pd.isna(row["mask_path"])
    assert row["tiling_status"] == "success"
    assert row["num_tiles"] == 2
    assert row["coordinates_npz_path"] == str(precomputed_artifacts.coordinates_npz_path)
    assert row["coordinates_meta_path"] == str(precomputed_artifacts.coordinates_meta_path)
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
            x=np.array([], dtype=np.int64),
            y=np.array([], dtype=np.int64),
            read_level=0,
            read_spacing_um=0.5,
            read_tile_size_px=224,
            read_step_px=224,
            resize_factor=1.0,
            tile_size_lv0=224,
            step_px_lv0=224,
        ),
    )
    monkeypatch.setattr(
        "hs2p.api.write_coordinate_preview",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("preview rendering should be skipped for zero tiles")
        ),
    )

    artifacts = tile_slides(
        [SlideSpec(sample_id="slide-0", image_path=Path("slide-0.svs"))],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        preview=PreviewConfig(save_tiling_preview=True),
        output_dir=tmp_path,
    )

    assert len(artifacts) == 1
    assert artifacts[0].num_tiles == 0
    assert artifacts[0].tiling_preview_path is None


def test_tile_slides_writes_preview_paths_when_previews_are_saved(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    monkeypatch.setattr("hs2p.api.extract_coordinates", lambda **_: _fake_extraction())

    def _fake_write_coordinate_preview(**kwargs):
        save_dir = Path(kwargs["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        sample_id = kwargs.get("sample_id", "preview")
        (save_dir / f"{sample_id}.jpg").write_bytes(b"preview")

    monkeypatch.setattr("hs2p.api.write_coordinate_preview", _fake_write_coordinate_preview)

    expected_mask_path = tmp_path / "preview" / "mask" / "slide-preview.jpg"

    def _fake_extract_with_mask_preview(**kwargs):
        mask_preview_path = kwargs["mask_preview_path"]
        if mask_preview_path is not None:
            mask_preview_path.parent.mkdir(parents=True, exist_ok=True)
            mask_preview_path.write_bytes(b"mask-preview")
        return _fake_extraction()

    monkeypatch.setattr("hs2p.api.extract_coordinates", _fake_extract_with_mask_preview)

    artifacts = tile_slides(
        [SlideSpec(sample_id="slide-preview", image_path=Path("slide-preview.svs"))],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        preview=PreviewConfig(save_mask_preview=True, save_tiling_preview=True),
        output_dir=tmp_path,
    )

    assert len(artifacts) == 1
    assert artifacts[0].mask_preview_path == expected_mask_path
    assert artifacts[0].mask_preview_path.is_file()
    assert (
        artifacts[0].tiling_preview_path
        == tmp_path / "preview" / "tiling" / "slide-preview.jpg"
    )
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
                "coordinates_npz_path": str(artifacts.coordinates_npz_path),
                "coordinates_meta_path": str(artifacts.coordinates_meta_path),
                "error": np.nan,
                "traceback": np.nan,
            }
        ]
    ).to_csv(tmp_path / "run" / "process_list.csv", index=False)

    monkeypatch.setattr(
        "hs2p.api.extract_coordinates",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("should not recompute stale resumed tiles")
        ),
    )

    reused = tile_slides(
        [SlideSpec(sample_id="slide-6", image_path=Path("requested-slide.svs"))],
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
    assert pd.isna(row["coordinates_npz_path"])
    assert pd.isna(row["coordinates_meta_path"])
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
        [SlideSpec(sample_id="slide-log", image_path=Path("slide-log.svs"))],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        output_dir=tmp_path,
    )

    assert artifacts == []
    captured = capsys.readouterr()
    assert "[tile_slides] FAILED slide-log: boom" in captured.out


def test_tile_slides_computes_resume_hash_once_per_batch(
    monkeypatch,
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
                "sample_id": "slide-1",
                "image_path": "slide-1.svs",
                "mask_path": np.nan,
                "tiling_status": "success",
                "num_tiles": 1,
                "coordinates_npz_path": "slide-1.coordinates.npz",
                "coordinates_meta_path": "slide-1.coordinates.meta.json",
                "error": np.nan,
                "traceback": np.nan,
            },
            {
                "sample_id": "slide-2",
                "image_path": "slide-2.svs",
                "mask_path": np.nan,
                "tiling_status": "success",
                "num_tiles": 1,
                "coordinates_npz_path": "slide-2.coordinates.npz",
                "coordinates_meta_path": "slide-2.coordinates.meta.json",
                "error": np.nan,
                "traceback": np.nan,
            },
        ]
    ).to_csv(run_dir / "process_list.csv", index=False)

    call_count = {"count": 0}

    def _fake_compute_config_hash(**kwargs):
        call_count["count"] += 1
        return "expected-hash"

    def _fake_validate_tiling_artifacts(**kwargs):
        sample_id = kwargs["whole_slide"].sample_id
        return TilingArtifacts(
            sample_id=sample_id,
            coordinates_npz_path=Path(f"{sample_id}.coordinates.npz"),
            coordinates_meta_path=Path(f"{sample_id}.coordinates.meta.json"),
            num_tiles=1,
        )

    monkeypatch.setattr("hs2p.api.compute_config_hash", _fake_compute_config_hash)
    monkeypatch.setattr(
        "hs2p.api.validate_tiling_artifacts", _fake_validate_tiling_artifacts
    )
    monkeypatch.setattr(
        "hs2p.api.extract_coordinates",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("resume path should not recompute tiles")
        ),
    )

    artifacts = tile_slides(
        [
            SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs")),
            SlideSpec(sample_id="slide-2", image_path=Path("slide-2.svs")),
        ],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        output_dir=run_dir,
        resume=True,
    )

    assert [artifact.sample_id for artifact in artifacts] == ["slide-1", "slide-2"]
    assert call_count["count"] == 1


def test_tile_slides_reuses_precomputed_hash_during_compute(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    call_count = {"count": 0}

    def _fake_compute_config_hash(**kwargs):
        del kwargs
        call_count["count"] += 1
        return "expected-hash"

    monkeypatch.setattr("hs2p.api.compute_config_hash", _fake_compute_config_hash)
    monkeypatch.setattr("hs2p.api.extract_coordinates", lambda **_: _fake_extraction())

    artifacts = tile_slides(
        [
            SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs")),
            SlideSpec(sample_id="slide-2", image_path=Path("slide-2.svs")),
        ],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        output_dir=tmp_path,
    )

    assert [artifact.sample_id for artifact in artifacts] == ["slide-1", "slide-2"]
    assert call_count["count"] == 1


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
            [SlideSpec(sample_id="slide-7", image_path=Path("slide-7.svs"))],
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
    npz_path = tmp_path / "broken.coordinates.npz"
    meta_path = tmp_path / "broken.coordinates.meta.json"
    np.savez(
        npz_path,
        tile_index=np.array([0], dtype=np.int32),
        y=np.array([20], dtype=np.int64),
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
                "config_hash": "hash",
            }
        )
    )

    with pytest.raises(ValueError, match="missing keys: x"):
        load_tiling_result(npz_path, meta_path)


def test_load_tiling_result_wraps_corrupt_npz_errors_with_path(tmp_path: Path):
    npz_path = tmp_path / "corrupt.coordinates.npz"
    meta_path = tmp_path / "corrupt.coordinates.meta.json"
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

    with pytest.raises(
        ValueError, match=r"Unable to load tiling npz artifact .*corrupt\.coordinates\.npz"
    ):
        load_tiling_result(npz_path, meta_path)


def test_load_tiling_result_rejects_missing_meta_keys(tmp_path: Path):
    npz_path = tmp_path / "broken-meta.coordinates.npz"
    meta_path = tmp_path / "broken-meta.coordinates.meta.json"
    np.savez(
        npz_path,
        tile_index=np.array([0], dtype=np.int32),
        x=np.array([10], dtype=np.int64),
        y=np.array([20], dtype=np.int64),
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
        SlideSpec(
            sample_id="slide-1",
            image_path=Path("slide-1.svs"),
            mask_path=Path("slide-1-mask.png"),
        ),
        SlideSpec(
            sample_id="slide-2",
            image_path=Path("slide-2.svs"),
            mask_path=None,
        ),
    ]


def test_load_whole_slides_from_rows_treats_nan_like_mask_values_as_missing():
    rows = [
        {
            "sample_id": "slide-nan",
            "image_path": "slide-nan.svs",
            "mask_path": np.nan,
        },
        {
            "sample_id": "slide-none-string",
            "image_path": "slide-none-string.svs",
            "mask_path": "None",
        },
        {
            "sample_id": "slide-nan-string",
            "image_path": "slide-nan-string.svs",
            "mask_path": "nan",
        },
    ]

    slides = load_whole_slides_from_rows(rows)

    assert slides == [
        SlideSpec(
            sample_id="slide-nan",
            image_path=Path("slide-nan.svs"),
            mask_path=None,
        ),
        SlideSpec(
            sample_id="slide-none-string",
            image_path=Path("slide-none-string.svs"),
            mask_path=None,
        ),
        SlideSpec(
            sample_id="slide-nan-string",
            image_path=Path("slide-nan-string.svs"),
            mask_path=None,
        ),
    ]


def test_coordinate_extraction_result_is_not_tuple_iterable():
    result = CoordinateExtractionResult(
        coordinates=[(1, 2)],
        contour_indices=[0],
        tissue_percentages=[0.5],
        x=np.array([1], dtype=np.int64),
        y=np.array([2], dtype=np.int64),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        read_step_px=224,
        resize_factor=1.0,
        tile_size_lv0=224,
        step_px_lv0=224,
    )

    with pytest.raises(TypeError, match="not iterable"):
        tuple(result)


def test_coordinate_extraction_result_rebuilds_coordinates_from_x_and_y_arrays():
    result = CoordinateExtractionResult(
        contour_indices=[0, 1],
        tissue_percentages=[0.25, 0.75],
        x=np.array([10, 30], dtype=np.int64),
        y=np.array([20, 40], dtype=np.int64),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        read_step_px=224,
        resize_factor=1.0,
        tile_size_lv0=224,
        step_px_lv0=224,
    )

    assert result.coordinates == [(10, 20), (30, 40)]


def test_write_process_list_removes_temp_file_on_failure(monkeypatch, tmp_path: Path):
    created_temp_paths: list[Path] = []
    original_named_temporary_file = tempfile.NamedTemporaryFile

    def _tracking_named_temporary_file(*args, **kwargs):
        handle = original_named_temporary_file(*args, **kwargs)
        created_temp_paths.append(Path(handle.name))
        return handle

    def _raise_to_csv(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "hs2p.api.tempfile.NamedTemporaryFile", _tracking_named_temporary_file
    )
    monkeypatch.setattr("hs2p.api.pd.DataFrame.to_csv", _raise_to_csv)

    with pytest.raises(OSError, match="disk full"):
        tile_slides(
            [],
            tiling=TilingConfig(
                backend="asap",
                target_spacing_um=0.5,
                target_tile_size_px=224,
                tolerance=0.07,
                overlap=0.1,
                tissue_threshold=0.2,
                drop_holes=False,
                use_padding=True,
            ),
            segmentation=SegmentationConfig(64, 8, 255, 7, 4, False, True),
            filtering=FilterConfig(224, 4, 2, 8, False, False, 220, 25, 0.9),
            output_dir=tmp_path,
        )

    assert created_temp_paths
    assert all(not path.exists() for path in created_temp_paths)


def test_config_dataclasses_apply_package_defaults_for_secondary_parameters():
    tiling = TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
    )
    segmentation = SegmentationConfig(downsample=64)
    filtering = FilterConfig(ref_tile_size=224, a_t=4, a_h=2)

    assert tiling.drop_holes == default_config.tiling.params.drop_holes
    assert tiling.use_padding == default_config.tiling.params.use_padding
    assert tiling.backend == "auto"
    assert tiling.requested_backend == "auto"
    assert segmentation.sthresh == default_config.tiling.seg_params.sthresh
    assert segmentation.sthresh_up == default_config.tiling.seg_params.sthresh_up
    assert segmentation.mthresh == default_config.tiling.seg_params.mthresh
    assert segmentation.close == default_config.tiling.seg_params.close
    assert segmentation.use_otsu == default_config.tiling.seg_params.use_otsu
    assert segmentation.use_hsv == default_config.tiling.seg_params.use_hsv
    assert filtering.max_n_holes == default_config.tiling.filter_params.max_n_holes
    assert filtering.filter_white == default_config.tiling.filter_params.filter_white
    assert filtering.filter_black == default_config.tiling.filter_params.filter_black
    assert (
        filtering.white_threshold == default_config.tiling.filter_params.white_threshold
    )
    assert (
        filtering.black_threshold == default_config.tiling.filter_params.black_threshold
    )
    assert filtering.fraction_threshold == pytest.approx(
        default_config.tiling.filter_params.fraction_threshold
    )


def test_hs2p_configs_reexports_runtime_config_models():
    assert ConfigsTilingConfig is TilingConfig
    assert ConfigsSegmentationConfig is SegmentationConfig
    assert ConfigsFilterConfig is FilterConfig
    assert ConfigsPreviewConfig is PreviewConfig
