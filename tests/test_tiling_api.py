import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import hs2p.preprocessing as preprocessing_mod
from hs2p.api import (
    CompatibilitySpec,
    FilterConfig,
    PreviewConfig,
    CoordinateOutputMode,
    CoordinateSelectionStrategy,
    SegmentationConfig,
    SlideSpec,
    TilingArtifacts,
    TilingConfig,
    load_tiling_result,
    maybe_load_existing_artifacts,
    save_tiling_result,
    tile_slide,
    tile_slides,
    validate_tiling_artifacts,
    write_tiling_preview,
)
from hs2p.artifacts import load_whole_slides_from_rows
from hs2p.configs import (
    FilterConfig as ConfigsFilterConfig,
    PreviewConfig as ConfigsPreviewConfig,
    SegmentationConfig as ConfigsSegmentationConfig,
    TilingConfig as ConfigsTilingConfig,
    default_config,
)
from hs2p.configs.resolvers import resolve_preview_config
from hs2p.utils import load_csv
from hs2p.wsi import CoordinateExtractionResult
from hs2p.wsi.streaming.plans import resolve_read_step_px
import hs2p.wsi.wsi as wsi_mod
import hs2p.api as api_mod


@pytest.fixture
def tiling_config() -> TilingConfig:
    return TilingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.07,
        overlap=0.1,
        tissue_threshold=0.2,
        backend="asap",
    )


@pytest.fixture
def segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(
        method="hsv",
        downsample=64,
        sthresh=8,
        sthresh_up=255,
        mthresh=7,
        close=4,
    )


@pytest.fixture
def filter_config() -> FilterConfig:
    return FilterConfig(
        ref_tile_size=224,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def _coords_array(result) -> np.ndarray:
    x = np.asarray(result.x, dtype=np.int64)
    y = np.asarray(result.y, dtype=np.int64)
    return np.column_stack((x, y))


def _coords_list(result) -> list[tuple[int, int]]:
    return list(zip(result.x, result.y))


def _split_coords(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    return coords[:, 0], coords[:, 1]


def _fake_extraction() -> CoordinateExtractionResult:
    return CoordinateExtractionResult(
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
) -> preprocessing_mod.TilingResult:
    return _build_preprocessing_result(
        sample_id=sample_id,
        image_path=image_path,
        mask_path=mask_path,
        coords=np.array([[10, 20]], dtype=np.int64),
        tissue_fractions=np.array([0.0], dtype=np.float32),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        tile_size_lv0=224,
        overlap=0.0,
        tissue_threshold=0.1,
        step_px_lv0=224,
    )


def _build_preprocessing_result(
    *,
    sample_id: str = "slide-preprocess",
    image_path: str = "slide-preprocess.svs",
    mask_path: str | None = None,
    annotation: str | None = None,
    selection_strategy: str | None = None,
    output_mode: str | None = None,
    coords: np.ndarray | None = None,
    tissue_fractions: np.ndarray | None = None,
    backend: str = "asap",
    requested_backend: str = "asap",
    read_level: int = 1,
    read_spacing_um: float = 1.0,
    read_tile_size_px: int = 448,
    tile_size_lv0: int = 448,
    overlap: float = 0.1,
    tissue_threshold: float = 0.2,
    step_px_lv0: int = 403,
) -> preprocessing_mod.TilingResult:
    if coords is None:
        coords = np.array([[100, 200], [300, 400]], dtype=np.int64)
    if tissue_fractions is None:
        tissue_fractions = np.array([0.25, 0.75], dtype=np.float32)
    coords = np.asarray(coords, dtype=np.int64)
    tissue_fractions = np.asarray(tissue_fractions, dtype=np.float32)
    if coords.shape[0] != tissue_fractions.shape[0]:
        raise ValueError("coords and tissue_fractions must be aligned")
    x, y = _split_coords(coords)
    return preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            x=x,
            y=y,
            tissue_fractions=tissue_fractions,
            tile_index=np.arange(coords.shape[0], dtype=np.int32),
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            read_level=read_level,
            read_tile_size_px=read_tile_size_px,
            read_spacing_um=read_spacing_um,
            tile_size_lv0=tile_size_lv0,
            is_within_tolerance=read_spacing_um == 0.5,
            base_spacing_um=0.25,
            slide_dimensions=[1000, 1200],
            level_downsamples=[1.0, 4.0],
            overlap=overlap,
            min_tissue_fraction=tissue_threshold,
            tissue_mask=np.full((8, 8), 255, dtype=np.uint8),
        ),
        sample_id=sample_id,
        image_path=image_path,
        backend=backend,
        requested_backend=requested_backend,
        tolerance=0.07,
        step_px_lv0=step_px_lv0,
        tissue_method="precomputed_mask" if mask_path is not None else "hsv",
        seg_downsample=64,
        seg_level=2,
        seg_spacing_um=16.0,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=224,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
        mask_path=mask_path,
        annotation=annotation,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
    )


def _artifact_compatibility(
    *,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
    selection_strategy: str | None = None,
    output_mode: str | None = None,
    annotation: str | None = None,
) -> CompatibilitySpec:
    return CompatibilitySpec(
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        selection_strategy=selection_strategy,
        output_mode=output_mode,
        annotation=annotation,
    )


def _patch_preprocess_slide(
    monkeypatch,
    *,
    result: preprocessing_mod.TilingResult | None = None,
    hook=None,
    error: Exception | None = None,
):
    def _fake_preprocess_slide(**kwargs):
        if hook is not None:
            hook(kwargs)
        if error is not None:
            raise error
        assert result is not None
        if callable(result):
            return result(**kwargs)
        return result

    monkeypatch.setattr(api_mod, "preprocess_slide", _fake_preprocess_slide)


def _save_valid_preprocessing_artifact(
    tmp_path: Path,
    *,
    sample_id: str = "slide-artifact",
) -> tuple[Path, Path]:
    result = _build_preprocessing_result(
        sample_id=sample_id,
        image_path=f"{sample_id}.svs",
        mask_path=f"{sample_id}-mask.png",
    )
    paths = preprocessing_mod._save_tiling_result(result, output_dir=tmp_path)
    return Path(paths["npz"]), Path(paths["meta"])


def test_tile_slide_passes_default_sampling_semantics_for_masked_slides(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    captured = {}

    _patch_preprocess_slide(
        monkeypatch,
        result=_build_preprocessing_result(
            sample_id="slide-with-mask",
            image_path="slide.svs",
            mask_path="slide-mask.png",
            selection_strategy=CoordinateSelectionStrategy.MERGED_DEFAULT_TILING,
            output_mode=CoordinateOutputMode.SINGLE_OUTPUT,
        ),
        hook=captured.update,
    )

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

    assert (
        captured["selection_strategy"]
        == CoordinateSelectionStrategy.MERGED_DEFAULT_TILING
    )
    assert captured["output_mode"] == CoordinateOutputMode.SINGLE_OUTPUT
    assert captured["tissue_mask_path"] == Path("slide-mask.png")


def test_tile_slide_returns_named_arrays(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    _patch_preprocess_slide(
        monkeypatch,
        result=_build_preprocessing_result(
            sample_id="slide-1",
            image_path="slide-1.svs",
            mask_path="slide-1-mask.png",
            selection_strategy=CoordinateSelectionStrategy.MERGED_DEFAULT_TILING,
            output_mode=CoordinateOutputMode.SINGLE_OUTPUT,
        ),
    )

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

    assert isinstance(result, preprocessing_mod.TilingResult)
    np.testing.assert_array_equal(
        _coords_array(result), np.array([[100, 200], [300, 400]], dtype=np.int64)
    )
    np.testing.assert_array_equal(result.tile_index, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        result.tissue_fractions,
        np.array([0.25, 0.75], dtype=np.float32),
    )
    assert result.sample_id == "slide-1"
    assert result.image_path == Path("slide-1.svs")
    assert result.mask_path == Path("slide-1-mask.png")
    assert result.read_level == 1
    assert result.read_spacing_um == pytest.approx(1.0)
    assert result.read_tile_size_px == 448
    assert result.tile_size_lv0 == 448
    assert result.requested_spacing_um == tiling_config.requested_spacing_um
    assert result.requested_tile_size_px == tiling_config.requested_tile_size_px
    assert result.overlap == tiling_config.overlap
    assert result.min_tissue_fraction == tiling_config.tissue_threshold
    assert len(result.x) == 2


def test_compute_tiling_result_uses_preprocessing_core(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    captured = {}

    _patch_preprocess_slide(
        monkeypatch,
        result=_build_preprocessing_result(
            sample_id="slide-1",
            image_path="slide-1.svs",
            mask_path="slide-1-mask.png",
            selection_strategy=CoordinateSelectionStrategy.MERGED_DEFAULT_TILING,
            output_mode=CoordinateOutputMode.SINGLE_OUTPUT,
        ),
        hook=captured.update,
    )

    result = api_mod._compute_tiling_result(
        SlideSpec(
            sample_id="slide-1",
            image_path=Path("slide-1.svs"),
            mask_path=Path("slide-1-mask.png"),
            spacing_at_level_0=0.25,
        ),
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        mask_preview_path=None,
        num_workers=2,
    )

    assert captured["image_path"] == Path("slide-1.svs")
    assert captured["sample_id"] == "slide-1"
    assert captured["tissue_mask_path"] == Path("slide-1-mask.png")
    assert captured["backend"] == tiling_config.backend
    assert captured["spacing_override"] == 0.25
    assert captured["requested_tile_size_px"] == tiling_config.requested_tile_size_px
    assert captured["requested_spacing_um"] == tiling_config.requested_spacing_um
    assert captured["min_tissue_fraction"] == tiling_config.tissue_threshold
    assert captured["overlap"] == tiling_config.overlap
    assert captured["seg_downsample"] == segmentation_config.downsample
    assert captured["tolerance"] == tiling_config.tolerance
    assert captured["ref_tile_size_px"] == filter_config.ref_tile_size
    assert captured["a_t"] == filter_config.a_t
    assert captured["a_h"] == filter_config.a_h
    assert captured["num_workers"] == 2
    assert (
        captured["selection_strategy"]
        == CoordinateSelectionStrategy.MERGED_DEFAULT_TILING
    )
    assert captured["output_mode"] == CoordinateOutputMode.SINGLE_OUTPUT

    assert result.sample_id == "slide-1"
    assert result.image_path == Path("slide-1.svs")
    assert result.mask_path == Path("slide-1-mask.png")
    assert result.backend == "asap"
    assert len(result.x) == 2
    assert result.selection_strategy == CoordinateSelectionStrategy.MERGED_DEFAULT_TILING
    assert result.output_mode == CoordinateOutputMode.SINGLE_OUTPUT
    np.testing.assert_array_equal(
        _coords_array(result), np.array([[100, 200], [300, 400]], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        result.tissue_fractions,
        np.array([0.25, 0.75], dtype=np.float32),
    )


def test_save_tiling_result_writes_preprocessing_npz_and_json(tmp_path: Path):
    result = _build_preprocessing_result(
        sample_id="slide-2",
        image_path="slide-2.svs",
        coords=np.array([[10, 20], [30, 40]], dtype=np.int64),
        tissue_fractions=np.array([0.3, 0.7], dtype=np.float32),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        tile_size_lv0=224,
        overlap=0.0,
        tissue_threshold=0.1,
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
        backend="asap",
        requested_backend="asap",
    )

    tiles = np.load(artifacts.coordinates_npz_path, allow_pickle=False)
    assert set(tiles.files) == {"tile_index", "x", "y", "tissue_fractions"}
    np.testing.assert_array_equal(tiles["tile_index"], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.column_stack((tiles["x"], tiles["y"])),
        np.array([[10, 20], [30, 40]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        tiles["tissue_fractions"],
        np.array([0.3, 0.7], dtype=np.float32),
    )

    meta = json.loads(artifacts.coordinates_meta_path.read_text())
    assert set(meta) == {
        "artifact",
        "filtering",
        "provenance",
        "segmentation",
        "slide",
        "tiling",
    }
    assert meta["provenance"] == {
        "sample_id": "slide-2",
        "image_path": "slide-2.svs",
        "mask_path": None,
        "backend": "asap",
        "requested_backend": "asap",
    }
    assert meta["tiling"]["requested_spacing_um"] == 0.5
    assert meta["tiling"]["requested_tile_size_px"] == 224
    assert meta["tiling"]["read_level"] == 0
    assert meta["tiling"]["read_spacing_um"] == 0.5
    assert meta["tiling"]["read_tile_size_px"] == 224
    assert meta["tiling"]["tile_size_lv0"] == 224
    assert meta["tiling"]["step_px_lv0"] == 224
    assert meta["tiling"]["overlap"] == 0.0
    assert meta["tiling"]["min_tissue_fraction"] == 0.1
    assert meta["tiling"]["n_tiles"] == 2
    assert meta["filtering"]["filter_grayspace"] is False
    assert meta["filtering"]["grayspace_saturation_threshold"] == pytest.approx(0.05)
    assert meta["filtering"]["grayspace_fraction_threshold"] == pytest.approx(0.6)
    assert meta["filtering"]["filter_blur"] is False
    assert meta["filtering"]["blur_threshold"] == pytest.approx(50.0)
    assert meta["filtering"]["qc_spacing_um"] == pytest.approx(2.0)


def test_save_and_load_tiling_result_round_trip(tmp_path: Path):
    result = _build_preprocessing_result(
        sample_id="slide-roundtrip",
        image_path="slide-roundtrip.svs",
        mask_path="slide-roundtrip-mask.png",
        coords=np.array([[10, 20], [30, 40]], dtype=np.int64),
        tissue_fractions=np.array([0.3, 0.7], dtype=np.float32),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        tile_size_lv0=224,
        overlap=0.1,
        tissue_threshold=0.2,
        step_px_lv0=202,
    )

    artifacts = save_tiling_result(result, output_dir=tmp_path)
    loaded = load_tiling_result(artifacts.coordinates_npz_path, artifacts.coordinates_meta_path)

    assert isinstance(loaded, preprocessing_mod.TilingResult)
    assert loaded.sample_id == result.sample_id
    assert loaded.image_path == result.image_path
    assert loaded.mask_path == result.mask_path
    assert loaded.backend == result.backend
    assert loaded.read_level == result.read_level
    assert loaded.read_spacing_um == result.read_spacing_um
    assert resolve_read_step_px(loaded) == resolve_read_step_px(result)
    assert loaded.read_tile_size_px == result.read_tile_size_px
    assert loaded.step_px_lv0 == result.step_px_lv0
    assert loaded.tile_size_lv0 == result.tile_size_lv0
    assert loaded.overlap == result.overlap
    assert loaded.min_tissue_fraction == result.min_tissue_fraction
    assert len(loaded.x) == len(result.x)
    np.testing.assert_array_equal(loaded.x, result.x)
    np.testing.assert_array_equal(loaded.y, result.y)
    np.testing.assert_array_equal(loaded.tile_index, result.tile_index)
    np.testing.assert_array_equal(loaded.tissue_fractions, result.tissue_fractions)


def test_preprocessing_result_builder_preserves_core_fields():
    result = _build_preprocessing_result(
        sample_id="slide-adapter",
        image_path="slide-adapter.svs",
        mask_path="slide-adapter-mask.png",
        coords=np.array([[10, 20], [30, 40]], dtype=np.int64),
        tissue_fractions=np.array([0.3, 0.7], dtype=np.float32),
        backend="openslide",
        requested_backend="openslide",
        read_level=1,
        read_spacing_um=1.0,
        read_tile_size_px=448,
        tile_size_lv0=448,
        overlap=0.1,
        tissue_threshold=0.2,
        step_px_lv0=806,
        annotation="tumor",
        selection_strategy="per_annotation",
        output_mode="multi_output",
    )

    assert isinstance(result, preprocessing_mod.TilingResult)
    assert result.sample_id == "slide-adapter"
    assert result.image_path == Path("slide-adapter.svs")
    assert result.mask_path == Path("slide-adapter-mask.png")
    assert result.backend == "openslide"
    assert result.requested_spacing_um == 0.5
    assert result.requested_tile_size_px == 224
    assert result.read_level == 1
    assert result.read_spacing_um == 1.0
    assert result.read_tile_size_px == 448
    assert resolve_read_step_px(result) == 403
    assert result.step_px_lv0 == 806
    assert result.tile_size_lv0 == 448
    assert result.overlap == 0.1
    assert result.min_tissue_fraction == 0.2
    assert len(result.x) == 2
    assert result.annotation == "tumor"
    assert result.selection_strategy == "per_annotation"
    assert result.output_mode == "multi_output"
    np.testing.assert_array_equal(
        _coords_array(result), np.array([[10, 20], [30, 40]], dtype=np.int64)
    )
    np.testing.assert_array_equal(result.tile_index, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        result.tissue_fractions,
        np.array([0.3, 0.7], dtype=np.float32),
    )


def test_load_tiling_result_accepts_preprocessing_artifact(tmp_path: Path):
    preprocessing_result = preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            x=np.array([10, 30], dtype=np.int64),
            y=np.array([20, 40], dtype=np.int64),
            tissue_fractions=np.array([0.25, 0.75], dtype=np.float32),
            tile_index=np.array([0, 1], dtype=np.int32),
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            read_level=1,
            read_tile_size_px=448,
            read_spacing_um=1.0,
            tile_size_lv0=448,
            is_within_tolerance=False,
            base_spacing_um=0.25,
            slide_dimensions=[1000, 1200],
            level_downsamples=[1.0, 4.0],
            overlap=0.1,
            min_tissue_fraction=0.2,
        ),
        sample_id="slide-preprocessing",
        image_path="slide-preprocessing.svs",
        backend="openslide",
        requested_backend="openslide",
        tolerance=0.07,
        step_px_lv0=806,
        tissue_method="hsv",
        seg_downsample=64,
        seg_level=2,
        seg_spacing_um=16.0,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=224,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
        mask_path="slide-preprocessing-mask.png",
        annotation="tumor",
        selection_strategy="per_annotation",
        output_mode="multi_output",
    )

    artifacts = preprocessing_mod._save_tiling_result(
        preprocessing_result,
        output_dir=tmp_path / "tiles",
    )

    loaded = load_tiling_result(artifacts["npz"], artifacts["meta"])

    assert isinstance(loaded, preprocessing_mod.TilingResult)
    assert loaded.sample_id == "slide-preprocessing"
    assert loaded.image_path == Path("slide-preprocessing.svs")
    assert loaded.mask_path == Path("slide-preprocessing-mask.png")
    assert loaded.backend == "openslide"
    assert loaded.requested_spacing_um == pytest.approx(0.5)
    assert loaded.requested_tile_size_px == 224
    assert loaded.read_level == 1
    assert loaded.read_spacing_um == pytest.approx(1.0)
    assert loaded.read_tile_size_px == 448
    assert resolve_read_step_px(loaded) == 403
    assert loaded.step_px_lv0 == 806
    assert loaded.tile_size_lv0 == 448
    assert loaded.overlap == pytest.approx(0.1)
    assert loaded.min_tissue_fraction == pytest.approx(0.2)
    assert len(loaded.x) == 2
    assert loaded.annotation == "tumor"
    assert loaded.selection_strategy == "per_annotation"
    assert loaded.output_mode == "multi_output"
    np.testing.assert_array_equal(
        _coords_array(loaded), np.array([[10, 20], [30, 40]], dtype=np.int64)
    )
    np.testing.assert_array_equal(loaded.tile_index, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        loaded.tissue_fractions,
        np.array([0.25, 0.75], dtype=np.float32),
    )


def test_write_tiling_preview_writes_expected_preview(monkeypatch, tmp_path: Path):
    result = _build_preprocessing_result(
        sample_id="slide-preview",
        image_path="slide-preview.svs",
        coords=np.array([[10, 20], [30, 40]], dtype=np.int64),
        tissue_fractions=np.array([0.0, 0.0], dtype=np.float32),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=224,
        tile_size_lv0=224,
        overlap=0.0,
        tissue_threshold=0.1,
        step_px_lv0=224,
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
        preview_downsample=32,
        mask_overlay_color=(157, 219, 129),
        mask_overlay_alpha=0.5,
        num_workers,
    ):
        del (
            tiling,
            segmentation,
            filtering,
            mask_preview_path,
            preview_downsample,
            mask_overlay_color,
            mask_overlay_alpha,
            num_workers,
        )
        events.append(f"compute:{whole_slide.sample_id}")
        return _build_result(
            sample_id=whole_slide.sample_id,
            image_path=str(whole_slide.image_path),
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
            num_tiles=len(result.x),
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
        return _build_preprocessing_result(
            sample_id="slide-1",
            image_path="slide-1.svs",
            coords=np.array([[10, 20]], dtype=np.int64),
            tissue_fractions=np.array([0.0], dtype=np.float32),
            backend="cucim",
            requested_backend="cucim",
            read_level=0,
            read_spacing_um=0.5,
            read_tile_size_px=224,
            tile_size_lv0=224,
            overlap=0.0,
            tissue_threshold=0.1,
            step_px_lv0=224,
        )

    def _fake_extract_tiles_to_tar(
        result,
        output_dir,
        *,
        filter_params=None,
        num_workers=1,
        jpeg_backend="turbojpeg",
        **kwargs,
    ):
        del output_dir, filter_params, kwargs
        seen["num_workers"] = num_workers
        seen["jpeg_backend"] = jpeg_backend
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
            num_tiles=len(result.x),
        )

    request = api_mod._ComputeRequest(
        input_index=0,
        whole_slide=SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs")),
        tiling=TilingConfig(
            backend="cucim",
            requested_spacing_um=0.5,
            requested_tile_size_px=224,
            tolerance=0.07,
            overlap=0.0,
            tissue_threshold=0.1,
        ),
        segmentation=SegmentationConfig(64, 8, 255, 7, 4, False, True),
        filtering=FilterConfig(224, 4, 2, False, False, 220, 25, 0.9),
        mask_preview_path=None,
        output_dir=tmp_path,
        num_workers=6,
        jpeg_backend="pil",
        save_tiles=True,
    )

    monkeypatch.setattr(api_mod, "_compute_tiling_result", _fake_compute_tiling_result)
    monkeypatch.setattr(api_mod, "extract_tiles_to_tar", _fake_extract_tiles_to_tar)
    monkeypatch.setattr(api_mod, "save_tiling_result", _fake_save_tiling_result)
    response = api_mod._compute_and_save_tiling_artifacts_from_request(request)

    assert response.ok
    assert seen["num_workers"] == 6
    assert seen["jpeg_backend"] == "pil"


def test_tile_slides_defaults_gpu_decode_to_disabled_for_saved_tiles(
    monkeypatch,
    tmp_path: Path,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    seen = {}

    def _fake_compute_and_save(request):
        seen["gpu_decode"] = request.gpu_decode
        coordinates_npz_path = tmp_path / "coords.npz"
        coordinates_meta_path = tmp_path / "coords.meta.json"
        coordinates_npz_path.write_bytes(b"npz")
        coordinates_meta_path.write_text("{}")
        artifact = TilingArtifacts(
            sample_id=request.whole_slide.sample_id,
            coordinates_npz_path=coordinates_npz_path,
            coordinates_meta_path=coordinates_meta_path,
            num_tiles=1,
            tiles_tar_path=tmp_path / "tiles" / "slide-1.tiles.tar",
        )
        return api_mod._ComputeResponse(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=True,
            artifact=artifact,
        )

    monkeypatch.setattr(
        api_mod,
        "resolve_backend",
        lambda *args, **kwargs: SimpleNamespace(
            backend="cucim",
            requested_backend="cucim",
            reason=None,
        ),
    )
    monkeypatch.setattr(
        api_mod,
        "_compute_and_save_tiling_artifacts_from_request",
        _fake_compute_and_save,
    )

    artifacts = tile_slides(
        [SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs"))],
        tiling=TilingConfig(
            backend="cucim",
            requested_spacing_um=0.5,
            requested_tile_size_px=224,
            tolerance=0.07,
            overlap=0.1,
            tissue_threshold=0.2,
        ),
        segmentation=segmentation_config,
        filtering=filter_config,
        output_dir=tmp_path / "run",
        save_tiles=True,
    )

    assert len(artifacts) == 1
    assert seen["gpu_decode"] is False


def test_save_tiling_result_rejects_invalid_tile_index(tmp_path: Path):
    with pytest.raises(ValueError, match="tile_index must be a 1D array aligned"):
        preprocessing_mod.TileGeometry(
            x=np.array([10], dtype=np.int64),
            y=np.array([20], dtype=np.int64),
            tissue_fractions=np.array([0.0], dtype=np.float32),
            tile_index=np.array([3, 4], dtype=np.int32),
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            read_level=0,
            read_tile_size_px=224,
            read_spacing_um=0.5,
            tile_size_lv0=224,
            is_within_tolerance=True,
            base_spacing_um=0.5,
            slide_dimensions=[224, 224],
            level_downsamples=[1.0],
            overlap=0.0,
            min_tissue_fraction=0.1,
        )


def test_save_tiling_result_rejects_non_vector_arrays(tmp_path: Path):
    with pytest.raises(ValueError, match="x and y must be 1D arrays"):
        preprocessing_mod.TileGeometry(
            x=np.array([[10, 11]], dtype=np.int64),
            y=np.array([20], dtype=np.int64),
            tissue_fractions=np.array([0.0], dtype=np.float32),
            tile_index=np.array([0], dtype=np.int32),
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            read_level=0,
            read_tile_size_px=224,
            read_spacing_um=0.5,
            tile_size_lv0=224,
            is_within_tolerance=True,
            base_spacing_um=0.5,
            slide_dimensions=[224, 224],
            level_downsamples=[1.0],
            overlap=0.0,
            min_tissue_fraction=0.1,
        )


def test_save_tiling_result_cleans_up_partial_outputs_when_metadata_write_fails(
    monkeypatch, tmp_path: Path
):
    result = _build_result(sample_id="slide-clean", image_path="slide-clean.svs")
    tiles_dir = tmp_path / "tiles"

    def _raise_json(*args, **kwargs):
        raise RuntimeError("json failure")

    monkeypatch.setattr("hs2p.preprocessing.json.dumps", _raise_json)

    with pytest.raises(RuntimeError, match="json failure"):
        save_tiling_result(result, output_dir=tmp_path)

    assert not (tiles_dir / "slide-clean.coordinates.npz").exists()
    assert not (tiles_dir / "slide-clean.coordinates.meta.json").exists()
    assert list(tiles_dir.glob("*")) == []


def test_tile_slide_surfaces_preprocessing_errors(
    monkeypatch, tiling_config, segmentation_config, filter_config
):
    _patch_preprocess_slide(
        monkeypatch,
        error=ValueError("bad preprocess result"),
    )

    with pytest.raises(ValueError, match="bad preprocess result"):
        tile_slide(
            SlideSpec(sample_id="slide-bad-tissue", image_path=Path("slide.svs")),
            tiling=tiling_config,
            segmentation=segmentation_config,
            filtering=filter_config,
        )


def test_validate_tiling_artifacts_rejects_mismatched_tiling_config(
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    result = _build_result(sample_id="slide-3", image_path="slide-3.svs")
    artifacts = save_tiling_result(result, output_dir=tmp_path)

    incompatible_tiling = TilingConfig(
        requested_spacing_um=0.75,
        requested_tile_size_px=tiling_config.requested_tile_size_px,
        tolerance=tiling_config.tolerance,
        overlap=tiling_config.overlap,
        tissue_threshold=tiling_config.tissue_threshold,
        backend=tiling_config.backend,
    )

    with pytest.raises(ValueError, match="requested_spacing_um mismatch"):
        validate_tiling_artifacts(
            whole_slide=SlideSpec(sample_id="slide-3", image_path=Path("slide-3.svs")),
            coordinates_npz_path=artifacts.coordinates_npz_path,
            coordinates_meta_path=artifacts.coordinates_meta_path,
            compatibility=_artifact_compatibility(
                tiling_config=incompatible_tiling,
                segmentation_config=segmentation_config,
                filter_config=filter_config,
            ),
        )


def test_validate_tiling_artifacts_ignores_disabled_filter_threshold_mismatches(
    tmp_path: Path,
):
    result = _build_result(sample_id="slide-filter", image_path="slide-filter.svs")
    artifacts = save_tiling_result(result, output_dir=tmp_path)
    matching_tiling = TilingConfig(
        requested_spacing_um=result.requested_spacing_um,
        requested_tile_size_px=result.requested_tile_size_px,
        tolerance=result.tolerance,
        overlap=result.overlap,
        tissue_threshold=result.min_tissue_fraction,
        backend=result.backend,
    )
    matching_segmentation = SegmentationConfig(
        method=result.tissue_method,
        downsample=result.seg_downsample,
        sthresh=result.seg_sthresh,
        sthresh_up=result.seg_sthresh_up,
        mthresh=result.seg_mthresh,
        close=result.seg_close,
    )

    compatibility_filter = FilterConfig(
        ref_tile_size=result.ref_tile_size_px,
        a_t=result.a_t,
        a_h=result.a_h,
        filter_white=False,
        filter_black=False,
        white_threshold=111,
        black_threshold=9,
        fraction_threshold=0.25,
        filter_grayspace=False,
        grayspace_saturation_threshold=0.99,
        grayspace_fraction_threshold=0.12,
        filter_blur=False,
        blur_threshold=5.0,
        qc_spacing_um=9.0,
    )

    validated = validate_tiling_artifacts(
        whole_slide=SlideSpec(
            sample_id="slide-filter",
            image_path=Path("slide-filter.svs"),
        ),
        coordinates_npz_path=artifacts.coordinates_npz_path,
        coordinates_meta_path=artifacts.coordinates_meta_path,
        compatibility=_artifact_compatibility(
            tiling_config=matching_tiling,
            segmentation_config=matching_segmentation,
            filter_config=compatibility_filter,
        ),
    )

    assert validated.coordinates_meta_path == artifacts.coordinates_meta_path


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
                compatibility=_artifact_compatibility(
                    tiling_config=TilingConfig(0.5, 224, 0.07, 0.0, 0.1, "asap"),
                    segmentation_config=SegmentationConfig(64, 8, 255, 7, 4, False, True),
                    filter_config=FilterConfig(224, 4, 2, False, False, 220, 25, 0.9),
                ),
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
                compatibility=_artifact_compatibility(
                    tiling_config=TilingConfig(0.5, 224, 0.07, 0.0, 0.1, "asap"),
                    segmentation_config=SegmentationConfig(64, 8, 255, 7, 4, False, True),
                    filter_config=FilterConfig(224, 4, 2, False, False, 220, 25, 0.9),
                ),
            )


def test_tile_slides_writes_process_list_and_can_reuse_precomputed_tiles(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    _patch_preprocess_slide(
        monkeypatch,
        result=_build_preprocessing_result(
            sample_id="slide-1",
            image_path="slide-1.svs",
            mask_path=None,
        ),
    )

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

    def _unexpected_preprocess(**kwargs):
        raise AssertionError(
            "tile extraction should not run when precomputed tiles are reused"
        )

    monkeypatch.setattr(api_mod, "preprocess_slide", _unexpected_preprocess)

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
        "requested_backend",
        "backend",
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
    _patch_preprocess_slide(
        monkeypatch,
        result=preprocessing_mod.TilingResult(
            tiles=preprocessing_mod.TileGeometry(
                x=np.empty(0, dtype=np.int64),
                y=np.empty(0, dtype=np.int64),
                tissue_fractions=np.empty(0, dtype=np.float32),
                tile_index=np.empty(0, dtype=np.int32),
                requested_tile_size_px=224,
                requested_spacing_um=0.5,
                read_level=0,
                read_tile_size_px=224,
                read_spacing_um=0.5,
                tile_size_lv0=224,
                is_within_tolerance=True,
                base_spacing_um=0.5,
                slide_dimensions=[1000, 1000],
                level_downsamples=[1.0],
                overlap=0.1,
                min_tissue_fraction=0.2,
                tissue_mask=np.zeros((8, 8), dtype=np.uint8),
            ),
            sample_id="slide-0",
            image_path="slide-0.svs",
            backend="asap",
            requested_backend="asap",
            tolerance=0.07,
            step_px_lv0=224,
            tissue_method="hsv",
            seg_downsample=64,
            seg_level=0,
            seg_spacing_um=0.5,
            seg_sthresh=8,
            seg_sthresh_up=255,
            seg_mthresh=7,
            seg_close=4,
            ref_tile_size_px=224,
            a_t=4,
            a_h=2,
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
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
    _patch_preprocess_slide(
        monkeypatch,
        result=_build_preprocessing_result(
            sample_id="slide-preview",
            image_path="slide-preview.svs",
            mask_path=None,
        ),
    )

    def _fake_write_coordinate_preview(**kwargs):
        save_dir = Path(kwargs["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        sample_id = kwargs.get("sample_id", "preview")
        (save_dir / f"{sample_id}.jpg").write_bytes(b"preview")

    def _fake_save_overlay_preview(**kwargs):
        preview_path = Path(kwargs["mask_preview_path"])
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_bytes(b"preview")

    monkeypatch.setattr("hs2p.api.write_coordinate_preview", _fake_write_coordinate_preview)
    monkeypatch.setattr("hs2p.api.save_overlay_preview", _fake_save_overlay_preview)

    expected_mask_path = tmp_path / "preview" / "mask" / "slide-preview.jpg"

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


def test_tile_slides_mask_preview_uses_overlay_renderer_with_preview_style(
    monkeypatch,
    tmp_path: Path,
    tiling_config: TilingConfig,
    segmentation_config: SegmentationConfig,
    filter_config: FilterConfig,
):
    captured: dict[str, object] = {}
    result = _build_preprocessing_result(
        sample_id="slide-preview",
        image_path="slide-preview.svs",
    )
    result.tiles = replace(
        result.tiles,
        tissue_mask=np.array([[0, 255], [255, 0]], dtype=np.uint8),
    )

    _patch_preprocess_slide(
        monkeypatch,
        result=result,
    )

    def _fake_save_overlay_preview(**kwargs):
        captured.update(kwargs)
        preview_path = Path(kwargs["mask_preview_path"])
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_bytes(b"preview")

    monkeypatch.setattr(api_mod, "save_overlay_preview", _fake_save_overlay_preview)

    artifacts = tile_slides(
        [SlideSpec(sample_id="slide-preview", image_path=Path("slide-preview.svs"))],
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        preview=PreviewConfig(
            save_mask_preview=True,
            downsample=16,
            mask_overlay_color=(10, 20, 30),
            mask_overlay_alpha=0.35,
        ),
        output_dir=tmp_path,
    )

    assert len(artifacts) == 1
    assert artifacts[0].mask_preview_path == (
        tmp_path / "preview" / "mask" / "slide-preview.jpg"
    )
    assert artifacts[0].mask_preview_path.is_file()
    assert captured["wsi_path"] == Path("slide-preview.svs")
    assert captured["backend"] == "asap"
    np.testing.assert_array_equal(
        captured["mask_arr"],
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
    )
    assert captured["downsample"] == 16
    assert captured["tile_size_lv0"] == result.tile_size_lv0
    assert captured["pixel_mapping"] == {"background": 0, "tissue": 1}
    assert captured["color_mapping"] == {
        "background": None,
        "tissue": [10, 20, 30],
    }
    assert captured["alpha"] == pytest.approx(0.35)


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
    )
    artifacts = save_tiling_result(result, output_dir=tmp_path / "run")
    pd.DataFrame(
        [
            {
                "sample_id": "slide-6",
                "image_path": "stored-slide.svs",
                "mask_path": np.nan,
                "requested_backend": "asap",
                "backend": "asap",
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
        api_mod,
        "preprocess_slide",
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
    _patch_preprocess_slide(monkeypatch, error=RuntimeError("boom"))

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
                "requested_backend": "asap",
                "backend": "asap",
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


def test_load_csv_rejects_legacy_mask_columns(tmp_path: Path):
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text(
        "sample_id,image_path,tissue_mask_path\n"
        "slide-1,slide-1.svs,slide-1-mask.png\n"
    )
    cfg = SimpleNamespace(csv=str(csv_path))

    with pytest.raises(ValueError, match="deprecated mask columns"):
        load_csv(cfg)


def test_load_tiling_result_rejects_missing_npz_keys(tmp_path: Path):
    npz_path, meta_path = _save_valid_preprocessing_artifact(
        tmp_path,
        sample_id="broken",
    )
    np.savez(
        npz_path,
        tile_index=np.array([0], dtype=np.int32),
        tissue_fractions=np.array([0.5], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="missing x"):
        load_tiling_result(npz_path, meta_path)


def test_load_tiling_result_wraps_corrupt_npz_errors_with_path(tmp_path: Path):
    npz_path, meta_path = _save_valid_preprocessing_artifact(
        tmp_path,
        sample_id="corrupt",
    )
    npz_path.write_bytes(b"not a valid npz")

    with pytest.raises(
        ValueError, match=r"Unable to load tiling artifacts .*corrupt\.coordinates\.npz"
    ):
        load_tiling_result(npz_path, meta_path)


def test_load_tiling_result_rejects_legacy_artifacts(tmp_path: Path):
    npz_path = tmp_path / "legacy.coordinates.npz"
    meta_path = tmp_path / "legacy.coordinates.meta.json"
    np.savez(
        npz_path,
        tile_index=np.array([0], dtype=np.int32),
        x=np.array([10], dtype=np.int64),
        y=np.array([20], dtype=np.int64),
        tissue_fraction=np.array([0.5], dtype=np.float32),
    )
    meta_path.write_text(
        json.dumps(
            {
                "sample_id": "legacy",
                "image_path": "legacy.svs",
                "mask_path": None,
                "backend": "asap",
                "requested_spacing_um": 0.5,
                "requested_tile_size_px": 224,
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

    with pytest.raises(ValueError, match="Invalid tiling metadata"):
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


def test_load_whole_slides_from_rows_rejects_legacy_mask_columns():
    rows = [
        {
            "sample_id": "slide-1",
            "image_path": "slide-1.svs",
            "tissue_mask_path": "slide-1-mask.png",
        }
    ]

    with pytest.raises(ValueError, match="deprecated mask columns"):
        load_whole_slides_from_rows(rows)


def test_coordinate_extraction_result_is_not_tuple_iterable():
    result = CoordinateExtractionResult(
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


def test_coordinate_extraction_result_preserves_x_and_y_arrays():
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

    np.testing.assert_array_equal(result.x, np.array([10, 30], dtype=np.int64))
    np.testing.assert_array_equal(result.y, np.array([20, 40], dtype=np.int64))


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
        "hs2p.artifacts.tempfile.NamedTemporaryFile",
        _tracking_named_temporary_file,
    )
    monkeypatch.setattr("hs2p.artifacts.pd.DataFrame.to_csv", _raise_to_csv)

    with pytest.raises(OSError, match="disk full"):
        tile_slides(
            [],
            tiling=TilingConfig(
                backend="asap",
                requested_spacing_um=0.5,
                requested_tile_size_px=224,
                tolerance=0.07,
                overlap=0.1,
                tissue_threshold=0.2,
            ),
            segmentation=SegmentationConfig(64, 8, 255, 7, 4, False, True),
            filtering=FilterConfig(224, 4, 2, False, False, 220, 25, 0.9),
            output_dir=tmp_path,
        )

    assert created_temp_paths
    assert all(not path.exists() for path in created_temp_paths)


def test_config_dataclasses_apply_package_defaults_for_secondary_parameters():
    tiling = TilingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
    )
    segmentation = SegmentationConfig(downsample=64)
    filtering = FilterConfig(ref_tile_size=224, a_t=4, a_h=2)
    preview = PreviewConfig()

    assert not hasattr(tiling, "drop_holes")
    assert tiling.backend == "auto"
    assert tiling.requested_backend == "auto"
    assert segmentation.sthresh == default_config.tiling.seg_params.sthresh
    assert segmentation.sthresh_up == default_config.tiling.seg_params.sthresh_up
    assert segmentation.mthresh == default_config.tiling.seg_params.mthresh
    assert segmentation.close == default_config.tiling.seg_params.close
    assert segmentation.method == default_config.tiling.seg_params.method
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
    assert (
        filtering.filter_grayspace
        == default_config.tiling.filter_params.filter_grayspace
    )
    assert filtering.grayspace_saturation_threshold == pytest.approx(
        default_config.tiling.filter_params.grayspace_saturation_threshold
    )
    assert filtering.grayspace_fraction_threshold == pytest.approx(
        default_config.tiling.filter_params.grayspace_fraction_threshold
    )
    assert filtering.filter_blur == default_config.tiling.filter_params.filter_blur
    assert filtering.blur_threshold == pytest.approx(
        default_config.tiling.filter_params.blur_threshold
    )
    assert filtering.qc_spacing_um == pytest.approx(
        default_config.tiling.filter_params.qc_spacing_um
    )
    assert preview.downsample == default_config.tiling.preview.downsample
    assert preview.mask_overlay_color == tuple(
        default_config.tiling.preview.mask_overlay_color
    )
    assert preview.mask_overlay_alpha == pytest.approx(
        default_config.tiling.preview.mask_overlay_alpha
    )


def test_preview_config_rejects_invalid_mask_overlay_alpha():
    with pytest.raises(ValueError, match="mask_overlay_alpha"):
        PreviewConfig(mask_overlay_alpha=1.5)


def test_resolve_preview_config_reads_mask_overlay_style():
    cfg = OmegaConf.create(
        {
            "tiling": {
                "preview": {
                    "save": True,
                    "downsample": 64,
                    "mask_overlay_color": [1, 2, 3],
                    "mask_overlay_alpha": 0.25,
                }
            },
        }
    )

    preview = resolve_preview_config(cfg)

    assert preview == PreviewConfig(
        save_mask_preview=True,
        save_tiling_preview=True,
        downsample=64,
        mask_overlay_color=(1, 2, 3),
        mask_overlay_alpha=0.25,
    )


def test_hs2p_configs_reexports_runtime_config_models():
    assert ConfigsTilingConfig is TilingConfig
    assert ConfigsSegmentationConfig is SegmentationConfig
    assert ConfigsFilterConfig is FilterConfig
    assert ConfigsPreviewConfig is PreviewConfig


# --- zero-tile npz skip ---


def test_save_tiling_result_zero_tiles_skips_npz(tmp_path: Path):
    result = _build_preprocessing_result(
        sample_id="zero-tile",
        image_path="zero-tile.svs",
        coords=np.empty((0, 2), dtype=np.int64),
        tissue_fractions=np.empty(0, dtype=np.float32),
    )

    artifacts = save_tiling_result(result, output_dir=tmp_path)

    assert artifacts.coordinates_npz_path is None
    assert artifacts.num_tiles == 0
    assert artifacts.coordinates_meta_path.exists()
    meta = json.loads(artifacts.coordinates_meta_path.read_text())
    assert meta["tiling"]["n_tiles"] == 0
    npz_path = tmp_path / "tiles" / "zero-tile.coordinates.npz"
    assert not npz_path.exists()


def test_load_tiling_result_zero_tiles_from_meta_only(tmp_path: Path):
    result = _build_preprocessing_result(
        sample_id="zero-tile-load",
        image_path="zero-tile-load.svs",
        coords=np.empty((0, 2), dtype=np.int64),
        tissue_fractions=np.empty(0, dtype=np.float32),
    )
    artifacts = save_tiling_result(result, output_dir=tmp_path)
    assert artifacts.coordinates_npz_path is None

    loaded = load_tiling_result(None, artifacts.coordinates_meta_path)

    assert loaded.num_tiles == 0
    assert loaded.sample_id == "zero-tile-load"
    assert str(loaded.image_path) == "zero-tile-load.svs"
    np.testing.assert_array_equal(loaded.x, np.empty(0, dtype=np.int64))
    np.testing.assert_array_equal(loaded.y, np.empty(0, dtype=np.int64))


def test_maybe_load_existing_artifacts_zero_tiles_meta_only(tmp_path: Path):
    result = _build_preprocessing_result(
        sample_id="zero-tile-maybe",
        image_path="zero-tile-maybe.svs",
        coords=np.empty((0, 2), dtype=np.int64),
        tissue_fractions=np.empty(0, dtype=np.float32),
    )
    artifacts = save_tiling_result(result, output_dir=tmp_path / "tiles")
    assert artifacts.coordinates_npz_path is None
    tiles_dir = tmp_path / "tiles" / "tiles"

    loaded = maybe_load_existing_artifacts(
        whole_slide=SlideSpec(
            sample_id="zero-tile-maybe",
            image_path=Path("zero-tile-maybe.svs"),
        ),
        read_coordinates_from=tiles_dir,
            compatibility=_artifact_compatibility(
                tiling_config=TilingConfig(0.5, 224, 0.07, 0.1, 0.2, "asap"),
                segmentation_config=SegmentationConfig(
                    method="hsv",
                    downsample=64,
                    sthresh=8,
                    sthresh_up=255,
                    mthresh=7,
                    close=4,
                ),
                filter_config=FilterConfig(224, 4, 2, False, False, 220, 25, 0.9),
            ),
        )

    assert loaded is not None
    assert loaded.num_tiles == 0
    assert loaded.coordinates_npz_path is None
    assert loaded.coordinates_meta_path.exists()


def test_build_success_process_row_zero_tiles_has_nan_npz_path(
    tmp_path: Path,
):
    result = _build_preprocessing_result(
        sample_id="zero-tile-row",
        image_path="zero-tile-row.svs",
        coords=np.empty((0, 2), dtype=np.int64),
        tissue_fractions=np.empty(0, dtype=np.float32),
    )
    artifacts = save_tiling_result(result, output_dir=tmp_path)
    assert artifacts.coordinates_npz_path is None

    whole_slide = SlideSpec(sample_id="zero-tile-row", image_path=Path("zero-tile-row.svs"))

    row = api_mod._build_success_process_row(
        whole_slide=whole_slide,
        artifact=artifacts,
    )

    assert row["requested_backend"] == "asap"
    assert row["backend"] == "asap"
    assert pd.isna(row["coordinates_npz_path"])
    assert row["num_tiles"] == 0


def test_build_failure_process_row_records_backend_provenance():
    row = api_mod._build_failure_process_row(
        whole_slide=SlideSpec(sample_id="failed", image_path=Path("failed.svs")),
        requested_backend="auto",
        backend="openslide",
        error="boom",
        traceback_text="traceback",
    )

    assert row["requested_backend"] == "auto"
    assert row["backend"] == "openslide"
    assert row["tiling_status"] == "failed"
