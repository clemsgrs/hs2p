"""Tests for extract_tiles_to_tar() and the save_tiles pipeline option."""

import csv
import io
import tarfile
import types
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import hs2p.preprocessing as preprocessing_mod
from hs2p.api import extract_tiles_to_tar
from hs2p.configs.models import FilterConfig
from hs2p.wsi import iter_tile_arrays_from_result
from hs2p.wsi.read_plans import GroupedReadPlan, iter_grouped_read_plans


def _make_tiling_result(
    num_tiles: int = 3,
    tile_size: int = 256,
    sample_id: str = "slide-1",
    *,
    step_px: int | None = None,
) -> preprocessing_mod.TilingResult:
    if step_px is None:
        step_px = tile_size
    overlap = 0.0 if step_px == tile_size else 1.0 - (step_px / tile_size)
    x = np.arange(num_tiles, dtype=np.int64) * tile_size
    y = np.zeros(num_tiles, dtype=np.int64)
    return preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            coordinates=np.column_stack([x, y]),
            tissue_fractions=np.zeros(num_tiles, dtype=np.float32),
            tile_index=np.arange(num_tiles, dtype=np.int32),
            requested_tile_size_px=tile_size,
            requested_spacing_um=0.5,
            read_level=0,
            effective_tile_size_px=tile_size,
            effective_spacing_um=0.5,
            tile_size_lv0=tile_size,
            is_within_tolerance=True,
            base_spacing_um=0.5,
            slide_dimensions=[num_tiles * step_px + tile_size, tile_size],
            level_downsamples=[1.0],
            overlap=overlap,
            min_tissue_fraction=0.1,
            use_padding=True,
        ),
        sample_id=sample_id,
        image_path=Path("/data/slide-1.svs"),
        mask_path=None,
        backend="openslide",
        requested_backend="openslide",
        step_px_lv0=step_px,
        tolerance=0.05,
        tissue_method="unknown",
        seg_downsample=64,
        seg_level=0,
        seg_spacing_um=0.0,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=tile_size,
        a_t=4,
        a_h=0,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def _solid_patch(color: tuple[int, int, int], size: int = 256) -> np.ndarray:
    """Return an (size, size, 3) uint8 array filled with *color*."""
    arr = np.empty((size, size, 3), dtype=np.uint8)
    arr[:] = color
    return arr


def _make_grid_tiling_result(
    *,
    columns: int,
    rows: int,
    tile_size: int,
    step_px: int,
) -> preprocessing_mod.TilingResult:
    x_coords: list[int] = []
    y_coords: list[int] = []
    for x_idx in range(columns):
        for y_idx in range(rows):
            x_coords.append(x_idx * step_px)
            y_coords.append(y_idx * step_px)
    overlap = 0.0 if step_px == tile_size else 1.0 - (step_px / tile_size)
    return preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            coordinates=np.column_stack(
                [
                    np.asarray(x_coords, dtype=np.int64),
                    np.asarray(y_coords, dtype=np.int64),
                ]
            ),
            tissue_fractions=np.zeros(columns * rows, dtype=np.float32),
            tile_index=np.arange(columns * rows, dtype=np.int32),
            requested_tile_size_px=tile_size,
            requested_spacing_um=0.5,
            read_level=0,
            effective_tile_size_px=tile_size,
            effective_spacing_um=0.5,
            tile_size_lv0=tile_size,
            is_within_tolerance=True,
            base_spacing_um=0.5,
            slide_dimensions=[columns * step_px + tile_size, rows * step_px + tile_size],
            level_downsamples=[1.0],
            overlap=overlap,
            min_tissue_fraction=0.1,
            use_padding=True,
        ),
        sample_id="grid-slide",
        image_path=Path("/data/grid-slide.svs"),
        mask_path=None,
        backend="openslide",
        requested_backend="openslide",
        step_px_lv0=step_px,
        tolerance=0.05,
        tissue_method="unknown",
        seg_downsample=64,
        seg_level=0,
        seg_spacing_um=0.0,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=tile_size,
        a_t=4,
        a_h=0,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def _make_custom_tiling_result(
    *,
    coords: list[tuple[int, int]],
    tile_size: int,
    sample_id: str = "custom-slide",
) -> preprocessing_mod.TilingResult:
    return preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            coordinates=np.asarray(coords, dtype=np.int64),
            tissue_fractions=np.zeros(len(coords), dtype=np.float32),
            tile_index=np.arange(len(coords), dtype=np.int32),
            requested_tile_size_px=tile_size,
            requested_spacing_um=0.5,
            read_level=0,
            effective_tile_size_px=tile_size,
            effective_spacing_um=0.5,
            tile_size_lv0=tile_size,
            is_within_tolerance=True,
            base_spacing_um=0.5,
            slide_dimensions=[1000, 1000],
            level_downsamples=[1.0],
            overlap=0.0,
            min_tissue_fraction=0.1,
            use_padding=True,
        ),
        sample_id=sample_id,
        image_path=Path("/data/custom-slide.svs"),
        mask_path=None,
        backend="openslide",
        requested_backend="openslide",
        step_px_lv0=tile_size,
        tolerance=0.05,
        tissue_method="unknown",
        seg_downsample=64,
        seg_level=0,
        seg_spacing_um=0.0,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=tile_size,
        a_t=4,
        a_h=0,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def _make_grouped_region(
    *,
    block_size: int,
    tile_size: int,
    step_px: int,
) -> np.ndarray:
    region_size = tile_size + (block_size - 1) * step_px
    region = np.zeros((region_size, region_size, 3), dtype=np.uint8)
    for x_idx in range(block_size):
        for y_idx in range(block_size):
            tile_index = x_idx * block_size + y_idx
            value = tile_index + 1
            y0 = y_idx * step_px
            x0 = x_idx * step_px
            region[y0 : y0 + tile_size, x0 : x0 + tile_size] = value
    return region


def _make_mock_reader(
    *regions: np.ndarray,
    level_dimensions: list[tuple[int, int]] | None = None,
    level_downsamples: list[tuple[float, float]] | None = None,
):
    reader = MagicMock()
    reader.read_region.side_effect = list(regions)
    reader.level_dimensions = level_dimensions or [(4096, 4096)]
    reader.level_downsamples = level_downsamples or [(1.0, 1.0)]
    reader.__enter__.return_value = reader
    reader.__exit__.return_value = None
    return reader


def _monkeypatch_cucim_import(monkeypatch, fake_cucim):
    import hs2p.wsi.cucim_reader as cucim_reader_mod
    original_cucim_reader_import = cucim_reader_mod.importlib.import_module
    monkeypatch.setattr(
        cucim_reader_mod.importlib,
        "import_module",
        lambda name: fake_cucim
        if name == "cucim"
        else original_cucim_reader_import(name),
    )


class TestExtractTilesToTar:
    """Unit tests for extract_tiles_to_tar()."""

    def test_creates_tar_with_correct_number_of_jpegs(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=3)
        colors = [(128, 0, 0), (0, 128, 0), (0, 0, 128)]

        mock_reader = _make_mock_reader(*[_solid_patch(c) for c in colors])

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tar_path, out_result = extract_tiles_to_tar(result, output_dir=tmp_path)

        assert tar_path == tmp_path / "tiles" / "slide-1.tiles.tar"
        assert tar_path.is_file()

        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            assert [m.name for m in members] == [
                "000000.jpg",
                "000001.jpg",
                "000002.jpg",
            ]
            for m in members:
                data = tf.extractfile(m).read()
                img = Image.open(io.BytesIO(data))
                assert img.size == (256, 256)

        # No filtering — result unchanged
        assert out_result is result

        manifest_path = tmp_path / "tiles" / "slide-1.tiles.manifest.csv"
        with manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows == [
            {"tile_index": "0", "x": "0", "y": "0"},
            {"tile_index": "1", "x": "256", "y": "0"},
            {"tile_index": "2", "x": "512", "y": "0"},
        ]

    def test_names_tar_members_from_original_tile_index(self, tmp_path: Path):
        coords = [
            (0, 0),
            (100, 0),
            (100, 16),
            (116, 0),
            (116, 16),
        ]
        result = _make_custom_tiling_result(coords=coords, tile_size=16)

        grouped_region = _make_grouped_region(block_size=2, tile_size=16, step_px=16)
        single_patch = _solid_patch((9, 9, 9), size=16)

        mock_reader = _make_mock_reader(grouped_region, single_patch)

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tar_path, out_result = extract_tiles_to_tar(result, output_dir=tmp_path)

        assert out_result is result

        with tarfile.open(tar_path, "r") as tf:
            assert [member.name for member in tf.getmembers()] == [
                "000001.jpg",
                "000002.jpg",
                "000003.jpg",
                "000004.jpg",
                "000000.jpg",
            ]

        manifest_path = tmp_path / "tiles" / "custom-slide.tiles.manifest.csv"
        with manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows == [
            {"tile_index": "1", "x": "100", "y": "0"},
            {"tile_index": "2", "x": "100", "y": "16"},
            {"tile_index": "3", "x": "116", "y": "0"},
            {"tile_index": "4", "x": "116", "y": "16"},
            {"tile_index": "0", "x": "0", "y": "0"},
        ]

    def test_grouped_read_plans_follow_custom_supertile_sizes(self):
        result = _make_grid_tiling_result(
            columns=4,
            rows=4,
            tile_size=16,
            step_px=16,
        )

        plans = list(
            iter_grouped_read_plans(
                result=result,
                read_step_px=16,
                step_px_lv0=16,
                supertile_sizes=(2,),
            )
        )

        assert len(plans) == 4
        assert all(plan.block_size == 2 for plan in plans)

        plans = list(
            iter_grouped_read_plans(
                result=result,
                read_step_px=16,
                step_px_lv0=16,
                supertile_sizes=(4, 2),
            )
        )

        assert len(plans) == 1
        assert plans[0].block_size == 4

    def test_uses_rgb_420_turbojpeg_encoding(self, tmp_path: Path, monkeypatch):
        result = _make_tiling_result(num_tiles=1)
        mock_reader = _make_mock_reader(_solid_patch((12, 34, 56)))

        captured: dict[str, int] = {}

        class _FakeTurboJPEG:
            def encode(
                self,
                img_array,
                quality=85,
                pixel_format=None,
                jpeg_subsample=None,
                flags=0,
                dst=None,
                lossless=False,
                icc_profile=None,
            ):
                captured["quality"] = quality
                captured["pixel_format"] = pixel_format
                captured["jpeg_subsample"] = jpeg_subsample
                return b"jpeg"

        monkeypatch.setitem(
            sys.modules,
            "turbojpeg",
            types.SimpleNamespace(
                TurboJPEG=lambda: _FakeTurboJPEG(),
                TJPF_RGB=0,
                TJSAMP_420=2,
            ),
        )

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tar_path, _ = extract_tiles_to_tar(result, output_dir=tmp_path)

        assert tar_path.is_file()
        assert captured == {
            "quality": 90,
            "pixel_format": 0,
            "jpeg_subsample": 2,
        }

    def test_uses_pil_encoding_when_requested(self, tmp_path: Path, monkeypatch):
        result = _make_tiling_result(num_tiles=1)
        mock_reader = _make_mock_reader(_solid_patch((12, 34, 56)))

        turbojpeg_called = False

        class _FakeTurboJPEG:
            def __init__(self, *_args, **_kwargs):
                nonlocal turbojpeg_called
                turbojpeg_called = True

            def encode(self, *_args, **_kwargs):
                raise AssertionError("TurboJPEG should not be used for PIL backend")

        monkeypatch.setitem(
            sys.modules,
            "turbojpeg",
            types.SimpleNamespace(
                TurboJPEG=lambda: _FakeTurboJPEG(),
                TJPF_RGB=0,
                TJSAMP_420=2,
            ),
        )

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tar_path, _ = extract_tiles_to_tar(
                result,
                output_dir=tmp_path,
                jpeg_backend="pil",
            )

        assert tar_path.is_file()
        assert turbojpeg_called is False

    def test_white_filtering_drops_white_tile(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=3)
        patches = [
            _solid_patch((128, 0, 0)),   # keep
            _solid_patch((250, 250, 250)),  # all-white → drop
            _solid_patch((0, 0, 128)),   # keep
        ]

        # QC reads all tiles in a single batched window; lay each 256×256 patch
        # at its x-offset within a (256, slide_width, 3) window.
        tile_size = 256
        slide_w = result.slide_dimensions[0]
        qc_window = np.zeros((tile_size, slide_w, 3), dtype=np.uint8)
        for i, p in enumerate(patches):
            qc_window[:, i * tile_size : (i + 1) * tile_size, :] = p
        qc_reader = _make_mock_reader(
            qc_window,
            level_dimensions=[tuple(result.slide_dimensions)],
            level_downsamples=result.level_downsamples,
        )
        export_reader = _make_mock_reader(patches[0], patches[2])

        filter_cfg = FilterConfig(
            filter_white=True,
            white_threshold=220,
            fraction_threshold=0.9,
        )

        with patch(
            "hs2p.wsi.tile_stream.open_slide",
            side_effect=[qc_reader, export_reader],
        ):
            tar_path, filtered = extract_tiles_to_tar(
                result,
                output_dir=tmp_path,
                filter_params=filter_cfg,
                jpeg_backend="pil",
            )

        # Tar should have 2 tiles
        with tarfile.open(tar_path, "r") as tf:
            assert len(tf.getmembers()) == 2

        # Result should reflect filtering
        assert len(filtered.coordinates) == 2
        np.testing.assert_array_equal(filtered.coordinates, [[0, 0], [512, 0]])
        assert len(filtered.tile_index) == 2
        np.testing.assert_array_equal(filtered.tile_index, [0, 1])

    def test_black_filtering_drops_black_tile(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=2)
        patches = [
            _solid_patch((5, 5, 5)),    # all-black → drop
            _solid_patch((128, 128, 0)),  # keep
        ]

        tile_size = 256
        slide_w = result.slide_dimensions[0]
        qc_window = np.zeros((tile_size, slide_w, 3), dtype=np.uint8)
        for i, p in enumerate(patches):
            qc_window[:, i * tile_size : (i + 1) * tile_size, :] = p
        qc_reader = _make_mock_reader(
            qc_window,
            level_dimensions=[tuple(result.slide_dimensions)],
            level_downsamples=result.level_downsamples,
        )
        export_reader = _make_mock_reader(patches[1])

        filter_cfg = FilterConfig(
            filter_black=True,
            black_threshold=25,
            fraction_threshold=0.9,
        )

        with patch(
            "hs2p.wsi.tile_stream.open_slide",
            side_effect=[qc_reader, export_reader],
        ):
            tar_path, filtered = extract_tiles_to_tar(
                result,
                output_dir=tmp_path,
                filter_params=filter_cfg,
                jpeg_backend="pil",
            )

        with tarfile.open(tar_path, "r") as tf:
            assert len(tf.getmembers()) == 1

        assert len(filtered.coordinates) == 1
        np.testing.assert_array_equal(filtered.coordinates, [[256, 0]])

    def test_filtered_preprocessing_results_keep_preprocessing_type(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=2)
        patches = [
            _solid_patch((5, 5, 5)),
            _solid_patch((128, 128, 0)),
        ]

        tile_size = 256
        slide_w = result.slide_dimensions[0]
        qc_window = np.zeros((tile_size, slide_w, 3), dtype=np.uint8)
        for i, p in enumerate(patches):
            qc_window[:, i * tile_size : (i + 1) * tile_size, :] = p
        qc_reader = _make_mock_reader(
            qc_window,
            level_dimensions=[tuple(result.slide_dimensions)],
            level_downsamples=result.level_downsamples,
        )
        export_reader = _make_mock_reader(patches[1])

        filter_cfg = FilterConfig(
            filter_black=True,
            black_threshold=25,
            fraction_threshold=0.9,
        )

        with patch(
            "hs2p.wsi.tile_stream.open_slide",
            side_effect=[qc_reader, export_reader],
        ):
            _, filtered = extract_tiles_to_tar(
                result,
                output_dir=tmp_path,
                filter_params=filter_cfg,
                jpeg_backend="pil",
            )

        assert isinstance(filtered, preprocessing_mod.TilingResult)
        assert len(filtered.coordinates) == 1
        np.testing.assert_array_equal(filtered.coordinates, [[256, 0]])
        np.testing.assert_array_equal(filtered.tile_index, [0])

    def test_all_tiles_filtered_produces_empty_tar(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=1)
        qc_reader = _make_mock_reader(
            _solid_patch((250, 250, 250)),
            level_dimensions=[tuple(result.slide_dimensions)],
            level_downsamples=result.level_downsamples,
        )
        export_reader = _make_mock_reader()

        filter_cfg = FilterConfig(
            filter_white=True,
            white_threshold=220,
            fraction_threshold=0.9,
        )

        with patch(
            "hs2p.wsi.tile_stream.open_slide",
            side_effect=[qc_reader, export_reader],
        ):
            tar_path, filtered = extract_tiles_to_tar(
                result, output_dir=tmp_path, filter_params=filter_cfg
            )

        with tarfile.open(tar_path, "r") as tf:
            assert len(tf.getmembers()) == 0

        assert len(filtered.coordinates) == 0

    def test_resizes_when_read_and_target_sizes_differ(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=1, tile_size=224)
        result = replace(
            result,
            tiles=replace(
                result.tiles,
                effective_tile_size_px=512,
            ),
        )

        mock_reader = _make_mock_reader(_solid_patch((100, 100, 100), size=512))

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tar_path, _ = extract_tiles_to_tar(result, output_dir=tmp_path)

        with tarfile.open(tar_path, "r") as tf:
            data = tf.extractfile(tf.getmembers()[0]).read()
            img = Image.open(io.BytesIO(data))
            assert img.size == (224, 224)

    def test_resizing_uses_bilinear_resampling(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=1, tile_size=224)
        result = replace(
            result,
            tiles=replace(
                result.tiles,
                effective_tile_size_px=512,
            ),
        )

        mock_reader = _make_mock_reader(_solid_patch((100, 100, 100), size=512))

        resize_calls: list[int] = []
        bilinear = Image.Resampling.BILINEAR
        original_fromarray = Image.fromarray

        class _ResizeSpy:
            def __init__(self, image: Image.Image):
                self._image = image

            def convert(self, mode: str):
                self._image = self._image.convert(mode)
                return self

            def resize(self, size, resample=None, box=None, reducing_gap=None):
                resize_calls.append(resample)
                self._image = self._image.resize(
                    size,
                    resample=resample,
                    box=box,
                    reducing_gap=reducing_gap,
                )
                return self

            def __array__(self, dtype=None):
                return np.asarray(self._image, dtype=dtype)

            def save(self, *args, **kwargs):
                return self._image.save(*args, **kwargs)

        def _fromarray_spy(*args, **kwargs):
            return _ResizeSpy(original_fromarray(*args, **kwargs))

        with (
            patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader),
            patch("PIL.Image.fromarray", side_effect=_fromarray_spy),
        ):
            extract_tiles_to_tar(result, output_dir=tmp_path)

        assert resize_calls == [bilinear]

    def test_strips_alpha_channel(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=1)
        rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        rgba[:, :, :3] = 100
        rgba[:, :, 3] = 255

        mock_reader = _make_mock_reader(rgba)

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tar_path, _ = extract_tiles_to_tar(result, output_dir=tmp_path)

        with tarfile.open(tar_path, "r") as tf:
            data = tf.extractfile(tf.getmembers()[0]).read()
            img = Image.open(io.BytesIO(data))
            assert img.mode == "RGB"

    def test_custom_tiles_dir(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=1)
        custom_dir = tmp_path / "custom_output"

        mock_reader = _make_mock_reader(_solid_patch((50, 50, 50)))

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tar_path, _ = extract_tiles_to_tar(
                result, output_dir=tmp_path, tiles_dir=custom_dir
            )

        assert tar_path == custom_dir / "slide-1.tiles.tar"
        assert tar_path.is_file()

    def test_no_filter_params_keeps_all_tiles(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=2)
        # Even white tiles should be kept when no filter_params
        mock_reader = _make_mock_reader(
            _solid_patch((255, 255, 255)),
            _solid_patch((0, 0, 0)),
        )

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            _, out_result = extract_tiles_to_tar(
                result, output_dir=tmp_path, filter_params=None
            )

        assert out_result is result  # unchanged

    def test_cucim_backend_uses_batched_read_region(self, monkeypatch, tmp_path: Path):
        result = _make_tiling_result(num_tiles=2)
        result = replace(
            result,
            backend="cucim",
            requested_backend="cucim",
            tiles=replace(
                result.tiles,
                read_level=3,
                effective_tile_size_px=128,
            ),
        )

        regions = [_solid_patch((10, 20, 30), size=128), _solid_patch((40, 50, 60), size=128)]
        mock_cu_image = MagicMock()
        mock_cu_image.read_region.return_value = iter(regions)
        fake_cucim = types.SimpleNamespace(CuImage=MagicMock(return_value=mock_cu_image))

        _monkeypatch_cucim_import(monkeypatch, fake_cucim)

        with patch("hs2p.wsi.tile_stream.open_slide") as mock_open_slide:
            tar_path, out_result = extract_tiles_to_tar(
                result,
                output_dir=tmp_path,
                num_workers=5,
                gpu_decode=False,
                jpeg_backend="pil",
            )

        assert tar_path.is_file()
        assert out_result is result
        fake_cucim.CuImage.assert_called_once_with(str(result.image_path))
        mock_cu_image.read_region.assert_called_once_with(
            [(0, 0), (256, 0)],
            (128, 128),
            level=3,
            num_workers=5,
        )
        mock_open_slide.assert_not_called()

    def test_cucim_backend_defaults_gpu_decode_to_disabled(
        self, monkeypatch, tmp_path: Path
    ):
        result = _make_tiling_result(num_tiles=1)
        result = replace(result, backend="cucim", requested_backend="cucim")

        mock_cu_image = MagicMock()
        mock_cu_image.read_region.return_value = iter([_solid_patch((10, 20, 30))])
        fake_cucim = types.SimpleNamespace(CuImage=MagicMock(return_value=mock_cu_image))

        _monkeypatch_cucim_import(monkeypatch, fake_cucim)

        with patch("hs2p.wsi.tile_stream.open_slide") as mock_open_slide:
            tar_path, out_result = extract_tiles_to_tar(
                result,
                output_dir=tmp_path,
                num_workers=2,
                jpeg_backend="pil",
            )

        assert tar_path.is_file()
        assert out_result is result
        fake_cucim.CuImage.assert_called_once_with(str(result.image_path))
        mock_cu_image.read_region.assert_called_once_with(
            [(0, 0)],
            (256, 256),
            level=0,
            num_workers=2,
        )
        mock_open_slide.assert_not_called()

    def test_cucim_backend_raises_when_cucim_is_unavailable(
        self, monkeypatch, tmp_path: Path
    ):
        result = _make_tiling_result(num_tiles=1)
        result = replace(result, backend="cucim", requested_backend="cucim")

        import hs2p.wsi.cucim_reader as cucim_reader_mod

        def _import_module(name):
            if name == "cucim":
                raise ModuleNotFoundError("No module named 'cucim'")
            raise AssertionError(f"unexpected module import: {name}")

        monkeypatch.setattr(cucim_reader_mod.importlib, "import_module", _import_module)
        with patch(
            "hs2p.wsi.tile_stream.open_slide",
        ) as mock_open_slide:
            with pytest.raises(ModuleNotFoundError, match="cucim"):
                extract_tiles_to_tar(
                    result,
                    output_dir=tmp_path,
                    num_workers=4,
                )

        mock_open_slide.assert_not_called()

    def test_cucim_iterator_groups_dense_4x4_grid_into_one_batched_read(
        self, monkeypatch
    ):
        result = _make_grid_tiling_result(columns=4, rows=4, tile_size=16, step_px=16)
        result = replace(result, backend="cucim", requested_backend="cucim")
        grouped_region = _make_grouped_region(block_size=4, tile_size=16, step_px=16)

        mock_cu_image = MagicMock()
        mock_cu_image.read_region.return_value = iter([grouped_region])
        fake_cucim = types.SimpleNamespace(CuImage=MagicMock(return_value=mock_cu_image))

        _monkeypatch_cucim_import(monkeypatch, fake_cucim)

        tiles = list(
            iter_tile_arrays_from_result(
                result=result,
                num_workers=7,
                gpu_decode=False,
            )
        )

        assert len(tiles) == 16
        fake_cucim.CuImage.assert_called_once_with(str(result.image_path))
        mock_cu_image.read_region.assert_called_once_with(
            [(0, 0)],
            (64, 64),
            level=0,
            num_workers=7,
        )
        assert int(tiles[0][0, 0, 0]) == 1
        assert int(tiles[1][0, 0, 0]) == 2
        assert int(tiles[4][0, 0, 0]) == 5
        assert int(tiles[-1][0, 0, 0]) == 16

    def test_cucim_iterator_batches_multiple_2x2_plans_in_one_read_call(
        self, monkeypatch
    ):
        result = _make_custom_tiling_result(
            coords=[
                (0, 0),
                (0, 16),
                (16, 0),
                (16, 16),
                (100, 0),
                (100, 16),
                (116, 0),
                (116, 16),
                (300, 0),
            ],
            tile_size=16,
        )
        result = replace(result, backend="cucim", requested_backend="cucim")

        grouped_region_a = _make_grouped_region(block_size=2, tile_size=16, step_px=16)
        grouped_region_b = _make_grouped_region(block_size=2, tile_size=16, step_px=16)
        single_region = _solid_patch((99, 99, 99), size=16)

        mock_cu_image = MagicMock()
        mock_cu_image.read_region.side_effect = [
            iter([grouped_region_a, grouped_region_b]),
            iter([single_region]),
        ]
        fake_cucim = types.SimpleNamespace(CuImage=MagicMock(return_value=mock_cu_image))

        _monkeypatch_cucim_import(monkeypatch, fake_cucim)

        tiles = list(
            iter_tile_arrays_from_result(
                result=result,
                num_workers=3,
                gpu_decode=False,
            )
        )

        assert len(tiles) == 9
        assert mock_cu_image.read_region.call_count == 2
        assert mock_cu_image.read_region.call_args_list[0].kwargs == {
            "level": 0,
            "num_workers": 3,
        }
        assert mock_cu_image.read_region.call_args_list[0].args == (
            [(0, 0), (100, 0)],
            (32, 32),
        )
        assert mock_cu_image.read_region.call_args_list[1].args == (
            [(300, 0)],
            (16, 16),
        )

    def test_cucim_iterator_groups_same_size_plans_even_when_interleaved(
        self, monkeypatch
    ):
        result = _make_tiling_result(num_tiles=1, tile_size=16)
        result = replace(result, backend="cucim", requested_backend="cucim")

        import hs2p.wsi.tile_stream as tile_stream_mod

        interleaved_plans = [
            GroupedReadPlan(
                x=0,
                y=0,
                read_size_px=32,
                block_size=2,
                tile_indices=(0, 1, 2, 3),
            ),
            GroupedReadPlan(
                x=200,
                y=0,
                read_size_px=16,
                block_size=1,
                tile_indices=(4,),
            ),
            GroupedReadPlan(
                x=100,
                y=0,
                read_size_px=32,
                block_size=2,
                tile_indices=(5, 6, 7, 8),
            ),
        ]

        grouped_region_a = _make_grouped_region(block_size=2, tile_size=16, step_px=16)
        grouped_region_b = _make_grouped_region(block_size=2, tile_size=16, step_px=16)
        single_region = _solid_patch((77, 77, 77), size=16)

        mock_cu_image = MagicMock()
        mock_cu_image.read_region.side_effect = [
            iter([grouped_region_a, grouped_region_b]),
            iter([single_region]),
        ]
        fake_cucim = types.SimpleNamespace(CuImage=MagicMock(return_value=mock_cu_image))

        monkeypatch.setattr(
            tile_stream_mod,
            "iter_grouped_read_plans",
            lambda **kwargs: iter(interleaved_plans),
        )
        _monkeypatch_cucim_import(monkeypatch, fake_cucim)

        tiles = list(
            iter_tile_arrays_from_result(
                result=result,
                num_workers=2,
                gpu_decode=False,
            )
        )

        assert len(tiles) == 9
        assert mock_cu_image.read_region.call_count == 2
        assert mock_cu_image.read_region.call_args_list[0].args == (
            [(0, 0), (100, 0)],
            (32, 32),
        )
        assert mock_cu_image.read_region.call_args_list[1].args == (
            [(200, 0)],
            (16, 16),
        )

    def test_reader_iterator_groups_dense_8x8_grid_into_one_read(self):
        result = _make_grid_tiling_result(columns=8, rows=8, tile_size=16, step_px=16)
        grouped_region = _make_grouped_region(block_size=8, tile_size=16, step_px=16)

        mock_reader = _make_mock_reader(grouped_region)

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tiles = list(iter_tile_arrays_from_result(result=result))

        assert len(tiles) == 64
        assert mock_reader.read_region.call_count == 1
        mock_reader.read_region.assert_called_once_with(
            (0, 0),
            0,
            (128, 128),
            pad_missing=True,
        )
        assert int(tiles[0][0, 0, 0]) == 1
        assert int(tiles[1][0, 0, 0]) == 2
        assert int(tiles[8][0, 0, 0]) == 9
        assert int(tiles[-1][0, 0, 0]) == 64

    def test_reader_iterator_groups_dense_4x4_grid_into_one_read(self):
        result = _make_grid_tiling_result(columns=4, rows=4, tile_size=16, step_px=16)
        grouped_region = _make_grouped_region(block_size=4, tile_size=16, step_px=16)

        mock_reader = _make_mock_reader(grouped_region)

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tiles = list(iter_tile_arrays_from_result(result=result))

        assert len(tiles) == 16
        mock_reader.read_region.assert_called_once_with(
            (0, 0),
            0,
            (64, 64),
            pad_missing=True,
        )
        assert int(tiles[0][0, 0, 0]) == 1
        assert int(tiles[-1][0, 0, 0]) == 16

    def test_reader_iterator_uses_stride_based_group_size_when_tiles_overlap(self):
        result = _make_grid_tiling_result(columns=4, rows=4, tile_size=32, step_px=24)
        grouped_region = _make_grouped_region(block_size=4, tile_size=32, step_px=24)

        mock_reader = _make_mock_reader(grouped_region)

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tiles = list(iter_tile_arrays_from_result(result=result))

        assert len(tiles) == 16
        mock_reader.read_region.assert_called_once_with(
            (0, 0),
            0,
            (104, 104),
            pad_missing=True,
        )

    def test_reader_iterator_uses_2x2_blocks_for_incomplete_4x4_grid(self):
        result = _make_grid_tiling_result(columns=4, rows=4, tile_size=16, step_px=16)
        keep_mask = np.ones(len(result.coordinates), dtype=bool)
        keep_mask[-1] = False
        result = replace(
            result,
            tiles=replace(
                result.tiles,
                coordinates=result.coordinates[keep_mask],
                tissue_fractions=result.tissue_fractions[keep_mask],
                tile_index=np.arange(15, dtype=np.int32),
            ),
        )

        mock_reader = _make_mock_reader(
            # 3 reads at 32x32 (2x2 blocks) + 3 reads at 16x16 (single tiles)
            _solid_patch((1, 1, 1), size=32),
            _solid_patch((2, 2, 2), size=32),
            _solid_patch((3, 3, 3), size=32),
            _solid_patch((4, 4, 4), size=16),
            _solid_patch((5, 5, 5), size=16),
            _solid_patch((6, 6, 6), size=16),
        )

        with patch("hs2p.wsi.tile_stream.open_slide", return_value=mock_reader):
            tiles = list(iter_tile_arrays_from_result(result=result))

        assert len(tiles) == 15
        assert mock_reader.read_region.call_count == 6
        read_sizes = [call.args[2][0] for call in mock_reader.read_region.call_args_list]
        assert read_sizes == [32, 32, 32, 16, 16, 16]


class TestNeedsPixelFiltering:
    def test_no_filtering(self):
        from hs2p.api import _needs_pixel_filtering

        assert not _needs_pixel_filtering(FilterConfig())

    def test_white_only(self):
        from hs2p.api import _needs_pixel_filtering

        assert _needs_pixel_filtering(FilterConfig(filter_white=True))

    def test_black_only(self):
        from hs2p.api import _needs_pixel_filtering

        assert _needs_pixel_filtering(FilterConfig(filter_black=True))

    def test_both(self):
        from hs2p.api import _needs_pixel_filtering

        assert _needs_pixel_filtering(
            FilterConfig(filter_white=True, filter_black=True)
        )

    def test_grayspace_only(self):
        from hs2p.api import _needs_pixel_filtering

        assert _needs_pixel_filtering(FilterConfig(filter_grayspace=True))

    def test_blur_only(self):
        from hs2p.api import _needs_pixel_filtering

        assert _needs_pixel_filtering(FilterConfig(filter_blur=True))
