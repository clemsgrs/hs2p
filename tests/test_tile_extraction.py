"""Tests for extract_tiles_to_tar() and the save_tiles pipeline option."""

import io
import tarfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from hs2p.api import (
    TilingResult,
    _iter_cucim_tile_arrays_for_tar_extraction,
    _iter_wsd_tile_arrays_for_tar_extraction,
    extract_tiles_to_tar,
)
from hs2p.configs.models import FilterConfig


def _make_tiling_result(
    num_tiles: int = 3,
    tile_size: int = 256,
    sample_id: str = "slide-1",
    *,
    step_px: int | None = None,
) -> TilingResult:
    if step_px is None:
        step_px = tile_size
    return TilingResult(
        sample_id=sample_id,
        image_path=Path("/data/slide-1.svs"),
        mask_path=None,
        backend="openslide",
        x=np.arange(num_tiles, dtype=np.int64) * tile_size,
        y=np.zeros(num_tiles, dtype=np.int64),
        tile_index=np.arange(num_tiles, dtype=np.int32),
        target_spacing_um=0.5,
        target_tile_size_px=tile_size,
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=tile_size,
        tile_size_lv0=tile_size,
        overlap=0.0,
        tissue_threshold=0.1,
        num_tiles=num_tiles,
        config_hash="abc123",
        read_step_px=step_px,
        step_px_lv0=step_px,
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
) -> TilingResult:
    x_coords: list[int] = []
    y_coords: list[int] = []
    for x_idx in range(columns):
        for y_idx in range(rows):
            x_coords.append(x_idx * step_px)
            y_coords.append(y_idx * step_px)
    return TilingResult(
        sample_id="grid-slide",
        image_path=Path("/data/grid-slide.svs"),
        mask_path=None,
        backend="openslide",
        x=np.asarray(x_coords, dtype=np.int64),
        y=np.asarray(y_coords, dtype=np.int64),
        tile_index=np.arange(columns * rows, dtype=np.int32),
        target_spacing_um=0.5,
        target_tile_size_px=tile_size,
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=tile_size,
        tile_size_lv0=tile_size,
        overlap=0.0 if step_px == tile_size else 1.0 - (step_px / tile_size),
        tissue_threshold=0.1,
        num_tiles=columns * rows,
        config_hash="grid-hash",
        read_step_px=step_px,
        step_px_lv0=step_px,
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


class TestExtractTilesToTar:
    """Unit tests for extract_tiles_to_tar()."""

    def test_creates_tar_with_correct_number_of_jpegs(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=3)
        colors = [(128, 0, 0), (0, 128, 0), (0, 0, 128)]

        mock_wsi = MagicMock()
        mock_wsi.get_patch.side_effect = [_solid_patch(c) for c in colors]

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
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

    def test_white_filtering_drops_white_tile(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=3)
        patches = [
            _solid_patch((128, 0, 0)),   # keep
            _solid_patch((250, 250, 250)),  # all-white → drop
            _solid_patch((0, 0, 128)),   # keep
        ]

        mock_wsi = MagicMock()
        mock_wsi.get_patch.side_effect = patches

        filter_cfg = FilterConfig(
            filter_white=True,
            white_threshold=220,
            fraction_threshold=0.9,
        )

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            tar_path, filtered = extract_tiles_to_tar(
                result, output_dir=tmp_path, filter_params=filter_cfg
            )

        # Tar should have 2 tiles
        with tarfile.open(tar_path, "r") as tf:
            assert len(tf.getmembers()) == 2

        # Result should reflect filtering
        assert filtered.num_tiles == 2
        np.testing.assert_array_equal(filtered.x, [0, 512])
        np.testing.assert_array_equal(filtered.y, [0, 0])
        assert len(filtered.tile_index) == 2
        np.testing.assert_array_equal(filtered.tile_index, [0, 1])

    def test_black_filtering_drops_black_tile(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=2)
        patches = [
            _solid_patch((5, 5, 5)),    # all-black → drop
            _solid_patch((128, 128, 0)),  # keep
        ]

        mock_wsi = MagicMock()
        mock_wsi.get_patch.side_effect = patches

        filter_cfg = FilterConfig(
            filter_black=True,
            black_threshold=25,
            fraction_threshold=0.9,
        )

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            tar_path, filtered = extract_tiles_to_tar(
                result, output_dir=tmp_path, filter_params=filter_cfg
            )

        with tarfile.open(tar_path, "r") as tf:
            assert len(tf.getmembers()) == 1

        assert filtered.num_tiles == 1
        np.testing.assert_array_equal(filtered.x, [256])

    def test_all_tiles_filtered_produces_empty_tar(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=1)
        mock_wsi = MagicMock()
        mock_wsi.get_patch.return_value = _solid_patch((250, 250, 250))

        filter_cfg = FilterConfig(
            filter_white=True,
            white_threshold=220,
            fraction_threshold=0.9,
        )

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            tar_path, filtered = extract_tiles_to_tar(
                result, output_dir=tmp_path, filter_params=filter_cfg
            )

        with tarfile.open(tar_path, "r") as tf:
            assert len(tf.getmembers()) == 0

        assert filtered.num_tiles == 0
        assert len(filtered.x) == 0

    def test_resizes_when_read_and_target_sizes_differ(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=1, tile_size=224)
        # Override read_tile_size_px to be different from target
        result = TilingResult(
            **{
                **{f.name: getattr(result, f.name) for f in result.__dataclass_fields__.values()},
                "read_tile_size_px": 512,
                "target_tile_size_px": 224,
            }
        )

        mock_wsi = MagicMock()
        mock_wsi.get_patch.return_value = _solid_patch((100, 100, 100), size=512)

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            tar_path, _ = extract_tiles_to_tar(result, output_dir=tmp_path)

        with tarfile.open(tar_path, "r") as tf:
            data = tf.extractfile(tf.getmembers()[0]).read()
            img = Image.open(io.BytesIO(data))
            assert img.size == (224, 224)

    def test_strips_alpha_channel(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=1)
        rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        rgba[:, :, :3] = 100
        rgba[:, :, 3] = 255

        mock_wsi = MagicMock()
        mock_wsi.get_patch.return_value = rgba

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            tar_path, _ = extract_tiles_to_tar(result, output_dir=tmp_path)

        with tarfile.open(tar_path, "r") as tf:
            data = tf.extractfile(tf.getmembers()[0]).read()
            img = Image.open(io.BytesIO(data))
            assert img.mode == "RGB"

    def test_custom_tiles_dir(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=1)
        custom_dir = tmp_path / "custom_output"

        mock_wsi = MagicMock()
        mock_wsi.get_patch.return_value = _solid_patch((50, 50, 50))

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            tar_path, _ = extract_tiles_to_tar(
                result, output_dir=tmp_path, tiles_dir=custom_dir
            )

        assert tar_path == custom_dir / "slide-1.tiles.tar"
        assert tar_path.is_file()

    def test_no_filter_params_keeps_all_tiles(self, tmp_path: Path):
        result = _make_tiling_result(num_tiles=2)
        mock_wsi = MagicMock()
        # Even white tiles should be kept when no filter_params
        mock_wsi.get_patch.side_effect = [
            _solid_patch((255, 255, 255)),
            _solid_patch((0, 0, 0)),
        ]

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            _, out_result = extract_tiles_to_tar(
                result, output_dir=tmp_path, filter_params=None
            )

        assert out_result is result  # unchanged

    def test_cucim_backend_uses_batched_read_region(self, monkeypatch, tmp_path: Path):
        result = _make_tiling_result(num_tiles=2)
        result.backend = "cucim"
        result.read_level = 3
        result.read_tile_size_px = 128

        regions = [_solid_patch((10, 20, 30), size=128), _solid_patch((40, 50, 60), size=128)]
        mock_cu_image = MagicMock()
        mock_cu_image.read_region.return_value = iter(regions)
        fake_cucim = types.SimpleNamespace(CuImage=MagicMock(return_value=mock_cu_image))

        import hs2p.api as api_mod

        monkeypatch.setattr(
            api_mod.importlib,
            "import_module",
            lambda name: fake_cucim if name == "cucim" else None,
        )

        with patch("wholeslidedata.WholeSlideImage") as mock_wsd:
            tar_path, out_result = extract_tiles_to_tar(
                result,
                output_dir=tmp_path,
                num_workers=5,
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
        mock_wsd.assert_not_called()

    def test_cucim_backend_falls_back_to_wsd_when_cucim_is_unavailable(
        self, monkeypatch, tmp_path: Path
    ):
        result = _make_tiling_result(num_tiles=1)
        result.backend = "cucim"

        mock_wsi = MagicMock()
        mock_wsi.get_patch.return_value = _solid_patch((70, 80, 90))

        import hs2p.api as api_mod

        def _import_module(name):
            if name == "cucim":
                raise ModuleNotFoundError("No module named 'cucim'")
            raise AssertionError(f"unexpected module import: {name}")

        monkeypatch.setattr(api_mod.importlib, "import_module", _import_module)

        with pytest.warns(UserWarning, match="CuCIM is unavailable"), patch(
            "wholeslidedata.WholeSlideImage",
            return_value=mock_wsi,
        ) as mock_wsd:
            tar_path, out_result = extract_tiles_to_tar(
                result,
                output_dir=tmp_path,
                num_workers=4,
            )

        assert tar_path.is_file()
        assert out_result is result
        mock_wsd.assert_called_once_with(str(result.image_path), backend="cucim")

    def test_cucim_iterator_groups_dense_4x4_grid_into_one_batched_read(
        self, monkeypatch
    ):
        result = _make_grid_tiling_result(columns=4, rows=4, tile_size=16, step_px=16)
        result.backend = "cucim"
        grouped_region = _make_grouped_region(block_size=4, tile_size=16, step_px=16)

        mock_cu_image = MagicMock()
        mock_cu_image.read_region.return_value = iter([grouped_region])
        fake_cucim = types.SimpleNamespace(CuImage=MagicMock(return_value=mock_cu_image))

        import hs2p.api as api_mod

        monkeypatch.setattr(
            api_mod.importlib,
            "import_module",
            lambda name: fake_cucim if name == "cucim" else None,
        )

        tiles = list(
            _iter_cucim_tile_arrays_for_tar_extraction(
                result=result,
                num_workers=7,
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

    def test_wsd_iterator_groups_dense_8x8_grid_into_one_read(self):
        result = _make_grid_tiling_result(columns=8, rows=8, tile_size=16, step_px=16)
        grouped_region = _make_grouped_region(block_size=8, tile_size=16, step_px=16)

        mock_wsi = MagicMock()
        mock_wsi.get_patch.side_effect = [grouped_region]

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            tiles = list(_iter_wsd_tile_arrays_for_tar_extraction(result=result))

        assert len(tiles) == 64
        assert mock_wsi.get_patch.call_count == 1
        mock_wsi.get_patch.assert_called_once_with(
            0,
            0,
            128,
            128,
            spacing=0.5,
            center=False,
        )
        assert int(tiles[0][0, 0, 0]) == 1
        assert int(tiles[1][0, 0, 0]) == 2
        assert int(tiles[8][0, 0, 0]) == 9
        assert int(tiles[-1][0, 0, 0]) == 64

    def test_wsd_iterator_groups_dense_4x4_grid_into_one_read(self):
        result = _make_grid_tiling_result(columns=4, rows=4, tile_size=16, step_px=16)
        grouped_region = _make_grouped_region(block_size=4, tile_size=16, step_px=16)

        mock_wsi = MagicMock()
        mock_wsi.get_patch.side_effect = [grouped_region]

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            tiles = list(_iter_wsd_tile_arrays_for_tar_extraction(result=result))

        assert len(tiles) == 16
        mock_wsi.get_patch.assert_called_once_with(
            0,
            0,
            64,
            64,
            spacing=0.5,
            center=False,
        )
        assert int(tiles[0][0, 0, 0]) == 1
        assert int(tiles[-1][0, 0, 0]) == 16

    def test_wsd_iterator_uses_stride_based_group_size_when_tiles_overlap(self):
        result = _make_grid_tiling_result(columns=4, rows=4, tile_size=32, step_px=24)
        grouped_region = _make_grouped_region(block_size=4, tile_size=32, step_px=24)

        mock_wsi = MagicMock()
        mock_wsi.get_patch.side_effect = [grouped_region]

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            tiles = list(_iter_wsd_tile_arrays_for_tar_extraction(result=result))

        assert len(tiles) == 16
        mock_wsi.get_patch.assert_called_once_with(
            0,
            0,
            104,
            104,
            spacing=0.5,
            center=False,
        )

    def test_wsd_iterator_uses_2x2_blocks_for_incomplete_4x4_grid(self):
        result = _make_grid_tiling_result(columns=4, rows=4, tile_size=16, step_px=16)
        keep_mask = np.ones(result.num_tiles, dtype=bool)
        keep_mask[-1] = False
        result = TilingResult(
            **{
                **{
                    field.name: getattr(result, field.name)
                    for field in result.__dataclass_fields__.values()
                },
                "x": result.x[keep_mask],
                "y": result.y[keep_mask],
                "tile_index": np.arange(15, dtype=np.int32),
                "num_tiles": 15,
            }
        )

        mock_wsi = MagicMock()
        # 3 reads at 32x32 (2x2 blocks) + 3 reads at 16x16 (single tiles)
        mock_wsi.get_patch.side_effect = [
            _solid_patch((1, 1, 1), size=32),
            _solid_patch((2, 2, 2), size=32),
            _solid_patch((3, 3, 3), size=32),
            _solid_patch((4, 4, 4), size=16),
            _solid_patch((5, 5, 5), size=16),
            _solid_patch((6, 6, 6), size=16),
        ]

        with patch("wholeslidedata.WholeSlideImage", return_value=mock_wsi):
            tiles = list(_iter_wsd_tile_arrays_for_tar_extraction(result=result))

        assert len(tiles) == 15
        assert mock_wsi.get_patch.call_count == 6
        read_sizes = [call.args[2] for call in mock_wsi.get_patch.call_args_list]
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
