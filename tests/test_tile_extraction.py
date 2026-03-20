"""Tests for extract_tiles_to_tar() and the save_tiles pipeline option."""

import io
import types
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from hs2p.api import TilingResult, extract_tiles_to_tar
from hs2p.configs.models import FilterConfig


def _make_tiling_result(
    num_tiles: int = 3,
    tile_size: int = 256,
    sample_id: str = "slide-1",
) -> TilingResult:
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
    )


def _solid_patch(color: tuple[int, int, int], size: int = 256) -> np.ndarray:
    """Return an (size, size, 3) uint8 array filled with *color*."""
    arr = np.empty((size, size, 3), dtype=np.uint8)
    arr[:] = color
    return arr


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
        mock_wsd.assert_called_once_with(result.image_path, backend="cucim")


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
