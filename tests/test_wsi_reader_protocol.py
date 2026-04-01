import importlib.util
from unittest.mock import MagicMock

import numpy as np
import pytest

from hs2p.wsi.backends.common import make_white_canvas
from hs2p.wsi.reader import BatchRegionReader, SlideReader, select_level


class SyntheticSlideReader:
    def __init__(
        self,
        *,
        width: int = 1000,
        height: int = 800,
        spacing: float = 0.5,
        n_levels: int = 3,
        backend_name: str = "synthetic",
    ) -> None:
        self._width = width
        self._height = height
        self._spacing = spacing
        self._n_levels = n_levels
        self._backend_name = backend_name
        rng = np.random.RandomState(42)
        self._image = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def spacing(self) -> float:
        return self._spacing

    @property
    def spacings(self) -> list[float]:
        return [self._spacing * (2**level) for level in range(self._n_levels)]

    @property
    def level_count(self) -> int:
        return self._n_levels

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return [
            (self._width // (2**level), self._height // (2**level))
            for level in range(self._n_levels)
        ]

    @property
    def level_downsamples(self) -> list[tuple[float, float]]:
        return [(float(2**level), float(2**level)) for level in range(self._n_levels)]

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        x, y = location
        width, height = size
        downsample = int(self.level_downsamples[level][0])
        x1 = min(int(x + width * downsample), self._width)
        y1 = min(int(y + height * downsample), self._height)
        region = self._image[int(y) : y1 : downsample, int(x) : x1 : downsample]
        padded = make_white_canvas(width, height)
        padded[: region.shape[0], : region.shape[1]] = region[:height, :width]
        return padded

    def read_level(self, level: int) -> np.ndarray:
        width, height = self.level_dimensions[level]
        return self.read_region((0, 0), level, (width, height))

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        return self.read_level(self.level_count - 1)[: int(size[1]), : int(size[0])]

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SyntheticBatchSlideReader(SyntheticSlideReader):
    def read_regions(
        self,
        locations: list[tuple[int, int]],
        level: int,
        size: tuple[int, int],
        *,
        num_workers: int | None = None,
    ):
        del num_workers
        return [self.read_region(location, level, size) for location in locations]


def test_synthetic_reader_conforms_to_slide_reader_protocol():
    reader = SyntheticSlideReader()
    assert isinstance(reader, SlideReader)
    region = reader.read_region((0, 0), 0, (64, 64))
    assert region.shape == (64, 64, 3)
    assert reader.level_dimensions[1] == (500, 400)


def test_synthetic_reader_uses_white_padding_for_out_of_bounds_reads():
    reader = SyntheticSlideReader(width=8, height=8, n_levels=1)

    region = reader.read_region((4, 4), 0, (8, 8))

    assert np.all(region[4:, :, :] == 255)
    assert np.all(region[:, 4:, :] == 255)


def test_synthetic_batch_reader_conforms_to_optional_batch_protocol():
    reader = SyntheticBatchSlideReader()
    assert isinstance(reader, BatchRegionReader)
    regions = list(reader.read_regions([(0, 0), (16, 16)], 0, (32, 32), num_workers=2))
    assert len(regions) == 2
    assert all(region.shape == (32, 32, 3) for region in regions)


def test_select_level_prefers_finer_level_when_closest_match_is_too_coarse():
    selection = select_level(
        requested_spacing_um=2.7,
        level0_spacing_um=0.5,
        level_downsamples=[(1.0, 1.0), (2.0, 2.0), (8.0, 8.0)],
        tolerance=0.01,
    )

    assert selection.level == 1
    assert selection.effective_spacing_um == 1.0
    assert not selection.is_within_tolerance


def test_openslide_reader_import_guard():
    if importlib.util.find_spec("openslide") is not None:
        pytest.skip("openslide is installed")

    from hs2p.wsi.backends.openslide import OpenSlideReader

    with pytest.raises(ImportError, match="openslide-python"):
        OpenSlideReader("fake.svs")


def test_cucim_reader_import_guard():
    if importlib.util.find_spec("cucim") is not None:
        pytest.skip("cucim is installed")

    from hs2p.wsi.backends.cucim import CuCIMReader

    with pytest.raises(ImportError, match="cucim"):
        CuCIMReader("fake.svs")


def test_cucim_reader_keeps_metadata_spacing_as_pyramid_baseline(monkeypatch):
    from hs2p.wsi.backends.cucim import CuCIMReader
    import hs2p.wsi.backends.cucim as cucim_reader_mod

    mock_cu_image = MagicMock()
    mock_cu_image.metadata = {
        "openslide": {"MPP": 0.5},
        "cucim": {
            "resolutions": {
                "level_dimensions": [[400, 200], [200, 100], [100, 50]],
                "level_downsamples": [1.0, 2.0, 4.0],
            }
        },
    }
    fake_cucim = type(
        "FakeCuCIMModule",
        (),
        {"CuImage": MagicMock(return_value=mock_cu_image)},
    )()
    original_import_module = cucim_reader_mod.importlib.import_module
    monkeypatch.setattr(
        cucim_reader_mod.importlib,
        "import_module",
        lambda name: fake_cucim if name == "cucim" else original_import_module(name),
    )

    reader = CuCIMReader("fake.svs", spacing_override=0.25)

    assert reader.native_spacing == 0.5
    assert reader.spacing == 0.25
    assert reader.spacings == [0.5, 1.0, 2.0]


def test_vips_reader_import_guard():
    if importlib.util.find_spec("pyvips") is not None:
        pytest.skip("pyvips is installed")

    from hs2p.wsi.backends.vips import VIPSReader

    with pytest.raises(ImportError, match="pyvips"):
        VIPSReader("fake.svs")
