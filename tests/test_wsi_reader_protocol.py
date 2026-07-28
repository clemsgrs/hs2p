import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
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


def _make_concrete_reader(
    monkeypatch,
    *,
    backend: str,
    native_spacing: float | None,
    spacing_override: float | None,
):
    level_dimensions = [(400, 200), (160, 80), (100, 50)]
    level_downsamples = [1.0, 2.5, 4.0]

    if backend == "asap":
        from hs2p.wsi.backends.asap import ASAPReader

        slide = SimpleNamespace(
            spacings=(
                [native_spacing * value for value in level_downsamples]
                if native_spacing is not None
                else []
            ),
            shapes=level_dimensions,
            downsamplings=level_downsamples,
        )
        fake_module = SimpleNamespace(WholeSlideImage=MagicMock(return_value=slide))
        monkeypatch.setitem(sys.modules, "wholeslidedata", fake_module)
        return ASAPReader("fake.svs", spacing_override=spacing_override)

    if backend == "cucim":
        from hs2p.wsi.backends.cucim import CuCIMReader
        import hs2p.wsi.backends.cucim as cucim_reader_mod

        metadata = {
            "cucim": {
                "resolutions": {
                    "level_dimensions": level_dimensions,
                    "level_downsamples": level_downsamples,
                }
            }
        }
        if native_spacing is not None:
            metadata["openslide"] = {"MPP": native_spacing}
        slide = SimpleNamespace(metadata=metadata)
        fake_module = SimpleNamespace(CuImage=MagicMock(return_value=slide))
        original_import_module = cucim_reader_mod.importlib.import_module
        monkeypatch.setattr(
            cucim_reader_mod.importlib,
            "import_module",
            lambda name: (
                fake_module if name == "cucim" else original_import_module(name)
            ),
        )
        return CuCIMReader("fake.svs", spacing_override=spacing_override)

    if backend == "openslide":
        from hs2p.wsi.backends.openslide import OpenSlideReader

        properties = (
            {"openslide.mpp-x": str(native_spacing)}
            if native_spacing is not None
            else {}
        )
        slide = SimpleNamespace(
            properties=properties,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            level_count=len(level_dimensions),
        )
        fake_module = SimpleNamespace(OpenSlide=MagicMock(return_value=slide))
        monkeypatch.setitem(sys.modules, "openslide", fake_module)
        return OpenSlideReader("fake.svs", spacing_override=spacing_override)

    if backend == "vips":
        from hs2p.wsi.backends.vips import VIPSReader

        class FakeVIPSImage:
            def __init__(self, width, height, fields):
                self.width = width
                self.height = height
                self._fields = fields

            def get_fields(self):
                return list(self._fields)

            def get(self, name):
                return self._fields[name]

        fields = {
            "vips-loader": "openslideload",
            "openslide.level-count": len(level_dimensions),
        }
        if native_spacing is not None:
            fields["openslide.mpp-x"] = native_spacing
        images = [
            FakeVIPSImage(width, height, fields)
            for width, height in level_dimensions
        ]

        def new_from_file(path, *, level=None, **kwargs):
            del path, kwargs
            return images[0 if level is None else int(level)]

        fake_module = SimpleNamespace(
            Image=SimpleNamespace(new_from_file=new_from_file)
        )
        monkeypatch.setitem(sys.modules, "pyvips", fake_module)
        return VIPSReader("fake.svs", spacing_override=spacing_override)

    raise AssertionError(f"unsupported test backend: {backend}")


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


@pytest.mark.parametrize("backend", ["asap", "cucim", "openslide", "vips"])
def test_spacing_override_rescues_missing_metadata_for_every_reader(
    monkeypatch, recwarn, backend
):
    reader = _make_concrete_reader(
        monkeypatch,
        backend=backend,
        native_spacing=None,
        spacing_override=0.25,
    )

    assert reader.native_spacing is None
    assert reader.spacing == 0.25
    assert reader.spacings == [0.25, 0.625, 1.0]
    assert len(recwarn) == 0


@pytest.mark.parametrize("backend", ["asap", "cucim", "openslide", "vips"])
def test_native_spacing_remains_baseline_without_override(
    monkeypatch, recwarn, backend
):
    reader = _make_concrete_reader(
        monkeypatch,
        backend=backend,
        native_spacing=0.5,
        spacing_override=None,
    )

    assert reader.spacing == 0.5
    assert reader.spacings == [0.5, 1.25, 2.0]
    assert len(recwarn) == 0


@pytest.mark.parametrize("backend", ["asap", "cucim", "openslide", "vips"])
@pytest.mark.parametrize(
    "spacing_override",
    [0.0, -0.25, float("nan"), float("inf"), float("-inf"), "not-a-spacing"],
)
def test_every_reader_rejects_invalid_spacing_overrides(
    monkeypatch, backend, spacing_override
):
    with pytest.raises(ValueError, match="finite positive"):
        _make_concrete_reader(
            monkeypatch,
            backend=backend,
            native_spacing=0.5,
            spacing_override=spacing_override,
        )


@pytest.mark.parametrize("backend", ["asap", "cucim", "openslide", "vips"])
def test_conflicting_override_warns_once_with_reader_context(
    monkeypatch, recwarn, backend
):
    reader = _make_concrete_reader(
        monkeypatch,
        backend=backend,
        native_spacing=0.5,
        spacing_override=0.25,
    )

    assert reader.spacing == 0.25
    assert reader.spacings == [0.25, 0.625, 1.0]
    assert len(recwarn) == 1
    message = str(recwarn[0].message)
    assert "path=fake.svs" in message
    assert "native=0.5" in message
    assert "supplied=0.25" in message
    assert f"backend={backend}" in message


@pytest.mark.parametrize("backend", ["asap", "cucim", "openslide", "vips"])
def test_numerically_equivalent_override_does_not_warn(
    monkeypatch, recwarn, backend
):
    reader = _make_concrete_reader(
        monkeypatch,
        backend=backend,
        native_spacing=0.1 + 0.2,
        spacing_override=0.3,
    )

    assert reader.spacing == 0.3
    assert reader.spacings == [0.3, 0.75, 1.2]
    assert len(recwarn) == 0


def test_select_level_prefers_finer_level_when_closest_match_is_too_coarse():
    selection = select_level(
        requested_spacing_um=2.7,
        level0_spacing_um=0.5,
        level_downsamples=[(1.0, 1.0), (2.0, 2.0), (8.0, 8.0)],
        tolerance=0.01,
    )

    assert selection.level == 1
    assert selection.read_spacing_um == 1.0
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


def test_cucim_reader_batched_reads_suppress_native_stderr():
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import os
import numpy as np
from unittest.mock import MagicMock
import hs2p.wsi.backends.cucim as m

mock_cu_image = MagicMock()
mock_cu_image.metadata = {
    "openslide": {"MPP": 0.5},
    "cucim": {"resolutions": {"level_dimensions": [[400, 200]], "level_downsamples": [1.0]}},
}

def _fake_read_region(**kwargs):
    del kwargs
    os.write(2, b"cuFile initialization failed\\n")
    return [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
    ]

mock_cu_image.read_region.side_effect = _fake_read_region
fake_cucim = type("FakeCuCIMModule", (), {"CuImage": MagicMock(return_value=mock_cu_image)})()
original_import_module = m.importlib.import_module
m.importlib.import_module = lambda name: fake_cucim if name == "cucim" else original_import_module(name)
reader = m.CuCIMReader("fake.svs")
list(reader.read_regions([(0, 0), (16, 0)], 0, (16, 16), num_workers=2))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""


def test_cucim_reader_repeated_single_reads_suppress_native_stderr():
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import os
import numpy as np
from unittest.mock import MagicMock
import hs2p.wsi.backends.cucim as m

mock_cu_image = MagicMock()
mock_cu_image.metadata = {
    "openslide": {"MPP": 0.5},
    "cucim": {"resolutions": {"level_dimensions": [[400, 200]], "level_downsamples": [1.0]}},
}

def _fake_read_region(**kwargs):
    del kwargs
    os.write(2, b"cuInit Failed, error CUDA_ERROR_NOT_INITIALIZED\\n")
    os.write(2, b"cuFile initialization failed\\n")
    return np.zeros((16, 16, 3), dtype=np.uint8)

mock_cu_image.read_region.side_effect = _fake_read_region
fake_cucim = type("FakeCuCIMModule", (), {"CuImage": MagicMock(return_value=mock_cu_image)})()
original_import_module = m.importlib.import_module
m.importlib.import_module = lambda name: fake_cucim if name == "cucim" else original_import_module(name)
reader = m.CuCIMReader("fake.svs")
reader.read_region((0, 0), 0, (16, 16))
reader.read_region((16, 0), 0, (16, 16))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""


def test_vips_reader_import_guard():
    if importlib.util.find_spec("pyvips") is not None:
        pytest.skip("pyvips is installed")

    from hs2p.wsi.backends.vips import VIPSReader

    with pytest.raises(ImportError, match="pyvips"):
        VIPSReader("fake.svs")
