import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from hs2p.wsi.reader import BatchRegionReader, SlideReader, select_level


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


@pytest.mark.parametrize(
    ("backend", "supports_batch"),
    [("asap", False), ("cucim", True), ("openslide", False), ("vips", False)],
)
def test_concrete_readers_conform_to_supported_protocols(monkeypatch, backend, supports_batch):
    reader = _make_concrete_reader(
        monkeypatch,
        backend=backend,
        native_spacing=0.5,
        spacing_override=None,
    )

    assert isinstance(reader, SlideReader)
    assert isinstance(reader, BatchRegionReader) is supports_batch


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
