import os
import signal
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
    require_spacing: bool = True,
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
        return ASAPReader(
            "fake.svs",
            spacing_override=spacing_override,
            require_spacing=require_spacing,
        )

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
        return CuCIMReader(
            "fake.svs",
            spacing_override=spacing_override,
            require_spacing=require_spacing,
        )

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
        return OpenSlideReader(
            "fake.svs",
            spacing_override=spacing_override,
            require_spacing=require_spacing,
        )

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
        return VIPSReader(
            "fake.svs",
            spacing_override=spacing_override,
            require_spacing=require_spacing,
        )

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
def test_every_reader_requires_spacing_by_default(monkeypatch, backend):
    with pytest.raises(
        ValueError, match=rf"Unable to infer slide spacing.*backend={backend}"
    ):
        _make_concrete_reader(
            monkeypatch,
            backend=backend,
            native_spacing=None,
            spacing_override=None,
        )


@pytest.mark.parametrize("backend", ["asap", "cucim", "openslide", "vips"])
def test_every_reader_opens_a_spacing_less_source_when_spacing_is_not_required(
    monkeypatch, recwarn, backend
):
    reader = _make_concrete_reader(
        monkeypatch,
        backend=backend,
        native_spacing=None,
        spacing_override=None,
        require_spacing=False,
    )

    assert reader.native_spacing is None
    assert reader.spacing is None
    assert reader.spacings == []
    assert reader.level_dimensions == [(400, 200), (160, 80), (100, 50)]
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


def _write_tiff(path: Path, **tags) -> Path:
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(
        path,
        np.zeros((32, 64), dtype=np.uint8),
        tile=(32, 32),
        photometric="minisblack",
        **tags,
    )
    return path


def test_vips_reader_rejects_untagged_tiff_by_default(tmp_path):
    pytest.importorskip("pyvips")
    from hs2p.wsi.backends.vips import VIPSReader

    path = _write_tiff(tmp_path / "untagged.tif")

    with pytest.raises(ValueError, match="Unable to infer slide spacing"):
        VIPSReader(path)


def test_vips_reader_opens_untagged_tiff_without_spacing(tmp_path):
    pytest.importorskip("pyvips")
    from hs2p.wsi.backends.vips import VIPSReader

    path = _write_tiff(tmp_path / "untagged.tif")

    reader = VIPSReader(path, require_spacing=False)

    assert reader.native_spacing is None
    assert reader.spacing is None


def test_vips_reader_reads_spacing_from_tiff_resolution_tag(tmp_path):
    pytest.importorskip("pyvips")
    from hs2p.wsi.backends.vips import VIPSReader

    path = _write_tiff(
        tmp_path / "tagged.tif", resolution=(10000, 10000), resolutionunit="CENTIMETER"
    )

    reader = VIPSReader(path)

    # 10000 px/cm = 1000 px/mm -> 1000 / 1000 = 1.0 um/px
    assert reader.native_spacing == 1.0


class _FakeNativeCuImage:
    """Stand-in for ``cucim.CuImage`` that records every ``read_region`` call.

    Its ``read_region`` has the defaults of ``cucim.CuImage.read_region`` (cuCIM 26.06:
    ``num_workers=0``, ``device=cpu``), so ``calls`` holds the values cuCIM reads with,
    whether hs2p passes them or leaves them to cuCIM's defaults.
    """

    def __init__(self, read_region):
        self.metadata = {
            "openslide": {"MPP": 0.5},
            "cucim": {
                "resolutions": {
                    "level_dimensions": [[400, 200]],
                    "level_downsamples": [1.0],
                }
            },
        }
        self.calls: list[dict] = []
        self._read_region = read_region

    def read_region(
        self, location=(), size=(), level=0, num_workers=0, device="cpu", **kwargs
    ):
        call = {
            "location": location,
            "size": size,
            "level": level,
            "num_workers": num_workers,
            "device": device,
            **kwargs,
        }
        self.calls.append(call)
        return self._read_region(**call)


def _make_cucim_reader_over(monkeypatch, native, *, gpu_decode: bool):
    import hs2p.wsi.backends.cucim as cucim_reader_mod

    monkeypatch.setenv("ENABLE_CUSLIDE2", "0")
    fake_module = SimpleNamespace(CuImage=MagicMock(return_value=native))
    original_import_module = cucim_reader_mod.importlib.import_module
    monkeypatch.setattr(
        cucim_reader_mod.importlib,
        "import_module",
        lambda name: (
            fake_module if name == "cucim" else original_import_module(name)
        ),
    )
    return cucim_reader_mod.CuCIMReader("fake.svs", gpu_decode=gpu_decode)


def _raise(error):
    def read_region(**kwargs):
        del kwargs
        raise error

    return read_region


def test_cucim_single_cpu_read_type_error_propagates_after_one_native_call(
    monkeypatch,
):
    original = TypeError("native read_region rejected arguments")
    native = _FakeNativeCuImage(_raise(original))
    reader = _make_cucim_reader_over(monkeypatch, native, gpu_decode=False)

    with pytest.raises(TypeError) as excinfo:
        reader.read_region((0, 0), 0, (16, 16))

    assert excinfo.value is original
    assert len(native.calls) == 1


def test_cucim_batched_cpu_read_type_error_propagates_after_one_native_call(
    monkeypatch,
):
    original = TypeError("native read_region rejected arguments")
    native = _FakeNativeCuImage(_raise(original))
    reader = _make_cucim_reader_over(monkeypatch, native, gpu_decode=False)

    with pytest.raises(TypeError) as excinfo:
        list(reader.read_regions([(0, 0), (16, 0)], 0, (16, 16), num_workers=2))

    assert excinfo.value is original
    assert len(native.calls) == 1


def _raise_type_error_on_gpu_request(error, pixels):
    """Native fake whose GPU request fails but whose CPU request would succeed.

    A device-less retry would therefore silently return CPU-decoded pixels.
    """

    def read_region(**kwargs):
        if kwargs.get("device") == "cuda":
            raise error
        return pixels

    return read_region


def test_cucim_single_gpu_read_type_error_propagates_without_cpu_fallback(
    monkeypatch,
):
    original = TypeError("GPU decode request rejected")
    cpu_pixels = np.full((16, 16, 3), 7, dtype=np.uint8)
    native = _FakeNativeCuImage(_raise_type_error_on_gpu_request(original, cpu_pixels))
    reader = _make_cucim_reader_over(monkeypatch, native, gpu_decode=True)

    with pytest.raises(TypeError) as excinfo:
        reader.read_region((0, 0), 0, (16, 16))

    assert excinfo.value is original
    assert [call.get("device") for call in native.calls] == ["cuda"]


def test_cucim_batched_gpu_read_type_error_propagates_without_cpu_fallback(
    monkeypatch,
):
    original = TypeError("GPU decode request rejected")
    cpu_pixels = [np.full((16, 16, 3), 7, dtype=np.uint8)] * 2
    native = _FakeNativeCuImage(_raise_type_error_on_gpu_request(original, cpu_pixels))
    reader = _make_cucim_reader_over(monkeypatch, native, gpu_decode=True)

    with pytest.raises(TypeError) as excinfo:
        list(reader.read_regions([(0, 0), (16, 0)], 0, (16, 16), num_workers=2))

    assert excinfo.value is original
    assert [call.get("device") for call in native.calls] == ["cuda"]


def test_cucim_successful_gpu_requests_keep_cuda_device_and_decoded_pixels(
    monkeypatch,
):
    def read_region(**kwargs):
        if isinstance(kwargs["location"], list):
            return [
                np.full((16, 16, 3), 10 + index, dtype=np.uint8)
                for index, _ in enumerate(kwargs["location"])
            ]
        return np.full((16, 16, 3), 42, dtype=np.uint8)

    native = _FakeNativeCuImage(read_region)
    reader = _make_cucim_reader_over(monkeypatch, native, gpu_decode=True)

    single = reader.read_region((0, 0), 0, (16, 16))
    batched = list(reader.read_regions([(0, 0), (16, 0)], 0, (16, 16), num_workers=3))

    assert np.array_equal(single, np.full((16, 16, 3), 42, dtype=np.uint8))
    assert [int(tile[0, 0, 0]) for tile in batched] == [10, 11]
    assert [call.get("device") for call in native.calls] == ["cuda", "cuda"]
    assert native.calls[1]["num_workers"] == 3


def _blank_pixels(**kwargs):
    width, height = kwargs["size"]
    if isinstance(kwargs["location"], list):
        return [np.zeros((height, width, 3), dtype=np.uint8) for _ in kwargs["location"]]
    return np.zeros((height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize(("gpu_decode", "device"), [(False, "cpu"), (True, "cuda")])
def test_cucim_reads_reach_native_read_region_with_unchanged_values(
    monkeypatch, gpu_decode, device
):
    """The values cuCIM reads with, as hs2p 6.1.1 sent them: a single read leaves the
    worker count at cuCIM's default of 0, a batched read asks for at least one worker,
    and only ``gpu_decode`` moves the device off the CPU."""
    native = _FakeNativeCuImage(_blank_pixels)
    reader = _make_cucim_reader_over(monkeypatch, native, gpu_decode=gpu_decode)
    locations = [(0, 0), (16, 0)]

    reader.read_region((8, 4), 0, (16, 16))
    reader.read_level(0)
    list(reader.read_regions(locations, 0, (16, 16)))
    list(reader.read_regions(locations, 0, (16, 16), num_workers=1))
    list(reader.read_regions(locations, 0, (16, 16), num_workers=4))

    single = {"level": 0, "num_workers": 0, "device": device}
    batched = {"location": [(0, 0), (16, 0)], "size": (16, 16), "level": 0, "device": device}
    assert native.calls == [
        {"location": (8, 4), "size": (16, 16), **single},
        {"location": (0, 0), "size": (400, 200), **single},
        {**batched, "num_workers": 1},
        {**batched, "num_workers": 1},
        {**batched, "num_workers": 4},
    ]


def test_cucim_reader_never_expands_a_dict_into_native_read_region():
    """No ``self._slide.read_region`` call expands a dict with ``**``, the form that
    segfaults cuCIM (#281). The child-process test below skips where cuCIM is not
    installed, as in CI; this check runs everywhere."""
    import ast

    import hs2p.wsi.backends.cucim as cucim_reader_mod

    tree = ast.parse(Path(cucim_reader_mod.__file__).read_text())
    native_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_region"
        and ast.unparse(node.func.value) == "self._slide"
    ]

    assert native_calls
    expanded = [
        ast.unparse(call)
        for call in native_calls
        if any(keyword.arg is None for keyword in call.keywords)
    ]
    assert expanded == []


def _write_nine_level_tiff(path: Path) -> Path:
    """A 512 px RGB pyramid halved eight times down to 2 x 2 px, at 0.25 um/px.

    Tiled and uncompressed, so writing it needs no codec.
    """
    tifffile = pytest.importorskip("tifffile")
    level0 = np.full((512, 512, 3), 200, dtype=np.uint8)
    pixels_per_cm = 1e4 / 0.25
    with tifffile.TiffWriter(path) as writer:
        for level in range(9):
            step = 2**level
            writer.write(
                level0[::step, ::step],
                tile=(16, 16),
                photometric="rgb",
                subfiletype=0 if level == 0 else 1,
                resolution=(pixels_per_cm / step, pixels_per_cm / step),
                resolutionunit="CENTIMETER",
            )
    return path


_CUCIM_UNAVAILABLE_EXIT = 77

_REPEATED_CUCIM_READS = f"""
import sys

sys.path.insert(0, sys.argv[1])
try:
    # Not ``import cucim``: it succeeds even when the native binding cannot load.
    from cucim import CuImage  # noqa: F401
except ImportError:
    sys.exit({_CUCIM_UNAVAILABLE_EXIT})

from hs2p.wsi.reader import open_slide

print("progress: auto probe", flush=True)
reader = open_slide(sys.argv[2], backend="auto")
print("progress: auto selected", reader.backend_name, flush=True)
assert reader.backend_name == "cucim"
assert reader.level_count == 9
for read in range(1, 21):
    level = read % reader.level_count
    tile = reader.read_region((0, 0), level, (16, 16))
    assert tile.shape == (16, 16, 3) and (tile[0, 0] == 200).all(), (read, level)
    print("progress: single read", read, flush=True)
locations = [(0, 0), (16, 0), (0, 16)]
for read in range(1, 21):
    num_workers = 1 if read % 2 else 4
    tiles = list(reader.read_regions(locations, 0, (16, 16), num_workers=num_workers))
    assert len(tiles) == 3 and all((tile == 200).all() for tile in tiles), read
    print("progress: batched read", read, flush=True)
"""


def test_cucim_reader_survives_repeated_reads_after_auto_selects_it(tmp_path):
    """cuCIM 26.06 on Python 3.11.0rc1 segfaults on the eighth ``read_region`` call
    made as ``read_region(**kwargs)``, and on the second or third batched one (#281).

    ``auto`` probes each of the nine levels once, then 20 single and 20 batched reads
    go through the reader. It runs in a child process because the fault kills the
    interpreter; the child also checks that ``cucim.CuImage`` imports, so the pytest
    process never loads cuCIM. GPUs are hidden from the child: every read decodes on the CPU, and a cuCIM
    import that fails to initialise a busy GPU is a different fault (#278). The cuSlide2
    plugin is turned off, because it cannot open the file with GPUs hidden.
    """
    repo_root = Path(__file__).resolve().parents[1]
    slide_path = _write_nine_level_tiff(tmp_path / "nine-levels.tif")

    result = subprocess.run(
        [sys.executable, "-c", _REPEATED_CUCIM_READS, str(repo_root), str(slide_path)],
        cwd=repo_root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "ENABLE_CUSLIDE2": "0"},
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    if result.returncode == _CUCIM_UNAVAILABLE_EXIT:
        pytest.skip("cucim.CuImage cannot be imported")
    outcome = (
        f"was killed by {signal.Signals(-result.returncode).name}"
        if result.returncode < 0
        else f"exited with {result.returncode}"
    )
    # cuCIM prints its own CUDA warnings to stdout, between the child's progress lines.
    progress = [line for line in result.stdout.splitlines() if line.startswith("progress:")]
    assert result.returncode == 0, (
        f"the child {outcome} after {(progress or ['no progress'])[-1]!r}\n"
        f"{result.stderr[-2000:]}"
    )
