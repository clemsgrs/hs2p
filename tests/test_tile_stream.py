from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np

import hs2p.preprocessing as preprocessing_mod
from hs2p.wsi.streaming.stream import (
    iter_tile_records_from_reader,
    iter_tile_records_from_result,
)


def _make_result(
    *,
    coords: list[tuple[int, int]],
    tile_size: int = 16,
    backend: str = "openslide",
) -> preprocessing_mod.TilingResult:
    coords = np.asarray(coords, dtype=np.int64)
    return preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            x=coords[:, 0],
            y=coords[:, 1],
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
            slide_dimensions=[256, 256],
            level_downsamples=[1.0],
            overlap=0.0,
            min_tissue_fraction=0.1,
        ),
        sample_id="stream-slide",
        image_path=Path("/data/stream-slide.svs"),
        mask_path=None,
        backend=backend,
        requested_backend=backend,
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


def _mock_reader(*regions: np.ndarray):
    reader = MagicMock()
    reader.read_region.side_effect = list(regions)
    reader.level_dimensions = [(256, 256)]
    reader.level_downsamples = [(1.0, 1.0)]
    reader.__enter__.return_value = reader
    reader.__exit__.return_value = None
    return reader


def test_iter_tile_records_from_reader_preserves_tile_index_and_coordinates():
    result = _make_result(coords=[(0, 0), (0, 16), (16, 0), (16, 16)])
    region = np.zeros((32, 32, 3), dtype=np.uint8)
    region[:16, :16] = 1
    region[16:, :16] = 2
    region[:16, 16:] = 3
    region[16:, 16:] = 4
    reader = _mock_reader(region)

    records = list(iter_tile_records_from_reader(reader, result=result))

    assert [(record.tile_index, record.x, record.y) for record in records] == [
        (0, 0, 0),
        (1, 0, 16),
        (2, 16, 0),
        (3, 16, 16),
    ]
    assert [int(record.tile_arr[0, 0, 0]) for record in records] == [1, 2, 3, 4]
    reader.read_region.assert_called_once_with((0, 0), 0, (32, 32))


def test_iter_tile_records_from_result_uses_open_slide_for_generic_backends():
    result = _make_result(coords=[(0, 0), (16, 0)])
    reader = _mock_reader(
        np.full((16, 16, 3), 10, dtype=np.uint8),
        np.full((16, 16, 3), 20, dtype=np.uint8),
    )

    with patch("hs2p.wsi.streaming.stream.open_slide", return_value=reader) as mock_open:
        records = list(iter_tile_records_from_result(result=result))

    assert [int(record.tile_arr[0, 0, 0]) for record in records] == [10, 20]
    mock_open.assert_called_once()


def test_iter_cucim_batched_read_regions_suppresses_native_stderr():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "hs2p" / "wsi" / "streaming" / "batched.py"

    script = f"""
import os
import sys
import types
import importlib.util
from pathlib import Path
import numpy as np

class _FakeCuCIMReader:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        os.write(2, b"cuFile initialization failed\\n")

    def read_regions(self, locations, level, size, *, num_workers=None):
        del locations, level, size, num_workers
        os.write(2, b"cuInit Failed, error CUDA_ERROR_NOT_INITIALIZED\\n")
        return [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
        ]

sys.modules["hs2p.wsi.backends.cucim"] = types.SimpleNamespace(CuCIMReader=_FakeCuCIMReader)
spec = importlib.util.spec_from_file_location("hs2p_streaming_batched_test", Path({str(module_path)!r}))
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)

requests = [
    m.BatchedReadRequest(location=(0, 0), size=(16, 16)),
    m.BatchedReadRequest(location=(16, 0), size=(16, 16)),
]
regions = list(
    m.iter_cucim_batched_read_regions(
        image_path="/tmp/fake.svs",
        requests=requests,
        level=0,
        num_workers=2,
        spacing_override=None,
        gpu_decode=False,
    )
)
assert len(regions) == 2
assert [request.location for request, _region in regions] == [(0, 0), (16, 0)]
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
