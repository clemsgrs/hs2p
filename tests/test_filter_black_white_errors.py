import numpy as np
import pytest

from hs2p.api import FilterConfig
from hs2p.wsi.tiling import filter_black_and_white_tiles


def test_filter_black_white_tile_errors_warn_and_keep_tile():
    class _Reader:
        def read_region(self, location, level, size, pad_missing=True):
            del location, level, size, pad_missing
            raise RuntimeError("read failure")

    with pytest.warns(UserWarning, match="tile filtering"):
        filtered = filter_black_and_white_tiles(
            reader=_Reader(),
            level_dimensions=[(100, 100)],
            level_downsamples=[(1.0, 1.0)],
            keep_flags=[1],
            coord_candidates=np.array([[0, 0]]),
            tile_size=32,
            tile_level=0,
            filter_params=FilterConfig(
                ref_tile_size=16,
                a_t=1,
                a_h=1,
                max_n_holes=1,
                filter_white=True,
                filter_black=False,
                white_threshold=220,
                black_threshold=25,
                fraction_threshold=0.9,
            ),
        )

    assert filtered == [1]


def test_filter_black_and_white_tiles_batches_reads_without_changing_decisions():
    canvas = np.array(
        [
            [[255, 255, 255], [255, 255, 255], [120, 120, 120], [120, 120, 120]],
            [[255, 255, 255], [255, 255, 255], [120, 120, 120], [120, 120, 120]],
            [[0, 0, 0], [0, 0, 0], [140, 140, 140], [140, 140, 140]],
            [[0, 0, 0], [0, 0, 0], [140, 140, 140], [140, 140, 140]],
        ],
        dtype=np.uint8,
    )
    calls = []

    class _Reader:
        def read_region(self, location, level, size, pad_missing=True):
            assert level == 0
            assert pad_missing is True
            x, y = location
            width, height = size
            calls.append((x, y, width, height))
            return canvas[y : y + height, x : x + width, :]

    filtered = filter_black_and_white_tiles(
        reader=_Reader(),
        level_dimensions=[(4, 4)],
        level_downsamples=[(1.0, 1.0)],
        keep_flags=[1, 1, 1, 1],
        coord_candidates=np.array([[0, 0], [2, 0], [0, 2], [2, 2]]),
        tile_size=2,
        tile_level=0,
        filter_params=FilterConfig(
            ref_tile_size=16,
            a_t=1,
            a_h=1,
            max_n_holes=1,
            filter_white=True,
            filter_black=True,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
    )

    assert filtered == [0, 1, 0, 1]
    assert len(calls) < 4
