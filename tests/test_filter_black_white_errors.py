import numpy as np
import pytest

wsi_mod = pytest.importorskip("hs2p.wsi.wsi")


def test_filter_black_white_tile_errors_warn_and_keep_tile():
    wsi = object.__new__(wsi_mod.WholeSlideImage)
    wsi.path = "fake-slide.tif"
    wsi.level_dimensions = [(100, 100)]
    wsi.level_downsamples = [(1.0, 1.0)]
    wsi.get_level_spacing = lambda level: 1.0

    def _raise(*args, **kwargs):
        raise RuntimeError("read failure")

    wsi.get_tile = _raise

    filter_params = wsi_mod.FilterParameters(
        ref_tile_size=16,
        a_t=1,
        a_h=1,
        max_n_holes=1,
        filter_white=True,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )

    keep_flags = [1]
    coord_candidates = np.array([[0, 0]])

    with pytest.warns(UserWarning, match="tile filtering"):
        filtered = wsi.filter_black_and_white_tiles(
            keep_flags=keep_flags,
            coord_candidates=coord_candidates,
            tile_size=32,
            tile_level=0,
            filter_params=filter_params,
        )

    assert filtered == [1]


def test_filter_black_and_white_tiles_batches_reads_without_changing_decisions():
    wsi = object.__new__(wsi_mod.WholeSlideImage)
    wsi.path = "fake-slide.tif"
    wsi.level_dimensions = [(4, 4)]
    wsi.level_downsamples = [(1.0, 1.0)]
    wsi.get_level_spacing = lambda level: 1.0

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

    def _get_tile(x, y, width, height, level):
        assert level == 0
        calls.append((x, y, width, height))
        return canvas[y : y + height, x : x + width, :]

    wsi.get_tile = _get_tile

    filter_params = wsi_mod.FilterParameters(
        ref_tile_size=16,
        a_t=1,
        a_h=1,
        max_n_holes=1,
        filter_white=True,
        filter_black=True,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )

    filtered = wsi.filter_black_and_white_tiles(
        keep_flags=[1, 1, 1, 1],
        coord_candidates=np.array([[0, 0], [2, 0], [0, 2], [2, 2]]),
        tile_size=2,
        tile_level=0,
        filter_params=filter_params,
    )

    assert filtered == [0, 1, 0, 1]
    assert len(calls) < 4
