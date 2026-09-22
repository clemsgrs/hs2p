"""Public multi-argument functions and methods are keyword-only (ADR 0003): a
positional call raises ``TypeError`` before the body runs, so a value can no longer
land in the wrong same-typed slot."""
from pathlib import Path

import numpy as np
import pytest

from hs2p.fileops import promote_temp_file
from hs2p.tiling.coverage import _compute_tile_coverage, compute_tile_coverage
from hs2p.tiling.generate import _tiles_for_contour
from hs2p.utils.setup import write_config
from hs2p.wsi.backends.common import make_white_canvas, paste_region
from hs2p.wsi.preview import draw_grid, draw_grid_from_coordinates, overlay_mask_on_tile
from hs2p.wsi.visualization import overlay_mask_on_slide
from hs2p.wsi.wsi import WSI

_CANDIDATES = np.array([[0, 0]], dtype=np.int64)
_MASK = np.ones((4, 4), dtype=np.uint8)
_CANVAS = np.zeros((4, 4, 3), dtype=np.uint8)
# The methods are called unbound; the positional call fails before ``self`` is used.
_WSI = object()

POSITIONAL_CALL_CASES = [
    pytest.param(promote_temp_file, (Path("a.tmp"), Path("a")), id="promote_temp_file"),
    pytest.param(
        compute_tile_coverage, (_CANDIDATES, _MASK, 224, (1000, 1000)), id="compute_tile_coverage"
    ),
    pytest.param(
        _compute_tile_coverage,
        (_CANDIDATES, _MASK, 224, (1000, 1000)),
        id="_compute_tile_coverage",
    ),
    pytest.param(
        _tiles_for_contour,
        (np.zeros((4, 1, 2), dtype=np.int32), [], _MASK, (1000, 1000), 224, 224, 0.1),
        id="_tiles_for_contour",
    ),
    pytest.param(write_config, ({}, Path("out"), "config.yaml", False), id="write_config"),
    pytest.param(make_white_canvas, (4, 4), id="make_white_canvas"),
    pytest.param(paste_region, (_CANVAS, _CANVAS), id="paste_region"),
    pytest.param(
        overlay_mask_on_tile, (None, None, None, {"tumor": 1}, {}, 0.5), id="overlay_mask_on_tile"
    ),
    pytest.param(draw_grid, (_CANVAS, np.array([0, 0]), (2, 2), 1, (0, 0, 0, 255)), id="draw_grid"),
    pytest.param(
        draw_grid_from_coordinates,
        (_CANVAS, _WSI, [(0, 0)], (2, 2), 0),
        id="draw_grid_from_coordinates",
    ),
    pytest.param(overlay_mask_on_slide, (Path("slide.tif"), 1, "asap"), id="overlay_mask_on_slide"),
    pytest.param(WSI.get_tile, (_WSI, 0, 0, 224, 224, 0), id="WSI.get_tile"),
    pytest.param(
        WSI.get_best_level_for_spacing, (_WSI, 0.5, 0.07), id="WSI.get_best_level_for_spacing"
    ),
    pytest.param(
        WSI.read_region_at_spacing, (_WSI, (0, 0), 0.5, (4, 4)), id="WSI.read_region_at_spacing"
    ),
]


@pytest.mark.parametrize("func, args", POSITIONAL_CALL_CASES)
def test_positional_call_raises_type_error(func, args):
    with pytest.raises(TypeError, match="positional argument"):
        func(*args)


def test_get_tile_positional_call_raises_type_error(fake_backend):
    fake_backend(np.zeros((16, 16, 1), dtype=np.uint8))
    wsi = WSI(path=Path("synthetic-slide.tif"), backend="asap")

    with pytest.raises(TypeError, match="positional argument"):
        wsi.get_tile(0, 0, 4, 4, 0)


def test_get_tile_keyword_call_reads_the_tile(fake_backend):
    fake_backend(np.zeros((16, 16, 1), dtype=np.uint8))
    wsi = WSI(path=Path("synthetic-slide.tif"), backend="asap")

    tile = wsi.get_tile(x=0, y=0, width=4, height=6, level=0)

    assert tile.shape == (6, 4, 3)
