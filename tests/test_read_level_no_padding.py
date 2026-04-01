"""read_level must never introduce 255-valued padding from the white canvas.

The padding mechanism in resolve_padded_read_bounds exists for read_region,
where a requested region may extend beyond the slide boundary. For read_level,
we are requesting the entire level — no padding should ever occur, even if the
backend returns an array 1 pixel short of the declared level dimensions (which
happens with pyramid TIFFs due to ceiling/floor rounding).

Regression for: binary tissue mask via cucim returning [0, 1, 255] unique values
because the white canvas (255) was not fully overwritten.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from hs2p.wsi.backends.cucim import CuCIMReader, _as_rgb_uint8
from hs2p.wsi.backends.openslide import OpenSlideReader


def _make_cucim_reader(declared_dims: tuple[int, int], backend_returns: np.ndarray) -> CuCIMReader:
    """Build a CuCIMReader without cucim installed, patching _read_region."""
    reader = object.__new__(CuCIMReader)
    width, height = declared_dims
    reader._level_dimensions = [(width, height)]
    reader._level_downsamples = [(1.0, 1.0)]
    reader._gpu_decode = False
    reader._slide = None

    def _fake_read_region(location, size, *, level, num_workers=1):
        del location, size, level, num_workers
        return backend_returns

    reader._read_region = _fake_read_region
    return reader


def _make_openslide_reader(declared_dims: tuple[int, int], backend_returns: np.ndarray) -> OpenSlideReader:
    """Build an OpenSlideReader without openslide installed, patching _slide."""
    from PIL import Image

    reader = object.__new__(OpenSlideReader)
    width, height = declared_dims
    reader._level_dimensions = [(width, height)]
    reader._level_downsamples = [(1.0, 1.0)]
    reader._spacing = 1.0
    reader._spacings = [1.0]

    pil_image = Image.fromarray(backend_returns[:, :, 0]).convert("RGB")
    fake_slide = MagicMock()
    fake_slide.read_region.return_value = pil_image
    reader._slide = fake_slide
    return reader


def test_cucim_read_level_no_padding_when_backend_returns_short_array():
    """When cucim returns 1 row fewer than declared, read_level must not pad with 255.

    Declared level dimensions: 4 wide, 4 tall
    Backend returns: 3 rows (off by 1, simulating pyramid rounding)
    Source pixel values: {0, 1} only (binary tissue mask)

    Old behavior (bug): creates a 4x4 white canvas (255), pastes 3 rows,
      bottom row stays 255 → unique values include 255.
    Expected behavior: no 255 introduced by the padding mechanism.
    """
    # 3 rows instead of the declared 4 — simulates cucim off-by-one
    backend_data = np.zeros((3, 4, 3), dtype=np.uint8)
    backend_data[1, 2] = 1  # one tissue pixel

    reader = _make_cucim_reader(declared_dims=(4, 4), backend_returns=backend_data)
    result = reader.read_level(0)

    assert 255 not in np.unique(result), (
        f"read_level introduced 255-valued padding. unique={np.unique(result)}"
    )


def test_openslide_read_level_no_padding_when_backend_returns_short_array():
    """Same contract for OpenSlideReader.read_level.

    Declared level dimensions: 4 wide, 4 tall
    Backend returns: 3-row image (off by 1)
    Source pixel values: {0, 1} only (binary tissue mask)
    """
    backend_data = np.zeros((3, 4, 3), dtype=np.uint8)
    backend_data[0, 1] = 1  # one tissue pixel

    reader = _make_openslide_reader(declared_dims=(4, 4), backend_returns=backend_data)
    result = reader.read_level(0)

    assert 255 not in np.unique(result), (
        f"read_level introduced 255-valued padding. unique={np.unique(result)}"
    )
