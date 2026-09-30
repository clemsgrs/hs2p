from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import hs2p.preprocessing as preprocessing_mod
from hs2p.wsi.streaming.stream import iter_tile_records_from_reader


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
            read_tile_size_px=tile_size,
            read_spacing_um=0.5,
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
        requested_seg_downsample=64,
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


class _CoordinateCodedReader:
    """A 2x pyramid whose level-1 pixels encode their own level-1 coordinates."""

    def __init__(self, level1_size: int = 128):
        from hs2p.wsi.backends.common import paste_region, resolve_padded_read_bounds

        self._paste_region = paste_region
        self._resolve_bounds = resolve_padded_read_bounds
        ys, xs = np.mgrid[:level1_size, :level1_size]
        self.level1 = np.stack([xs % 256, ys % 256, np.full_like(xs, 120)], axis=-1).astype(
            np.uint8
        )
        self.level_dimensions = [(2 * level1_size, 2 * level1_size), (level1_size, level1_size)]
        self.level_downsamples = [(1.0, 1.0), (2.0, 2.0)]

    def read_region(self, location, level, size):
        assert level == 1
        bounds = self._resolve_bounds(
            location=location,
            size=size,
            level_dimensions=self.level_dimensions[1],
            downsample=2.0,
        )
        x = int(np.floor(max(location[0], 0) / 2.0))
        y = int(np.floor(max(location[1], 0) / 2.0))
        width, height = bounds.read_size
        return self._paste_region(
            canvas=bounds.canvas,
            region=self.level1[y : y + height, x : x + width],
            paste_offset=bounds.paste_offset,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _two_x_overlap_result(step_px_lv0: int) -> preprocessing_mod.TilingResult:
    from dataclasses import replace

    coords = [(x * step_px_lv0, y * step_px_lv0) for x in range(8) for y in range(8)]
    result = _make_result(coords=coords, tile_size=16)
    return replace(
        result,
        tiles=replace(
            result.tiles,
            read_level=1,
            read_tile_size_px=16,
            tile_size_lv0=32,
            level_downsamples=[1.0, 2.0],
            overlap=0.1,
            slide_dimensions=[256, 256],
        ),
        step_px_lv0=step_px_lv0,
    )


@pytest.mark.parametrize("step_px_lv0", [28, 29])
def test_grouped_reads_return_the_pixels_of_individual_reads(step_px_lv0):
    result = _two_x_overlap_result(step_px_lv0)
    reader = _CoordinateCodedReader()

    grouped = {
        record.tile_index: record.tile_arr.copy()
        for record in iter_tile_records_from_reader(reader, result=result)
    }
    individual = {
        record.tile_index: record.tile_arr.copy()
        for record in iter_tile_records_from_reader(reader, result=result, supertile_sizes=())
    }

    assert sorted(grouped) == list(range(64))
    for tile_index in range(64):
        expected_x = int(result.x[tile_index]) // 2
        expected_y = int(result.y[tile_index]) // 2
        # the tile starts at its own floored level-1 origin
        assert individual[tile_index][0, 0].tolist() == [expected_x, expected_y, 120]
        np.testing.assert_array_equal(grouped[tile_index], individual[tile_index])
