from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

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


def test_grouped_records_above_level_0_keep_the_saved_coordinates():
    from dataclasses import replace

    # a 2x2 grid read at a 2x level: 16 px tiles are 32 level-0 px apart
    result = _make_result(coords=[(0, 0), (0, 32), (32, 0), (32, 32)], tile_size=16)
    result = replace(
        result,
        tiles=replace(
            result.tiles, read_level=1, tile_size_lv0=32, level_downsamples=[1.0, 2.0]
        ),
        step_px_lv0=32,
    )
    reader = _mock_reader(np.zeros((32, 32, 3), dtype=np.uint8))

    records = list(iter_tile_records_from_reader(reader, result=result))

    reader.read_region.assert_called_once_with((0, 0), 1, (32, 32))
    assert [(record.tile_index, record.x, record.y) for record in records] == [
        (0, 0, 0),
        (1, 0, 32),
        (2, 32, 0),
        (3, 32, 32),
    ]
