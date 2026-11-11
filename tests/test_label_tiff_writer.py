"""Writing a pyramidal label TIFF from a stream of tiles, read back through hs2p."""

import json
import tracemalloc

import numpy as np
import pytest

from hs2p import AnnotationLabels, Mask, write_label_tiff
from hs2p.wsi.backends.tifffile import TifffileReader

tifffile = pytest.importorskip("tifffile")


def _row_order_tiles(raster, tile_size, *, edge_fill=None):
    """Tiles of ``raster`` left to right, top to bottom.

    Tiles on the right and bottom edges are cropped to the raster, or, with
    ``edge_fill``, kept at the full tile size with that value past the raster edge.
    """
    height, width = raster.shape
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = raster[y : y + tile_size, x : x + tile_size]
            if edge_fill is not None and tile.shape != (tile_size, tile_size):
                full = np.full((tile_size, tile_size), edge_fill, dtype=raster.dtype)
                full[: tile.shape[0], : tile.shape[1]] = tile
                tile = full
            yield tile


def _random_labels(height, width, *, dtype=np.uint8, seed=0):
    high = 256 if np.dtype(dtype) == np.uint8 else 1000
    return np.random.default_rng(seed).integers(0, high, size=(height, width), dtype=dtype)


def test_base_level_round_trips_through_the_tifffile_reader(tmp_path):
    labels = _random_labels(128, 192)

    path = write_label_tiff(
        path=tmp_path / "labels.tif",
        tiles=_row_order_tiles(labels, 64),
        dimensions=(192, 128),
        tile_size=64,
        spacing_um=0.5,
    )

    with TifffileReader(path) as reader:
        assert reader.dimensions == (192, 128)
        base = reader.read_level(0)
    assert base.dtype == np.uint8
    np.testing.assert_array_equal(base, labels)


def test_each_pyramid_level_is_a_nearest_neighbour_subsample_down_to_one_tile(tmp_path):
    labels = _random_labels(256, 320)

    path = write_label_tiff(
        path=tmp_path / "labels.tif",
        tiles=_row_order_tiles(labels, 32),
        dimensions=(320, 256),
        tile_size=32,
        spacing_um=0.5,
    )

    with TifffileReader(path) as reader:
        # halving until the coarsest level fits in one 32 px tile
        assert reader.level_dimensions == [(320, 256), (160, 128), (80, 64), (40, 32), (20, 16)]
        levels = [reader.read_level(level) for level in range(reader.level_count)]
    for level, array in enumerate(levels):
        step = 2**level
        np.testing.assert_array_equal(array, labels[::step, ::step], err_msg=f"level {level}")


@pytest.mark.parametrize("edge_fill", [None, 7], ids=["cropped-edge-tiles", "full-edge-tiles"])
def test_a_raster_that_is_not_a_multiple_of_the_tile_size_round_trips(tmp_path, edge_fill):
    """Edge tiles may be cropped to the raster or full-size; pixels past the edge are dropped."""
    labels = _random_labels(201, 299)

    path = write_label_tiff(
        path=tmp_path / "labels.tif",
        tiles=_row_order_tiles(labels, 64, edge_fill=edge_fill),
        dimensions=(299, 201),
        tile_size=64,
        spacing_um=0.5,
    )

    with TifffileReader(path) as reader:
        assert reader.level_dimensions == [(299, 201), (150, 101), (75, 51), (38, 26)]
        levels = [reader.read_level(level) for level in range(reader.level_count)]
        # a window across the clipped right and bottom edge tiles
        np.testing.assert_array_equal(
            reader.read_region((250, 150), 0, (49, 51)), labels[150:, 250:]
        )
    for level, array in enumerate(levels):
        step = 2**level
        np.testing.assert_array_equal(array, labels[::step, ::step], err_msg=f"level {level}")


@pytest.mark.parametrize(
    ("dimensions", "expected_levels"),
    [
        ((128, 1), [(128, 1)]),
        ((1, 128), [(1, 128)]),
        ((128, 3), [(128, 3), (64, 2), (32, 1)]),
    ],
)
def test_every_level_of_a_narrow_raster_is_discoverable(tmp_path, dimensions, expected_levels):
    """Halving stops at a 1 px side, which a further level could not reduce."""
    width, height = dimensions
    labels = _random_labels(height, width)

    path = write_label_tiff(
        path=tmp_path / "labels.tif",
        tiles=_row_order_tiles(labels, 16),
        dimensions=dimensions,
        tile_size=16,
        spacing_um=0.5,
    )

    with tifffile.TiffFile(path) as tiff:
        page_count = len(tiff.pages)
    with TifffileReader(path) as reader:
        assert reader.level_dimensions == expected_levels
        assert reader.level_count == page_count
        levels = [reader.read_level(level) for level in range(reader.level_count)]
    for level, array in enumerate(levels):
        step = 2**level
        np.testing.assert_array_equal(array, labels[::step, ::step], err_msg=f"level {level}")


def test_a_tile_buffer_the_stream_refills_is_written_as_each_tile_was_yielded(tmp_path):
    """256 px tiles are large enough for tifffile to encode them on worker threads."""
    labels = _random_labels(512, 768)
    buffer = np.empty((256, 256), dtype=np.uint8)

    def refilled_tiles():
        for tile in _row_order_tiles(labels, 256):
            buffer[...] = tile
            yield buffer

    path = write_label_tiff(
        path=tmp_path / "labels.tif",
        tiles=refilled_tiles(),
        dimensions=(768, 512),
        tile_size=256,
        spacing_um=0.5,
    )

    with TifffileReader(path) as reader:
        levels = [reader.read_level(level) for level in range(reader.level_count)]
    assert len(levels) == 3
    for level, array in enumerate(levels):
        step = 2**level
        np.testing.assert_array_equal(array, labels[::step, ::step], err_msg=f"level {level}")


def test_a_tile_buffer_the_stream_clears_after_its_last_yield_is_written_as_yielded(
    tmp_path,
):
    """The writer asks for one tile past the grid, which resumes the stream's cleanup."""
    labels = _random_labels(64, 64)
    buffer = np.empty((32, 32), dtype=np.uint8)

    def refilled_then_cleared_tiles():
        try:
            for tile in _row_order_tiles(labels, 32):
                buffer[...] = tile
                yield buffer
        finally:
            buffer.fill(0)

    path = write_label_tiff(
        path=tmp_path / "labels.tif",
        tiles=refilled_then_cleared_tiles(),
        dimensions=(64, 64),
        tile_size=32,
        spacing_um=0.5,
    )

    with TifffileReader(path) as reader:
        levels = [reader.read_level(level) for level in range(reader.level_count)]
    assert len(levels) == 2
    np.testing.assert_array_equal(levels[0], labels, err_msg="level 0")
    np.testing.assert_array_equal(levels[1], labels[::2, ::2], err_msg="level 1")


def _write(tmp_path, labels, *, tile_size=32, spacing_um=0.5, name="labels.tif", **kwargs):
    height, width = labels.shape
    return write_label_tiff(
        path=tmp_path / name,
        tiles=_row_order_tiles(labels, tile_size),
        dimensions=(width, height),
        tile_size=tile_size,
        spacing_um=spacing_um,
        **kwargs,
    )


@pytest.mark.parametrize("spacing_um", [0.2425, 0.5, 8.0])
def test_the_reader_reports_the_written_spacing_at_every_level(tmp_path, spacing_um):
    path = _write(tmp_path, _random_labels(64, 128), spacing_um=spacing_um)

    with TifffileReader(path) as reader:
        assert reader.native_spacing == pytest.approx(spacing_um, rel=1e-9)
        assert reader.spacings == pytest.approx([spacing_um, 2 * spacing_um, 4 * spacing_um])
    with tifffile.TiffFile(path) as tiff:
        # each reduced level carries its own resolution tags, so readers that take the
        # spacing per level agree with the base level
        page_spacings = [10000.0 / page.resolution[0] for page in tiff.pages]
    assert page_spacings == pytest.approx([spacing_um, 2 * spacing_um, 4 * spacing_um])


@pytest.mark.parametrize(
    "metadata",
    [
        {"labels": {"0": "background", "1": "tumor", "2": "stroma"}, "model": "seg-v1"},
        # tifffile reads a JSON description holding a "shape" key as its own array
        # metadata; the written file must still open as the written pyramid
        {"shape": "irregular", "model": {"input": {"shape": [512, 512]}}, "note": 'say "shape":'},
    ],
    ids=["label-mapping", "shape-keys"],
)
def test_the_metadata_dict_round_trips_in_the_image_description(tmp_path, metadata):
    labels = _random_labels(64, 128)
    path = _write(tmp_path, labels, metadata=metadata)

    with tifffile.TiffFile(path) as tiff:
        description = tiff.pages[0].description
    assert json.loads(description) == metadata
    with TifffileReader(path) as reader:
        assert reader.level_dimensions == [(128, 64), (64, 32), (32, 16)]
        np.testing.assert_array_equal(reader.read_level(0), labels)


def test_without_metadata_no_image_description_is_written(tmp_path):
    path = _write(tmp_path, _random_labels(64, 128))

    with tifffile.TiffFile(path) as tiff:
        assert tiff.pages[0].description == ""


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize(
    "compression, tag",
    [("deflate", tifffile.COMPRESSION.ADOBE_DEFLATE), ("lzw", tifffile.COMPRESSION.LZW)],
)
def test_unsigned_rasters_round_trip_under_each_lossless_compression(
    tmp_path, dtype, compression, tag
):
    labels = _random_labels(64, 96, dtype=dtype)
    path = _write(tmp_path, labels, dtype=dtype, compression=compression)

    with tifffile.TiffFile(path) as tiff:
        assert [page.compression for page in tiff.pages] == [tag, tag, tag]
        assert {page.photometric for page in tiff.pages} == {tifffile.PHOTOMETRIC.MINISBLACK}
    with TifffileReader(path) as reader:
        for level in range(reader.level_count):
            array = reader.read_level(level)
            assert array.dtype == dtype
            np.testing.assert_array_equal(array, labels[:: 2**level, :: 2**level])


def test_missing_and_short_tiles_are_written_as_zeros(tmp_path):
    """A 96x64 raster in 32 px tiles: tile 1 is missing and tile 4 is short."""
    stream = [np.full((32, 32), value, dtype=np.uint8) for value in (1, 2, 3, 4, 5, 6)]
    stream[1] = None
    stream[4] = np.full((20, 10), 5, dtype=np.uint8)
    expected = np.array(
        [[1] * 32 + [0] * 32 + [3] * 32] * 32
        + [[4] * 32 + [5] * 10 + [0] * 22 + [6] * 32] * 20
        + [[4] * 32 + [0] * 32 + [6] * 32] * 12,
        dtype=np.uint8,
    )

    path = write_label_tiff(
        path=tmp_path / "labels.tif",
        tiles=iter(stream),
        dimensions=(96, 64),
        tile_size=32,
        spacing_um=0.5,
    )

    with TifffileReader(path) as reader:
        levels = [reader.read_level(level) for level in range(reader.level_count)]
    assert len(levels) == 3
    for level, array in enumerate(levels):
        np.testing.assert_array_equal(array, expected[:: 2**level, :: 2**level])


class _InferenceCrashed(Exception):
    pass


def test_a_write_interrupted_by_an_exception_leaves_no_file(tmp_path):
    labels = _random_labels(64, 128)

    def crashing_tiles():
        tiles = _row_order_tiles(labels, 32)
        for _ in range(5):
            yield next(tiles)
        raise _InferenceCrashed("GPU fell over")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(_InferenceCrashed):
        write_label_tiff(
            path=out_dir / "labels.tif",
            tiles=crashing_tiles(),
            dimensions=(128, 64),
            tile_size=32,
            spacing_um=0.5,
        )

    assert not (out_dir / "labels.tif").exists()
    assert list(out_dir.iterdir()) == []


def test_a_completed_write_replaces_the_previous_file_only_when_done(tmp_path):
    first = _random_labels(64, 128, seed=1)
    path = _write(tmp_path, first)

    def crashing_tiles():
        yield np.zeros((32, 32), dtype=np.uint8)
        raise _InferenceCrashed("GPU fell over")

    with pytest.raises(_InferenceCrashed):
        write_label_tiff(
            path=path, tiles=crashing_tiles(), dimensions=(128, 64), tile_size=32, spacing_um=0.5
        )
    with TifffileReader(path) as reader:
        np.testing.assert_array_equal(reader.read_level(0), first)

    second = _random_labels(64, 128, seed=2)
    _write(tmp_path, second)
    with TifffileReader(path) as reader:
        np.testing.assert_array_equal(reader.read_level(0), second)
    assert sorted(child.name for child in tmp_path.iterdir()) == ["labels.tif"]


@pytest.mark.parametrize(
    "count, message",
    [(7, "ended after 7 of the 8 tiles"), (9, "more than the 8 tiles")],
    ids=["short-stream", "long-stream"],
)
def test_a_stream_with_the_wrong_tile_count_fails_and_leaves_no_file(tmp_path, count, message):
    """A 128x64 raster in 32 px tiles is a 4x2 grid of 8 tiles."""
    stream = (np.zeros((32, 32), dtype=np.uint8) for _ in range(count))

    with pytest.raises(ValueError, match=message):
        write_label_tiff(
            path=tmp_path / "labels.tif",
            tiles=stream,
            dimensions=(128, 64),
            tile_size=32,
            spacing_um=0.5,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad_tile, message",
    [
        # an argmax output: casting it would silently wrap label 300 to 44
        (np.full((32, 32), 300, dtype=np.int64), "dtype int64, expected uint8"),
        (np.zeros((32, 32, 3), dtype=np.uint8), r"shape \(32, 32, 3\)"),
        (np.zeros((32, 48), dtype=np.uint8), r"shape \(32, 48\)"),
    ],
    ids=["wrong-dtype", "multichannel", "larger-than-a-tile"],
)
def test_a_tile_that_cannot_be_stored_unchanged_fails_and_leaves_no_file(
    tmp_path, bad_tile, message
):
    stream = [np.zeros((32, 32), dtype=np.uint8) for _ in range(8)]
    stream[5] = bad_tile

    with pytest.raises(ValueError, match=rf"tile 5 .*{message}"):
        write_label_tiff(
            path=tmp_path / "labels.tif",
            tiles=stream,
            dimensions=(128, 64),
            tile_size=32,
            spacing_um=0.5,
        )
    assert list(tmp_path.iterdir()) == []


def test_peak_memory_while_writing_stays_far_below_the_raw_raster_size(tmp_path):
    """A 16384 px square uint8 raster is 256 MiB raw. The writer holds the 2x level (a
    quarter of that) and a bounded number of tiles in flight, never the base level.

    The tiles are incompressible and freshly allocated, like inference output: a tile
    held anywhere, before or after encoding, costs its full size."""
    size, tile_size = 16384, 512
    raw_bytes = size * size
    columns = size // tile_size

    def tile_at(index):
        rng = np.random.default_rng(index)
        return rng.integers(0, 256, size=(tile_size, tile_size), dtype=np.uint8)

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        path = write_label_tiff(
            path=tmp_path / "large.tif",
            tiles=(tile_at(index) for index in range(columns * columns)),
            dimensions=(size, size),
            tile_size=tile_size,
            spacing_um=0.5,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_bytes < raw_bytes / 2, f"peak {peak_bytes / 2**20:.0f} MiB of 256 MiB raw"
    with TifffileReader(path) as reader:
        assert reader.level_count == 6
        tile = reader.read_region((5 * tile_size, 2 * tile_size), 0, (tile_size, tile_size))
    np.testing.assert_array_equal(tile, tile_at(2 * columns + 5))


_READER_MODULES = {"cucim": "cucim", "openslide": "openslide", "asap": "wholeslidedata"}


@pytest.mark.parametrize("backend", ["tifffile", "cucim", "openslide", "asap"])
def test_a_written_label_map_reads_back_as_a_mask_aligned_to_its_slide(tmp_path, backend):
    """A 0.5 um/px label map of an 800x600 slide scanned at 0.25 um/px, written with the
    default settings, reads back through every installed mask reader."""
    if backend in _READER_MODULES:
        pytest.importorskip(_READER_MODULES[backend])
    labels = np.zeros((300, 400), dtype=np.uint8)
    labels[50:150, 100:300] = 1
    labels[200:280, 20:120] = 2
    mapping = {"background": 0, "tumor": 1, "stroma": 2}
    path = write_label_tiff(
        path=tmp_path / "slide-1-labels.tif",
        tiles=_row_order_tiles(labels, 64),
        dimensions=(400, 300),
        tile_size=64,
        spacing_um=0.5,
        metadata={"labels": mapping},
    )

    with Mask(
        path=path, labels=AnnotationLabels(pixel_mapping=mapping), backend=backend
    ) as mask:
        assert mask.backend == backend
        aligned = mask.align_to(reference_spacing_um=0.25, reference_dimensions=(800, 600))
        full = aligned.read_full(target_spacing_um=0.5, target_dimensions=(400, 300))
        # slide level-0 (200, 100) is (50, 25) on the 1.0 um/px grid of the mask's level 1
        region = aligned.read_region(
            location=(200, 100), target_spacing_um=1.0, target_dimensions=(50, 40)
        )

    np.testing.assert_array_equal(full.labels, labels)
    assert full.read_level == 0
    np.testing.assert_array_equal(region.labels, labels[::2, ::2][25:65, 50:100])
    assert region.read_level == 1


@pytest.mark.parametrize(
    "argument, value, error, message",
    [
        ("tile_size", 50, ValueError, "tile_size must be a positive multiple of 16, got 50"),
        ("tile_size", 0, ValueError, "tile_size must be a positive multiple of 16, got 0"),
        ("dimensions", (128, 0), ValueError, r"dimensions must be two positive .*\(128, 0\)"),
        ("spacing_um", 0.0, ValueError, "spacing_um must be finite and positive, got 0.0"),
        ("spacing_um", float("nan"), ValueError, "spacing_um must be finite and positive"),
        ("dtype", np.int16, ValueError, "dtype must be an unsigned integer type, got int16"),
        ("dtype", np.float32, ValueError, "dtype must be an unsigned integer type, got float32"),
        ("compression", "jpeg", ValueError, r"compression must be one of \['lzw', 'deflate'\]"),
        ("metadata", {"model": object()}, TypeError, "not JSON serializable"),
        ("metadata", [("labels", 1)], TypeError, "metadata must be a mapping"),
    ],
)
def test_invalid_arguments_fail_before_anything_is_written(
    tmp_path, argument, value, error, message
):
    arguments = dict(
        path=tmp_path / "labels.tif",
        tiles=(np.zeros((32, 32), dtype=np.uint8) for _ in range(8)),
        dimensions=(128, 64),
        tile_size=32,
        spacing_um=0.5,
    )
    arguments[argument] = value

    with pytest.raises(error, match=message):
        write_label_tiff(**arguments)
    assert list(tmp_path.iterdir()) == []
