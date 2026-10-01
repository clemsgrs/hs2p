import struct

import numpy as np
import pytest

from hs2p.wsi.tiff_header import (
    TiffSampleFormat,
    find_lossy_tiff_directory,
    read_tiff_sample_formats,
)

tifffile = pytest.importorskip("tifffile")


def _write(path, array, **kwargs):
    tifffile.imwrite(path, array, **kwargs)
    return path


def _write_mixed_width_pyramid(path, *, chained, **kwargs):
    """A uint8 64x64 root with a uint16 32x32 reduced level, chained or as a SubIFD."""
    with tifffile.TiffWriter(path, **kwargs) as writer:
        writer.write(
            np.ones((64, 64), dtype=np.uint8),
            tile=(16, 16),
            photometric="minisblack",
            subifds=0 if chained else 1,
        )
        writer.write(
            np.ones((32, 32), dtype=np.uint16),
            tile=(16, 16),
            photometric="minisblack",
            subfiletype=1,
        )
    return path


@pytest.mark.parametrize("bigtiff", [False, True])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_8bit_unsigned_minisblack_is_lossless_for_display_readers(
    tmp_path, bigtiff, byteorder
):
    path = _write(
        tmp_path / "mask.tif",
        np.zeros((16, 16), dtype=np.uint8),
        bigtiff=bigtiff,
        byteorder=byteorder,
        photometric="minisblack",
    )

    sample_formats = read_tiff_sample_formats(path)

    assert sample_formats == (
        TiffSampleFormat(
            directory=0,
            bits_per_sample=(8,),
            sample_format=1,
            samples_per_pixel=1,
            photometric=1,
        ),
    )
    assert sample_formats[0].is_lossless_for_display_readers
    assert find_lossy_tiff_directory(path) is None


def test_8bit_rgb_is_lossless_for_display_readers(tmp_path):
    path = _write(
        tmp_path / "rgb.tif", np.zeros((16, 16, 3), dtype=np.uint8), photometric="rgb"
    )

    (sample_format,) = read_tiff_sample_formats(path)

    assert sample_format.bits_per_sample == (8, 8, 8)
    assert sample_format.samples_per_pixel == 3
    assert sample_format.is_lossless_for_display_readers
    assert find_lossy_tiff_directory(path) is None


@pytest.mark.parametrize(
    ("array", "kwargs", "expected_bits", "expected_format", "conversion"),
    [
        (np.zeros((16, 16), dtype=np.uint16), {}, (16,), 1, "rescales to 8 bits"),
        (np.zeros((16, 16), dtype=np.int8), {}, (8,), 2, "converts to unsigned 8-bit"),
        (np.zeros((16, 16), dtype=np.float32), {}, (32,), 3, "rescales to 8 bits"),
        (
            np.zeros((16, 16), dtype=np.uint16),
            {"bigtiff": True},
            (16,),
            1,
            "rescales to 8 bits",
        ),
    ],
)
def test_non_8bit_unsigned_samples_are_lossy_for_display_readers(
    tmp_path, array, kwargs, expected_bits, expected_format, conversion
):
    path = _write(tmp_path / "mask.tif", array, photometric="minisblack", **kwargs)

    lossy = find_lossy_tiff_directory(path)

    assert lossy.directory == 0
    assert lossy.bits_per_sample == expected_bits
    assert lossy.sample_format == expected_format
    assert not lossy.is_lossless_for_display_readers
    assert lossy.display_conversion() == conversion


def test_min_is_white_samples_are_lossy_for_display_readers(tmp_path):
    # A display decode inverts min-is-white: stored 255 comes back as 0.
    path = _write(
        tmp_path / "inverted.tif",
        np.full((16, 16), 255, dtype=np.uint8),
        photometric="miniswhite",
    )

    lossy = find_lossy_tiff_directory(path)

    assert lossy.photometric == 0
    assert lossy.bits_per_sample == (8,)
    assert not lossy.is_lossless_for_display_readers
    assert lossy.display_conversion() == "inverts"
    assert "photometric min-is-white" in lossy.describe()


def test_palette_indices_are_lossy_for_display_readers(tmp_path):
    colormap = np.zeros((3, 256), dtype=np.uint16)
    path = _write(
        tmp_path / "palette.tif",
        np.zeros((16, 16), dtype=np.uint8),
        photometric="palette",
        colormap=colormap,
    )

    lossy = find_lossy_tiff_directory(path)

    assert lossy.photometric == 3
    assert not lossy.is_lossless_for_display_readers
    assert lossy.display_conversion() == "expands from palette indices to colors"
    assert "palette" in lossy.describe()


@pytest.mark.parametrize("bigtiff", [False, True])
@pytest.mark.parametrize("chained", [False, True])
def test_every_pyramid_directory_is_checked(tmp_path, bigtiff, chained):
    path = _write_mixed_width_pyramid(
        tmp_path / "mixed.tif", chained=chained, bigtiff=bigtiff
    )

    sample_formats = read_tiff_sample_formats(path)
    lossy = find_lossy_tiff_directory(path)

    assert [fmt.directory for fmt in sample_formats] == [0, 1]
    assert [fmt.bits_per_sample for fmt in sample_formats] == [(8,), (16,)]
    assert sample_formats[0].is_lossless_for_display_readers
    assert lossy == sample_formats[1]
    assert lossy.describe().startswith("directory 1 stores 16-bit unsigned integer")


def test_uniform_pyramid_directories_are_all_lossless(tmp_path):
    with tifffile.TiffWriter(tmp_path / "pyramid.tif") as writer:
        writer.write(
            np.ones((64, 64), dtype=np.uint8),
            tile=(16, 16),
            photometric="minisblack",
            subifds=2,
        )
        for step in (2, 4):
            writer.write(
                np.ones((64 // step, 64 // step), dtype=np.uint8),
                tile=(16, 16),
                photometric="minisblack",
                subfiletype=1,
            )

    sample_formats = read_tiff_sample_formats(tmp_path / "pyramid.tif")

    assert [fmt.directory for fmt in sample_formats] == [0, 1, 2]
    assert all(fmt.is_lossless_for_display_readers for fmt in sample_formats)
    assert find_lossy_tiff_directory(tmp_path / "pyramid.tif") is None


def test_a_directory_chain_cycle_terminates(tmp_path):
    path = _write(
        tmp_path / "cycle.tif", np.zeros((16, 16), dtype=np.uint8), photometric="minisblack"
    )
    data = bytearray(path.read_bytes())
    (first_ifd,) = struct.unpack("<I", data[4:8])
    (entry_count,) = struct.unpack("<H", data[first_ifd : first_ifd + 2])
    next_at = first_ifd + 2 + entry_count * 12
    data[next_at : next_at + 4] = struct.pack("<I", first_ifd)
    path.write_bytes(bytes(data))

    sample_formats = read_tiff_sample_formats(path)

    assert len(sample_formats) == 1
    assert find_lossy_tiff_directory(path) is None


def test_non_tiff_and_truncated_files_return_none(tmp_path):
    png = tmp_path / "mask.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    truncated = tmp_path / "truncated.tif"
    truncated.write_bytes(b"II*\x00\x08\x00\x00\x00")
    missing = tmp_path / "missing.tif"

    for path in (png, truncated, missing):
        assert read_tiff_sample_formats(path) is None
        assert find_lossy_tiff_directory(path) is None
