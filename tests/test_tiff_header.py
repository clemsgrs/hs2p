import numpy as np
import pytest

from hs2p.wsi.tiff_header import TiffSampleFormat, read_tiff_sample_format

tifffile = pytest.importorskip("tifffile")


def _write(path, array, **kwargs):
    tifffile.imwrite(path, array, **kwargs)
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

    sample_format = read_tiff_sample_format(path)

    assert sample_format == TiffSampleFormat(
        bits_per_sample=(8,), sample_format=1, samples_per_pixel=1, photometric=1
    )
    assert sample_format.is_lossless_for_display_readers


def test_8bit_rgb_is_lossless_for_display_readers(tmp_path):
    path = _write(
        tmp_path / "rgb.tif", np.zeros((16, 16, 3), dtype=np.uint8), photometric="rgb"
    )

    sample_format = read_tiff_sample_format(path)

    assert sample_format.bits_per_sample == (8, 8, 8)
    assert sample_format.samples_per_pixel == 3
    assert sample_format.is_lossless_for_display_readers


@pytest.mark.parametrize(
    ("array", "kwargs", "expected_bits", "expected_format"),
    [
        (np.zeros((16, 16), dtype=np.uint16), {}, (16,), 1),
        (np.zeros((16, 16), dtype=np.int8), {}, (8,), 2),
        (np.zeros((16, 16), dtype=np.float32), {}, (32,), 3),
        (np.zeros((16, 16), dtype=np.uint16), {"bigtiff": True}, (16,), 1),
    ],
)
def test_non_8bit_unsigned_samples_are_lossy_for_display_readers(
    tmp_path, array, kwargs, expected_bits, expected_format
):
    path = _write(tmp_path / "mask.tif", array, photometric="minisblack", **kwargs)

    sample_format = read_tiff_sample_format(path)

    assert sample_format.bits_per_sample == expected_bits
    assert sample_format.sample_format == expected_format
    assert not sample_format.is_lossless_for_display_readers


def test_palette_indices_are_lossy_for_display_readers(tmp_path):
    colormap = np.zeros((3, 256), dtype=np.uint16)
    path = _write(
        tmp_path / "palette.tif",
        np.zeros((16, 16), dtype=np.uint8),
        photometric="palette",
        colormap=colormap,
    )

    sample_format = read_tiff_sample_format(path)

    assert sample_format.photometric == 3
    assert not sample_format.is_lossless_for_display_readers
    assert "palette" in sample_format.describe()


def test_non_tiff_and_truncated_files_return_none(tmp_path):
    png = tmp_path / "mask.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    truncated = tmp_path / "truncated.tif"
    truncated.write_bytes(b"II*\x00\x08\x00\x00\x00")
    missing = tmp_path / "missing.tif"

    assert read_tiff_sample_format(png) is None
    assert read_tiff_sample_format(truncated) is None
    assert read_tiff_sample_format(missing) is None
