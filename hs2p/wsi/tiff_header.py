"""Dependency-free inspection of a TIFF file's sample format.

Native display readers (cuCIM, OpenSlide, VIPS, ASAP) decode every TIFF to 8-bit RGB
for display. That conversion is silent and lossy for label rasters stored with any other
sample format, so :class:`hs2p.mask.Mask` asks here, from the file header alone, whether
a native reader can return the stored values unchanged.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

_CLASSIC_MAGIC = 42
_BIGTIFF_MAGIC = 43

_TAG_PHOTOMETRIC = 262
_TAG_SAMPLES_PER_PIXEL = 277
_TAG_BITS_PER_SAMPLE = 258
_TAG_SAMPLE_FORMAT = 339

# TIFF field type -> byte size, for the integer field types the sample tags may use.
_INTEGER_TYPE_SIZES = {1: 1, 3: 2, 4: 4, 16: 8}
_INTEGER_TYPE_FORMATS = {1: "B", 3: "H", 4: "I", 16: "Q"}

SAMPLE_FORMAT_UNSIGNED = 1
SAMPLE_FORMAT_NAMES = {1: "unsigned integer", 2: "signed integer", 3: "float"}
PHOTOMETRIC_PALETTE = 3
PHOTOMETRIC_NAMES = {
    0: "min-is-white",
    1: "min-is-black",
    2: "RGB",
    3: "palette",
    4: "transparency mask",
    5: "CMYK",
    6: "YCbCr",
}


@dataclass(frozen=True)
class TiffSampleFormat:
    """The first image directory's sample layout: what one pixel stores on disk."""

    bits_per_sample: tuple[int, ...]
    sample_format: int
    samples_per_pixel: int
    photometric: int

    @property
    def is_lossless_for_display_readers(self) -> bool:
        """True when 8-bit RGB display decoding returns the stored sample values.

        Any width other than 8 bits is rescaled, non-unsigned formats are converted,
        and palette indices are expanded to their colors.
        """
        return (
            all(bits == 8 for bits in self.bits_per_sample)
            and self.sample_format == SAMPLE_FORMAT_UNSIGNED
            and self.photometric != PHOTOMETRIC_PALETTE
        )

    def describe(self) -> str:
        bits = "/".join(str(value) for value in self.bits_per_sample)
        photometric = PHOTOMETRIC_NAMES.get(self.photometric, str(self.photometric))
        sample_format = SAMPLE_FORMAT_NAMES.get(
            self.sample_format, str(self.sample_format)
        )
        return (
            f"{bits}-bit {sample_format} samples, {self.samples_per_pixel} per pixel, "
            f"photometric {photometric}"
        )


def read_tiff_sample_format(path: str | Path) -> TiffSampleFormat | None:
    """Return the sample format of ``path``'s first image directory.

    Reads only the header and one directory of a classic or BigTIFF file. Returns
    ``None`` for a file that is not a TIFF, or whose header cannot be parsed, leaving the
    reader that opens it to report the failure.
    """
    try:
        with open(path, "rb") as handle:
            return _parse(handle)
    except (OSError, struct.error, ValueError):
        return None


def _parse(handle) -> TiffSampleFormat | None:
    header = handle.read(8)
    if len(header) < 8:
        return None
    byte_order = header[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return None
    (magic,) = struct.unpack(endian + "H", header[2:4])
    if magic == _CLASSIC_MAGIC:
        (first_ifd,) = struct.unpack(endian + "I", header[4:8])
        count_format, entry_format, entry_size, inline_capacity = "H", "HHII", 12, 4
    elif magic == _BIGTIFF_MAGIC:
        rest = handle.read(8)
        if len(rest) < 8:
            return None
        (offset_size,) = struct.unpack(endian + "H", header[4:6])
        if offset_size != 8:
            return None
        (first_ifd,) = struct.unpack(endian + "Q", rest)
        count_format, entry_format, entry_size, inline_capacity = "Q", "HHQQ", 20, 8
    else:
        return None

    handle.seek(first_ifd)
    count_size = struct.calcsize(endian + count_format)
    (entry_count,) = struct.unpack(endian + count_format, handle.read(count_size))
    entries = handle.read(entry_count * entry_size)
    if len(entries) < entry_count * entry_size:
        return None

    values: dict[int, tuple[int, ...]] = {}
    wanted = {
        _TAG_PHOTOMETRIC,
        _TAG_SAMPLES_PER_PIXEL,
        _TAG_BITS_PER_SAMPLE,
        _TAG_SAMPLE_FORMAT,
    }
    for index in range(entry_count):
        entry = entries[index * entry_size : (index + 1) * entry_size]
        tag, field_type, value_count, raw = struct.unpack(endian + entry_format, entry)
        if tag not in wanted or field_type not in _INTEGER_TYPE_SIZES:
            continue
        item_size = _INTEGER_TYPE_SIZES[field_type]
        item_format = _INTEGER_TYPE_FORMATS[field_type]
        total = item_size * value_count
        if total <= inline_capacity:
            payload = entry[-inline_capacity:][:total]
        else:
            handle.seek(raw)
            payload = handle.read(total)
            if len(payload) < total:
                return None
        values[tag] = struct.unpack(endian + item_format * value_count, payload)

    samples_per_pixel = values.get(_TAG_SAMPLES_PER_PIXEL, (1,))[0]
    bits_per_sample = values.get(_TAG_BITS_PER_SAMPLE, (1,))
    sample_format = values.get(_TAG_SAMPLE_FORMAT, (SAMPLE_FORMAT_UNSIGNED,))[0]
    photometric = values.get(_TAG_PHOTOMETRIC, (1,))[0]
    return TiffSampleFormat(
        bits_per_sample=tuple(int(value) for value in bits_per_sample),
        sample_format=int(sample_format),
        samples_per_pixel=int(samples_per_pixel),
        photometric=int(photometric),
    )


__all__ = [
    "PHOTOMETRIC_PALETTE",
    "SAMPLE_FORMAT_UNSIGNED",
    "TiffSampleFormat",
    "read_tiff_sample_format",
]
