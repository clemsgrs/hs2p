"""Dependency-free inspection of a TIFF file's sample formats.

Native display readers (cuCIM, OpenSlide, VIPS, ASAP) decode every TIFF to 8-bit RGB
for display. That conversion is silent and lossy for label rasters stored with any other
sample layout, so :class:`hs2p.mask.Mask` asks here, from the file header alone, whether
a native reader can return the stored values of every directory unchanged. Every
directory matters: a pyramid's reduced levels are separate directories, and a reduced
level read natively is converted exactly like the root.
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
_TAG_SUB_IFDS = 330
_TAG_SAMPLE_FORMAT = 339

# TIFF field type -> byte size and struct code, for the integer field types the sample
# tags and directory offsets may use (BYTE, SHORT, LONG, IFD, LONG8, IFD8).
_INTEGER_TYPE_SIZES = {1: 1, 3: 2, 4: 4, 13: 4, 16: 8, 18: 8}
_INTEGER_TYPE_FORMATS = {1: "B", 3: "H", 4: "I", 13: "I", 16: "Q", 18: "Q"}

# A directory walk stops here even when the offsets keep pointing somewhere new.
_MAX_DIRECTORIES = 4096

SAMPLE_FORMAT_UNSIGNED = 1
SAMPLE_FORMAT_NAMES = {1: "unsigned integer", 2: "signed integer", 3: "float"}
PHOTOMETRIC_MINISWHITE = 0
PHOTOMETRIC_MINISBLACK = 1
PHOTOMETRIC_RGB = 2
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
# The photometrics whose 8-bit unsigned samples a display decode returns unchanged.
# Min-is-white is inverted, palette indices become colors, and every other photometric
# is a color-space conversion.
LOSSLESS_PHOTOMETRICS = frozenset({PHOTOMETRIC_MINISBLACK, PHOTOMETRIC_RGB})


@dataclass(frozen=True)
class TiffSampleFormat:
    """One image directory's sample layout: what one pixel stores on disk.

    ``directory`` is the directory's position in the walk order of
    :func:`read_tiff_sample_formats` (0 is the first directory, a pyramid's root).
    """

    directory: int
    bits_per_sample: tuple[int, ...]
    sample_format: int
    samples_per_pixel: int
    photometric: int

    @property
    def is_lossless_for_display_readers(self) -> bool:
        """True when 8-bit RGB display decoding returns the stored sample values.

        Only 8-bit unsigned min-is-black or RGB samples pass through: any other width
        is rescaled, signed and float samples are converted, min-is-white samples are
        inverted, palette indices are expanded to colors, and the remaining
        photometrics are color-space conversions.
        """
        return (
            all(bits == 8 for bits in self.bits_per_sample)
            and self.sample_format == SAMPLE_FORMAT_UNSIGNED
            and self.photometric in LOSSLESS_PHOTOMETRICS
        )

    def display_conversion(self) -> str:
        """What a display reader does to these samples, as a verb phrase."""
        if any(bits != 8 for bits in self.bits_per_sample):
            return "rescales to 8 bits"
        if self.sample_format != SAMPLE_FORMAT_UNSIGNED:
            return "converts to unsigned 8-bit"
        if self.photometric == PHOTOMETRIC_MINISWHITE:
            return "inverts"
        if self.photometric == PHOTOMETRIC_PALETTE:
            return "expands from palette indices to colors"
        return "converts to RGB"

    def describe(self) -> str:
        bits = "/".join(str(value) for value in self.bits_per_sample)
        photometric = PHOTOMETRIC_NAMES.get(self.photometric, str(self.photometric))
        sample_format = SAMPLE_FORMAT_NAMES.get(
            self.sample_format, str(self.sample_format)
        )
        return (
            f"directory {self.directory} stores {bits}-bit {sample_format} samples, "
            f"{self.samples_per_pixel} per pixel, photometric {photometric}"
        )


def read_tiff_sample_formats(path: str | Path) -> tuple[TiffSampleFormat, ...] | None:
    """Return the sample format of every image directory in ``path``.

    Walks the main directory chain and each directory's SubIFDs, so every level of a
    chained or SubIFD pyramid is included, in file order. Reads only the header and the
    directory entries. Returns ``None`` for a file that is not a TIFF, or whose
    directories cannot be parsed, leaving the reader that opens it to report the failure.
    """
    try:
        with open(path, "rb") as handle:
            return _parse(handle)
    except (OSError, struct.error, ValueError):
        return None


def find_lossy_tiff_directory(path: str | Path) -> TiffSampleFormat | None:
    """Return the first directory of ``path`` a display reader would not return unchanged.

    ``None`` means every directory is lossless for display readers, or the file is not a
    parseable TIFF.
    """
    sample_formats = read_tiff_sample_formats(path)
    if sample_formats is None:
        return None
    for sample_format in sample_formats:
        if not sample_format.is_lossless_for_display_readers:
            return sample_format
    return None


@dataclass(frozen=True)
class _Layout:
    endian: str
    count_format: str
    entry_format: str
    entry_size: int
    inline_capacity: int
    offset_format: str


def _parse(handle) -> tuple[TiffSampleFormat, ...] | None:
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
        layout = _Layout(endian, "H", "HHII", 12, 4, "I")
    elif magic == _BIGTIFF_MAGIC:
        rest = handle.read(8)
        if len(rest) < 8:
            return None
        (offset_size,) = struct.unpack(endian + "H", header[4:6])
        if offset_size != 8:
            return None
        (first_ifd,) = struct.unpack(endian + "Q", rest)
        layout = _Layout(endian, "Q", "HHQQ", 20, 8, "Q")
    else:
        return None

    formats: list[TiffSampleFormat] = []
    seen: set[int] = set()
    # Depth first: a directory's SubIFDs (its reduced levels) come before the next
    # directory of the main chain, matching how pyramid readers order levels.
    pending = [int(first_ifd)]
    while pending and len(formats) < _MAX_DIRECTORIES:
        offset = pending.pop()
        if offset == 0 or offset in seen:
            continue
        seen.add(offset)
        directory = _read_directory(handle, offset, layout)
        if directory is None:
            return None
        values, next_offset, sub_ifds = directory
        formats.append(_sample_format(len(formats), values))
        pending.append(int(next_offset))
        pending.extend(int(sub_ifd) for sub_ifd in reversed(sub_ifds))
    return tuple(formats)


def _read_directory(
    handle, offset: int, layout: _Layout
) -> tuple[dict[int, tuple[int, ...]], int, tuple[int, ...]] | None:
    handle.seek(offset)
    count_size = struct.calcsize(layout.endian + layout.count_format)
    count_bytes = handle.read(count_size)
    if len(count_bytes) < count_size:
        return None
    (entry_count,) = struct.unpack(layout.endian + layout.count_format, count_bytes)
    entries = handle.read(entry_count * layout.entry_size)
    if len(entries) < entry_count * layout.entry_size:
        return None
    offset_size = struct.calcsize(layout.endian + layout.offset_format)
    next_bytes = handle.read(offset_size)
    if len(next_bytes) < offset_size:
        return None
    (next_offset,) = struct.unpack(layout.endian + layout.offset_format, next_bytes)

    values: dict[int, tuple[int, ...]] = {}
    wanted = {
        _TAG_PHOTOMETRIC,
        _TAG_SAMPLES_PER_PIXEL,
        _TAG_BITS_PER_SAMPLE,
        _TAG_SAMPLE_FORMAT,
        _TAG_SUB_IFDS,
    }
    for index in range(entry_count):
        entry = entries[index * layout.entry_size : (index + 1) * layout.entry_size]
        tag, field_type, value_count, raw = struct.unpack(
            layout.endian + layout.entry_format, entry
        )
        if tag not in wanted or field_type not in _INTEGER_TYPE_SIZES:
            continue
        item_size = _INTEGER_TYPE_SIZES[field_type]
        item_format = _INTEGER_TYPE_FORMATS[field_type]
        total = item_size * value_count
        if total <= layout.inline_capacity:
            payload = entry[-layout.inline_capacity :][:total]
        else:
            handle.seek(raw)
            payload = handle.read(total)
            if len(payload) < total:
                return None
        values[tag] = struct.unpack(layout.endian + item_format * value_count, payload)
    return values, int(next_offset), values.get(_TAG_SUB_IFDS, ())


def _sample_format(directory: int, values: dict[int, tuple[int, ...]]) -> TiffSampleFormat:
    samples_per_pixel = values.get(_TAG_SAMPLES_PER_PIXEL, (1,))[0]
    bits_per_sample = values.get(_TAG_BITS_PER_SAMPLE, (1,))
    sample_format = values.get(_TAG_SAMPLE_FORMAT, (SAMPLE_FORMAT_UNSIGNED,))[0]
    photometric = values.get(_TAG_PHOTOMETRIC, (PHOTOMETRIC_MINISBLACK,))[0]
    return TiffSampleFormat(
        directory=directory,
        bits_per_sample=tuple(int(value) for value in bits_per_sample),
        sample_format=int(sample_format),
        samples_per_pixel=int(samples_per_pixel),
        photometric=int(photometric),
    )


__all__ = [
    "LOSSLESS_PHOTOMETRICS",
    "PHOTOMETRIC_MINISBLACK",
    "PHOTOMETRIC_MINISWHITE",
    "PHOTOMETRIC_PALETTE",
    "PHOTOMETRIC_RGB",
    "SAMPLE_FORMAT_UNSIGNED",
    "TiffSampleFormat",
    "find_lossy_tiff_directory",
    "read_tiff_sample_formats",
]
