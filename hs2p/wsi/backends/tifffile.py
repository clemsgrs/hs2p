"""Lossless TIFF reader: returns stored sample values in their stored dtype.

The display readers (cuCIM, OpenSlide, VIPS, ASAP) decode to 8-bit RGB. This reader
exists for label rasters, where the stored integer is the label: it returns 2-D arrays
for single-sample pages and ``(height, width, samples)`` arrays otherwise, never
rescaling, expanding a palette or converting the dtype. Out-of-canvas padding is 255,
like the PIL reader, so a padded label read fails validation unless 255 is declared.

:func:`write_label_tiff` is the writing side: a pyramidal label TIFF streamed from tiles,
which this reader returns unchanged at every level.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import IO, Any, Literal

import numpy as np
from numpy.typing import DTypeLike

from hs2p.fileops import promote_temp_file
from hs2p.wsi.backends.common import (
    paste_region,
    resolve_level0_spacing,
    resolve_padded_read_bounds,
)
from hs2p.wsi.geometry import compute_level_spacings

TIFFFILE_SUPPORTED_SUFFIXES = frozenset({".tif", ".tiff", ".btf", ".tf8"})
PADDING_VALUE = 255
_PLANAR_CONTIG = 1
_RESUNIT_INCH = 2
_RESUNIT_CENTIMETER = 3
_UM_PER_INCH = 25400.0
_UM_PER_CENTIMETER = 10000.0
# The lossless codecs a label TIFF may use, by the name the writer accepts.
_LABEL_TIFF_COMPRESSIONS = {"lzw": "lzw", "deflate": "zlib"}
_STREAM_END = object()
# tifffile takes up to this many bytes of tiles from the stream at once to encode them in
# parallel (at least one tile per worker). Its default, 512 MiB, holds most of a large
# raster: every pending tile and every encoded one not yet written.
_ENCODE_BUFFER_BYTES = 8 * 2**20


def supports_tifffile_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TIFFFILE_SUPPORTED_SUFFIXES


def _spacing_from_page(page: Any) -> float | None:
    """Level-0 spacing in um/px from the resolution tags, or ``None`` when untagged."""
    try:
        unit = int(page.resolutionunit)
        pixels_per_unit = float(page.resolution[0])
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    if unit == _RESUNIT_INCH:
        um_per_unit = _UM_PER_INCH
    elif unit == _RESUNIT_CENTIMETER:
        um_per_unit = _UM_PER_CENTIMETER
    else:
        return None
    if not math.isfinite(pixels_per_unit) or pixels_per_unit <= 0:
        return None
    return um_per_unit / pixels_per_unit


class TifffileReader:
    def __init__(
        self,
        path: str | Path,
        *,
        spacing_override: float | None = None,
        require_spacing: bool = True,
    ):
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError(
                "tifffile is required for the tifffile backend. "
                "Install it with: pip install 'hs2p[tifffile]'"
            ) from exc

        self._path = str(path)
        self._lock = RLock()
        self._tiff = tifffile.TiffFile(self._path)
        try:
            self._levels = self._pyramid_levels()
            self._pages = [level.keyframe for level in self._levels]
            self._level_dimensions = [
                (int(page.imagewidth), int(page.imagelength)) for page in self._pages
            ]
            width0, height0 = self._level_dimensions[0]
            self._level_downsamples = [
                (width0 / width, height0 / height)
                for width, height in self._level_dimensions
            ]
            self.native_spacing = _spacing_from_page(self._pages[0])
            self._spacing = resolve_level0_spacing(
                path=self._path,
                backend=self.backend_name,
                native_spacing=self.native_spacing,
                spacing_override=spacing_override,
                require_spacing=require_spacing,
            )
            self._spacings = (
                []
                if self._spacing is None
                else compute_level_spacings(
                    level0_spacing_um=self._spacing,
                    level_downsamples=self._level_downsamples,
                )
            )
        except Exception:
            self._tiff.close()
            raise

    def _pyramid_levels(self) -> list[Any]:
        if not self._tiff.series:
            raise ValueError(f"tifffile found no image series in path={self._path}")
        levels = list(self._tiff.series[0].levels)
        for level in levels:
            page = level.keyframe
            if page is None:
                raise ValueError(
                    f"tifffile backend cannot read a level without a page in path={self._path}"
                )
            if int(page.imagedepth) != 1:
                raise ValueError(
                    f"tifffile backend does not support volumetric pages in path={self._path}"
                )
            if int(page.samplesperpixel) > 1 and int(page.planarconfig) != _PLANAR_CONTIG:
                raise ValueError(
                    "tifffile backend does not support planar-separate samples in "
                    f"path={self._path}"
                )
        widths = [int(level.keyframe.imagewidth) for level in levels]
        if widths != sorted(widths, reverse=True):
            raise ValueError(
                f"tifffile backend expects pyramid levels ordered fine to coarse in path={self._path}, "
                f"got widths {widths}"
            )
        return levels

    @property
    def backend_name(self) -> str:
        return "tifffile"

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._level_dimensions[0]

    @property
    def spacing(self) -> float:
        return self._spacing

    @property
    def spacings(self) -> list[float]:
        return list(self._spacings)

    @property
    def level_count(self) -> int:
        return len(self._levels)

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return list(self._level_dimensions)

    @property
    def level_downsamples(self) -> list[tuple[float, float]]:
        return list(self._level_downsamples)

    def _samples(self, level: int) -> int:
        return int(self._pages[level].samplesperpixel)

    def _canvas(self, *, width: int, height: int, level: int) -> np.ndarray:
        samples = self._samples(level)
        shape = (height, width) if samples == 1 else (height, width, samples)
        return np.full(shape, PADDING_VALUE, dtype=self._pages[level].dtype)

    def read_level(self, level: int) -> np.ndarray:
        with self._lock:
            array = np.asarray(self._levels[level].asarray())
        return self._normalized(array, level)

    def _normalized(self, array: np.ndarray, level: int) -> np.ndarray:
        width, height = self._level_dimensions[level]
        samples = self._samples(level)
        shape = (height, width) if samples == 1 else (height, width, samples)
        return array.reshape(shape)

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        level = int(level)
        bounds = resolve_padded_read_bounds(
            location=location,
            size=size,
            level_dimensions=self._level_dimensions[level],
            downsample=float(self._level_downsamples[level][0]),
        )
        canvas = self._canvas(width=int(size[0]), height=int(size[1]), level=level)
        read_width, read_height = bounds.read_size
        if read_width <= 0 or read_height <= 0:
            return canvas
        # ``resolve_padded_read_bounds`` returns the clipped origin in level-0 pixels for
        # readers that take level-0 locations; this reader crops the level directly.
        downsample = float(self._level_downsamples[level][0])
        x_level = int(np.floor(location[0] / downsample))
        y_level = int(np.floor(location[1] / downsample))
        x1, y1 = max(x_level, 0), max(y_level, 0)
        region = self._read_level_window(
            level=level, x1=x1, y1=y1, x2=x1 + read_width, y2=y1 + read_height
        )
        return paste_region(
            canvas=canvas, region=region, paste_offset=bounds.paste_offset
        )

    def _read_level_window(
        self, *, level: int, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """Decode only the segments (tiles or strips) overlapping the level window."""
        page = self._pages[level]
        width, height = self._level_dimensions[level]
        samples = self._samples(level)
        if page.is_tiled:
            segment_width, segment_height = int(page.tilewidth), int(page.tilelength)
        else:
            segment_width = width
            segment_height = min(int(page.rowsperstrip) or height, height)
        columns = -(-width // segment_width)
        window = np.empty((y2 - y1, x2 - x1, samples), dtype=page.dtype)
        decode = page.decode
        handle = self._tiff.filehandle
        for row in range(y1 // segment_height, (y2 - 1) // segment_height + 1):
            for column in range(x1 // segment_width, (x2 - 1) // segment_width + 1):
                index = row * columns + column
                byte_count = int(page.databytecounts[index])
                with self._lock:
                    if byte_count > 0:
                        handle.seek(int(page.dataoffsets[index]))
                        data = handle.read(byte_count)
                    else:
                        data = None
                    segment, position, shape = decode(
                        data, index, jpegtables=page.jpegtables
                    )
                segment_y, segment_x = int(position[2]), int(position[3])
                if segment is None:
                    segment = np.full(
                        (int(shape[1]), int(shape[2]), samples),
                        page.nodata,
                        dtype=page.dtype,
                    )
                else:
                    segment = np.asarray(segment)[0]
                overlap_x1, overlap_y1 = max(segment_x, x1), max(segment_y, y1)
                overlap_x2 = min(segment_x + segment.shape[1], x2, width)
                overlap_y2 = min(segment_y + segment.shape[0], y2, height)
                if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
                    continue
                window[
                    overlap_y1 - y1 : overlap_y2 - y1, overlap_x1 - x1 : overlap_x2 - x1
                ] = segment[
                    overlap_y1 - segment_y : overlap_y2 - segment_y,
                    overlap_x1 - segment_x : overlap_x2 - segment_x,
                ]
        if samples == 1:
            return window[..., 0]
        return window

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        level = self.level_count - 1
        array = self.read_level(level)
        width, height = self._level_dimensions[level]
        step = max(
            1,
            int(math.ceil(max(width / max(1, int(size[0])), height / max(1, int(size[1]))))),
        )
        return np.ascontiguousarray(array[::step, ::step])

    def close(self) -> None:
        with self._lock:
            self._tiff.close()

    def __enter__(self) -> "TifffileReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def write_label_tiff(
    *,
    path: str | Path,
    tiles: Iterable[np.ndarray | None],
    dimensions: tuple[int, int],
    tile_size: int,
    spacing_um: float,
    dtype: DTypeLike = np.uint8,
    compression: Literal["lzw", "deflate"] = "lzw",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a pyramidal label TIFF from ``tiles``, a stream of base-level tiles.

    The raster is ``dimensions`` ``(width, height)`` pixels of one unsigned-integer
    sample, stored losslessly as min-is-black ``tile_size`` px tiles. ``tiles`` yields the
    ``ceil(width / tile_size) * ceil(height / tile_size)`` tiles in row order: left to
    right, then top to bottom. tifffile writes compressed tiles only in that order, so
    the caller is responsible for it. Each tile is a 2-D ``dtype`` array of at most
    ``tile_size`` px per side:

    - a tile smaller than ``tile_size`` is zero-padded on the right and bottom, as
      tifffile pads incomplete tiles, so the pixels it lacks are written as zeros;
    - ``None`` stands for a missing tile and is written as zeros;
    - on the right and bottom edges, pixels past the raster are discarded.

    A stream with fewer or more tiles, or a tile of another dtype or shape, raises
    ``ValueError``. Each tile is copied as it is taken, so the stream may refill and
    yield one buffer.

    Reduced levels halve the level below, rounding up, until one tile holds the coarsest
    level or a side is 1 px. Each keeps every other pixel of the level below
    (nearest-neighbour), never an average, so it holds only stored labels. The base level
    is never held whole: only the 2x level, a quarter of its size, fills in memory while
    the base level streams, and each coarser level is then copied out of the one below.

    ``spacing_um`` is the base level's spacing in um/px. Every level's resolution tags
    carry it, scaled by the level's downsample, where :class:`TifffileReader` and the
    native readers find it. ``metadata``, such as the caller's value-to-class mapping, is
    stored as JSON in the base level's image description:
    ``json.loads(tifffile.TiffFile(path).pages[0].description)`` returns it.

    ``compression`` is ``"lzw"``, the default, or ``"deflate"``. The tifffile, cuCIM,
    OpenSlide and ASAP backends decode LZW; cuCIM and ASAP do not decode deflate
    single-sample tiles, so a deflate label TIFF reads through tifffile or OpenSlide only.

    The file is written under a hidden temporary name beside ``path`` and renamed to
    ``path``, replacing any file there, only once complete. When writing fails, the
    temporary file is removed and ``path`` is left as it was. Returns ``path``.
    """
    import tifffile

    path = Path(path)
    dtype = np.dtype(dtype)
    _validate_label_tiff_arguments(
        dimensions=dimensions,
        tile_size=tile_size,
        spacing_um=spacing_um,
        dtype=dtype,
        compression=compression,
        metadata=metadata,
    )
    # NumPy integers pass validation, but unsigned ones wrap or turn float when negated
    dimensions = (int(dimensions[0]), int(dimensions[1]))
    tile_size = int(tile_size)
    description = None if metadata is None else _metadata_description(metadata)
    level_dimensions = _halved_dimensions(dimensions=dimensions, tile_size=tile_size)
    pixels_per_centimeter = _UM_PER_CENTIMETER / spacing_um
    page_options: dict[str, Any] = dict(
        tile=(tile_size, tile_size),
        photometric="minisblack",
        compression=_LABEL_TIFF_COMPRESSIONS[compression],
        resolutionunit="CENTIMETER",
        metadata=None,
        buffersize=_ENCODE_BUFFER_BYTES,
    )
    width, height = dimensions
    # The 2x level, written only when the base level is halved.
    reduced = np.zeros((-(-height // 2), -(-width // 2)), dtype=dtype)
    # BigTIFF, like ASAP's writer: the compressed size is unknown until written, and
    # classic TIFF offsets stop at 4 GiB.
    with _replaced_when_complete(path) as handle, tifffile.TiffWriter(
        handle, bigtiff=True
    ) as writer:
        writer.write(
            _base_level_tiles(
                iter(tiles),
                dimensions=dimensions,
                tile_size=tile_size,
                dtype=dtype,
                reduced=reduced,
            ),
            shape=(height, width),
            dtype=dtype,
            resolution=(pixels_per_centimeter, pixels_per_centimeter),
            description=description,
            **page_options,
        )
        for level in range(1, len(level_dimensions)):
            if level > 1:
                # rebinding frees the level below once this one is copied out of it
                reduced = np.ascontiguousarray(reduced[::2, ::2])
            level_resolution = pixels_per_centimeter / 2**level
            writer.write(
                reduced,
                subfiletype=1,
                resolution=(level_resolution, level_resolution),
                **page_options,
            )
    return path


def _base_level_tiles(
    stream: Iterator[np.ndarray | None],
    *,
    dimensions: tuple[int, int],
    tile_size: int,
    dtype: np.dtype[Any],
    reduced: np.ndarray,
) -> Iterator[np.ndarray]:
    """Check the caller's row-order tiles and fill ``reduced``, the 2x level, from them.

    Yields exactly one tile per grid position. The grid's last tile is yielded only after
    the stream is found exhausted, so a longer stream fails before the level is complete.
    """
    width, height = dimensions
    columns = -(-width // tile_size)
    tile_count = columns * -(-height // tile_size)
    grid = f"the {tile_count} tiles of a {width}x{height} raster in {tile_size} px tiles"
    # A missing tile is written as encoded zeros rather than as an empty segment, which
    # not every native reader decodes as zeros.
    zero_tile = np.zeros((tile_size, tile_size), dtype=dtype)
    # Each tile is copied, in its own dtype, as it is taken: resuming the stream may
    # refill or clear the caller's array, and tifffile encodes tiles on worker threads
    # after taking more of them.
    owned = (None if tile is None else np.array(tile) for tile in stream)
    for index in range(tile_count):
        tile = next(owned, _STREAM_END)
        if tile is _STREAM_END:
            raise ValueError(f"tile stream ended after {index} of {grid}")
        if index == tile_count - 1 and next(owned, _STREAM_END) is not _STREAM_END:
            raise ValueError(f"tile stream yielded more than {grid}")
        if tile is None:
            yield zero_tile
            continue
        # tifffile casts every tile to the page dtype, which would wrap labels silently
        if tile.dtype != dtype:
            raise ValueError(f"tile {index} has dtype {tile.dtype}, expected {dtype}")
        if tile.ndim != 2 or tile.shape[0] > tile_size or tile.shape[1] > tile_size:
            raise ValueError(
                f"tile {index} has shape {tile.shape}, expected a 2-D array of at most "
                f"({tile_size}, {tile_size})"
            )
        row, column = divmod(index, columns)
        y, x = row * tile_size, column * tile_size
        # an even tile_size puts every tile origin on the 2x grid
        sampled = tile[: height - y : 2, : width - x : 2]
        reduced[
            y // 2 : y // 2 + sampled.shape[0], x // 2 : x // 2 + sampled.shape[1]
        ] = sampled
        yield tile


@contextmanager
def _replaced_when_complete(path: Path) -> Iterator[IO[bytes]]:
    """A file handle whose content replaces ``path`` only when the block completes.

    The handle writes a hidden temporary file beside ``path``, without the TIFF suffix,
    so a crash leaves nothing that looks finished. On success it is synced to disk and
    renamed over ``path``; on any failure it is removed.
    """
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        promote_temp_file(temp_path=temp_path, target_path=path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _validate_label_tiff_arguments(
    *,
    dimensions: Any,
    tile_size: Any,
    spacing_um: Any,
    dtype: np.dtype[Any],
    compression: Any,
    metadata: Any,
) -> None:
    if not (
        isinstance(dimensions, tuple)
        and len(dimensions) == 2
        and all(_is_integer(value) and value > 0 for value in dimensions)
    ):
        raise ValueError(
            f"dimensions must be two positive integers (width, height), got {dimensions!r}"
        )
    # TIFF tile sides are multiples of 16
    if not (_is_integer(tile_size) and tile_size > 0 and tile_size % 16 == 0):
        raise ValueError(f"tile_size must be a positive multiple of 16, got {tile_size!r}")
    if not (
        isinstance(spacing_um, (int, float, np.integer, np.floating))
        and not isinstance(spacing_um, bool)
        and math.isfinite(spacing_um)
        and spacing_um > 0
    ):
        raise ValueError(f"spacing_um must be finite and positive, got {spacing_um!r}")
    if dtype.kind != "u":
        raise ValueError(f"dtype must be an unsigned integer type, got {dtype}")
    if compression not in _LABEL_TIFF_COMPRESSIONS:
        raise ValueError(
            f"compression must be one of {list(_LABEL_TIFF_COMPRESSIONS)}, got {compression!r}"
        )
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _metadata_description(metadata: Mapping[str, Any]) -> str:
    """``metadata`` as the JSON image description.

    tifffile opens a description that starts with ``{`` and contains ``"shape":`` as its
    own array metadata, and fails on any other ``"shape"`` key. A space before every
    colon keeps a caller's ``"shape"`` key from matching while the text stays plain
    JSON; inside strings, quotes are escaped, so ``"shape":`` cannot occur there either.
    """
    return json.dumps(dict(metadata), separators=(", ", " : "))


def _halved_dimensions(
    *, dimensions: tuple[int, int], tile_size: int
) -> list[tuple[int, int]]:
    """Level ``(width, height)``s, halving (rounding up) until one tile holds a level.

    Halving also stops at a 1 px side: tifffile finds a level only when every side is
    reduced from the level above.
    """
    levels = [dimensions]
    while max(levels[-1]) > tile_size and min(levels[-1]) > 1:
        level_width, level_height = levels[-1]
        levels.append((-(-level_width // 2), -(-level_height // 2)))
    return levels


__all__ = [
    "PADDING_VALUE",
    "TIFFFILE_SUPPORTED_SUFFIXES",
    "TifffileReader",
    "supports_tifffile_path",
    "write_label_tiff",
]
