"""Lossless TIFF reader: returns stored sample values in their stored dtype.

The display readers (cuCIM, OpenSlide, VIPS, ASAP) decode to 8-bit RGB. This reader
exists for label rasters, where the stored integer is the label: it returns 2-D arrays
for single-sample pages and ``(height, width, samples)`` arrays otherwise, never
rescaling, expanding a palette or converting the dtype. Out-of-canvas padding is 255,
like the PIL reader, so a padded label read fails validation unless 255 is declared.
"""

from __future__ import annotations

import math
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

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


__all__ = [
    "PADDING_VALUE",
    "TIFFFILE_SUPPORTED_SUFFIXES",
    "TifffileReader",
    "supports_tifffile_path",
]
