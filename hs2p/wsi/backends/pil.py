from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from hs2p.wsi.backends.common import (
    paste_region,
    resolve_level0_spacing,
    resolve_padded_read_bounds,
)


PIL_SUPPORTED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
PIL_MAX_IMAGE_PIXELS = 89_478_485
_PIL_HEADER_LOCK = Lock()
_LABEL_MODES = frozenset({"1", "I", "I;16", "L", "P"})


class PILImageTooLargeError(ValueError):
    """Raised before decoding a flat raster above the project pixel ceiling."""


def supports_pil_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in PIL_SUPPORTED_SUFFIXES


class PILReader:
    def __init__(self, path: str | Path, *, spacing_override: float | None = None):
        self._path = str(path)
        try:
            with _PIL_HEADER_LOCK:
                pillow_ceiling = Image.MAX_IMAGE_PIXELS
                try:
                    Image.MAX_IMAGE_PIXELS = None
                    self._image = Image.open(self._path)
                finally:
                    Image.MAX_IMAGE_PIXELS = pillow_ceiling
        except Exception as exc:
            raise RuntimeError(
                f"PIL backend failed to open path={self._path}: {exc}"
            ) from exc
        width, height = self.dimensions
        pixel_count = width * height
        if pixel_count > PIL_MAX_IMAGE_PIXELS:
            self._image.close()
            raise PILImageTooLargeError(
                "PIL image exceeds the defensive size limit: "
                f"path={self._path}, dimensions={width}x{height}, "
                f"pixel_count={pixel_count}, ceiling={PIL_MAX_IMAGE_PIXELS}"
            )
        self.native_spacing = None
        try:
            self._spacing = resolve_level0_spacing(
                path=self._path,
                backend=self.backend_name,
                native_spacing=self.native_spacing,
                spacing_override=spacing_override,
            )
        except Exception:
            self._image.close()
            raise

    @property
    def backend_name(self) -> str:
        return "pil"

    @property
    def dimensions(self) -> tuple[int, int]:
        return (int(self._image.width), int(self._image.height))

    @property
    def spacing(self) -> float:
        return self._spacing

    @property
    def spacings(self) -> list[float]:
        return [self._spacing]

    @property
    def level_count(self) -> int:
        return 1

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return [self.dimensions]

    @property
    def level_downsamples(self) -> list[tuple[float, float]]:
        return [(1.0, 1.0)]

    def _array_from_image(self, image: Image.Image) -> np.ndarray:
        if image.mode in _LABEL_MODES:
            array = np.asarray(image)
            if array.dtype == np.bool_:
                return array.astype(np.uint8)
            return array
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _validate_level(level: int) -> None:
        if int(level) != 0:
            raise IndexError(f"PIL flat raster has only level 0, got level={level}")

    def read_level(self, level: int) -> np.ndarray:
        self._validate_level(level)
        return self._array_from_image(self._image)

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        self._validate_level(level)
        bounds = resolve_padded_read_bounds(
            location=location,
            size=size,
            level_dimensions=self.dimensions,
            downsample=1.0,
        )
        canvas = bounds.canvas
        if self._image.mode in _LABEL_MODES:
            dtype = np.uint16 if self._image.mode == "I;16" else np.uint8
            canvas = np.full((int(size[1]), int(size[0])), 255, dtype=dtype)

        read_width, read_height = bounds.read_size
        if read_width <= 0 or read_height <= 0:
            return canvas
        x, y = bounds.read_location
        region = self._image.crop(
            (x, y, x + int(read_width), y + int(read_height))
        )
        return paste_region(
            canvas,
            self._array_from_image(region),
            paste_offset=bounds.paste_offset,
        )

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        thumbnail = self._image.copy()
        resample = (
            Image.Resampling.NEAREST
            if thumbnail.mode in _LABEL_MODES
            else Image.Resampling.LANCZOS
        )
        thumbnail.thumbnail(
            (max(1, int(size[0])), max(1, int(size[1]))),
            resample=resample,
        )
        return self._array_from_image(thumbnail)

    def close(self) -> None:
        self._image.close()

    def __enter__(self) -> "PILReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
