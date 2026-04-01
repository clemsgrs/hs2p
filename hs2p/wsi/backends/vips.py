from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hs2p.wsi.backends.common import paste_region, resolve_padded_read_bounds
from hs2p.wsi.geometry import compute_level_spacings

VIPS_SUPPORTED_SUFFIXES = {
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".svslide",
    ".bif",
}


def supports_vips_path(path: str | Path) -> bool:
    path_str = str(path).lower()
    return any(path_str.endswith(suffix) for suffix in VIPS_SUPPORTED_SUFFIXES)


def _vips_to_numpy(image) -> np.ndarray:
    arr = np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=[image.height, image.width, image.bands],
    )
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] >= 4:
        arr = arr[..., :3]
    return arr

class VIPSReader:
    def __init__(self, path: str | Path, *, spacing_override: float | None = None):
        try:
            import pyvips
        except (ImportError, OSError) as exc:
            raise ImportError(
                "pyvips is required for the vips backend. "
                "Install it with: pip install pyvips and ensure libvips is available."
            ) from exc

        self._vips = pyvips
        self._path = str(path)
        self._root = pyvips.Image.new_from_file(self._path)
        self._loader = (
            self._root.get("vips-loader")
            if "vips-loader" in self._root.get_fields()
            else None
        )
        self._level_dimensions = self._resolve_level_dimensions()
        self._level_downsamples = self._resolve_level_downsamples()
        self.native_spacing = self._extract_spacing()
        self._spacing = (
            float(spacing_override)
            if spacing_override is not None
            else self.native_spacing
        )
        self._spacings = compute_level_spacings(
            level0_spacing_um=self.native_spacing,
            level_downsamples=self._level_downsamples,
        )

    def _extract_spacing(self) -> float:
        fields = set(self._root.get_fields())
        if "openslide.mpp-x" in fields:
            return float(self._root.get("openslide.mpp-x"))
        if "aperio.MPP" in fields:
            return float(self._root.get("aperio.MPP"))
        if "xres" in fields:
            xres = float(self._root.get("xres"))
            if xres > 0:
                return 1000.0 / xres
        raise ValueError(
            "Unable to infer slide spacing from libvips metadata. "
            "Provide spacing_at_level_0 or use a slide with valid spacing metadata."
        )

    def _resolve_level_dimensions(self) -> list[tuple[int, int]]:
        levels: list[tuple[int, int]] = []
        fields = set(self._root.get_fields())
        if self._loader == "tiffload" and "n-pages" in fields:
            n_pages = int(self._root.get("n-pages"))
            for page in range(n_pages):
                image = self._vips.Image.new_from_file(self._path, page=page)
                dims = (int(image.width), int(image.height))
                if not levels or dims != levels[-1]:
                    levels.append(dims)
            return levels

        if "openslide.level-count" in fields:
            level_count = int(self._root.get("openslide.level-count"))
            for level in range(level_count):
                image = self._vips.Image.new_from_file(self._path, level=level)
                levels.append((int(image.width), int(image.height)))
            return levels

        levels.append((int(self._root.width), int(self._root.height)))
        return levels

    def _resolve_level_downsamples(self) -> list[tuple[float, float]]:
        width_0, height_0 = self._level_dimensions[0]
        return [
            (width_0 / float(width), height_0 / float(height))
            for width, height in self._level_dimensions
        ]

    def _open_level_image(self, level: int):
        if self._loader == "tiffload":
            return self._vips.Image.new_from_file(self._path, page=int(level))
        if len(self._level_dimensions) > 1:
            return self._vips.Image.new_from_file(self._path, level=int(level))
        return self._vips.Image.new_from_file(self._path)

    @property
    def backend_name(self) -> str:
        return "vips"

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
        return len(self._level_dimensions)

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return list(self._level_dimensions)

    @property
    def level_downsamples(self) -> list[tuple[float, float]]:
        return list(self._level_downsamples)

    def read_level(self, level: int) -> np.ndarray:
        return _vips_to_numpy(self._open_level_image(level))

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        image = self._open_level_image(level)
        bounds = resolve_padded_read_bounds(
            location=location,
            size=size,
            level_dimensions=self._level_dimensions[level],
            downsample=float(self._level_downsamples[level][0]),
        )
        read_width, read_height = bounds.read_size
        if read_width <= 0 or read_height <= 0:
            return bounds.canvas

        x = int(bounds.read_location[0] / self._level_downsamples[level][0])
        y = int(bounds.read_location[1] / self._level_downsamples[level][0])
        region = _vips_to_numpy(image.crop(x, y, int(read_width), int(read_height)))
        return paste_region(bounds.canvas, region, paste_offset=bounds.paste_offset)

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        image = self._open_level_image(self.level_count - 1)
        target_width = max(1, int(size[0]))
        target_height = max(1, int(size[1]))
        scale = min(
            target_width / max(image.width, 1),
            target_height / max(image.height, 1),
        )
        thumb = image.resize(scale)
        return _vips_to_numpy(thumb)

    def close(self) -> None:
        return None

    def __enter__(self) -> "VIPSReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
