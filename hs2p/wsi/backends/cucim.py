from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from hs2p.wsi.backends.common import paste_region, resolve_padded_read_bounds
from hs2p.wsi.geometry import compute_level_spacings

CUCIM_SUPPORTED_SUFFIXES = {".svs", ".tif", ".tiff"}


def supports_cucim_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in CUCIM_SUPPORTED_SUFFIXES


def _as_rgb_uint8(region: Any) -> np.ndarray:
    arr = np.asarray(region)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] >= 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_value = float(np.nanmax(arr)) if arr.size else 0.0
            if max_value <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    return arr


def _extract_spacing_from_metadata(metadata: dict[str, Any]) -> float:
    for entry in metadata.values():
        if not isinstance(entry, dict):
            continue
        if "MPP" in entry:
            return float(entry["MPP"])
        if "DICOM_PIXEL_SPACING" in entry:
            spacing = entry["DICOM_PIXEL_SPACING"]
            if isinstance(spacing, (list, tuple)):
                spacing = spacing[0]
            return float(spacing) * 1000.0
        spacing = entry.get("spacing")
        if spacing is None:
            continue
        if isinstance(spacing, (list, tuple)):
            spacing = spacing[0]
        units = entry.get("spacing_units")
        if isinstance(units, (list, tuple)):
            units = units[0]
        factor = {
            "mm": 1000.0,
            "millimeter": 1000.0,
            "millimeters": 1000.0,
            "cm": 10000.0,
            "centimeter": 10000.0,
            "centimeters": 10000.0,
            "um": 1.0,
            "micron": 1.0,
            "microns": 1.0,
            "micrometer": 1.0,
            "micrometers": 1.0,
        }.get(str(units).lower() if units is not None else "")
        if factor is not None:
            return float(spacing) * factor
    raise ValueError(
        "Unable to infer slide spacing from cuCIM metadata. "
        "Provide spacing_at_level_0 or use a slide with valid spacing metadata."
    )


class CuCIMReader:
    def __init__(
        self,
        path: str | Path,
        *,
        spacing_override: float | None = None,
        gpu_decode: bool = False,
    ):
        if gpu_decode:
            os.environ["ENABLE_CUSLIDE2"] = "1"
        try:
            cucim = importlib.import_module("cucim")
        except ImportError as exc:
            raise ImportError(
                "cucim is required for the cucim backend. "
                "Install it with: pip install cucim-cuXX"
            ) from exc

        self._slide = cucim.CuImage(str(path))
        self._gpu_decode = bool(gpu_decode)
        self._metadata = getattr(self._slide, "metadata", {}) or {}
        cucim_meta = (
            self._metadata.get("cucim", {}) if isinstance(self._metadata, dict) else {}
        )
        resolutions = (
            cucim_meta.get("resolutions", {}) if isinstance(cucim_meta, dict) else {}
        )
        self._level_dimensions = [
            (int(dim[0]), int(dim[1]))
            for dim in resolutions.get("level_dimensions", [])
        ]
        if not self._level_dimensions:
            shape = cucim_meta.get("shape", ())
            if len(shape) >= 2:
                self._level_dimensions = [(int(shape[1]), int(shape[0]))]
        self._level_downsamples = [
            (float(value), float(value))
            for value in resolutions.get("level_downsamples", [1.0])
        ]
        if len(self._level_downsamples) != len(self._level_dimensions):
            self._level_downsamples = [
                (
                    self._level_dimensions[0][0] / float(width),
                    self._level_dimensions[0][1] / float(height),
                )
                for width, height in self._level_dimensions
            ]
        self.native_spacing = _extract_spacing_from_metadata(self._metadata)
        self._spacing = (
            float(spacing_override)
            if spacing_override is not None
            else self.native_spacing
        )
        self._spacings = compute_level_spacings(
            level0_spacing_um=self.native_spacing,
            level_downsamples=self._level_downsamples,
        )

    @property
    def backend_name(self) -> str:
        return "cucim"

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

    def _read_region(self, location, size, *, level: int, num_workers: int = 1):
        kwargs: dict[str, Any] = {
            "location": (int(location[0]), int(location[1])),
            "size": (int(size[0]), int(size[1])),
            "level": int(level),
        }
        if num_workers > 1:
            kwargs["num_workers"] = max(1, int(num_workers))
        if self._gpu_decode:
            kwargs["device"] = "cuda"
        try:
            return self._slide.read_region(**kwargs)
        except TypeError:
            kwargs.pop("device", None)
            return self._slide.read_region(**kwargs)

    def read_level(self, level: int) -> np.ndarray:
        dims = self._level_dimensions[level]
        return self.read_region((0, 0), level, dims)

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        bounds = resolve_padded_read_bounds(
            location=location,
            size=size,
            level_dimensions=self._level_dimensions[level],
            downsample=float(self._level_downsamples[level][0]),
        )
        read_width, read_height = bounds.read_size
        if read_width <= 0 or read_height <= 0:
            return bounds.canvas

        region = _as_rgb_uint8(
            self._read_region(bounds.read_location, bounds.read_size, level=int(level))
        )
        return paste_region(bounds.canvas, region, paste_offset=bounds.paste_offset)

    def read_regions(
        self,
        locations: list[tuple[int, int]],
        level: int,
        size: tuple[int, int],
        *,
        num_workers: int | None = None,
    ) -> Iterable[np.ndarray]:
        requested_size = (int(size[0]), int(size[1]))
        bounds_per_location = [
            resolve_padded_read_bounds(
                location=location,
                size=requested_size,
                level_dimensions=self._level_dimensions[level],
                downsample=float(self._level_downsamples[level][0]),
            )
            for location in locations
        ]
        if any(
            bounds.read_size != requested_size or bounds.paste_offset != (0, 0)
            for bounds in bounds_per_location
        ):
            for location in locations:
                yield self.read_region(location, level, requested_size)
            return

        kwargs: dict[str, Any] = {
            "location": [(int(x), int(y)) for x, y in locations],
            "size": requested_size,
            "level": int(level),
            "num_workers": max(1, int(num_workers or 1)),
        }
        if self._gpu_decode:
            kwargs["device"] = "cuda"
        try:
            regions = self._slide.read_region(**kwargs)
        except TypeError:
            kwargs.pop("device", None)
            regions = self._slide.read_region(**kwargs)
        for bounds, region in zip(bounds_per_location, regions):
            arr = _as_rgb_uint8(region)
            yield paste_region(bounds.canvas, arr, paste_offset=bounds.paste_offset)

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        level = self.level_count - 1
        region = self.read_level(level)
        target_width, target_height = int(size[0]), int(size[1])
        if target_width <= 0 or target_height <= 0:
            return region
        scale = min(
            target_width / max(region.shape[1], 1),
            target_height / max(region.shape[0], 1),
        )
        width = max(1, int(round(region.shape[1] * scale)))
        height = max(1, int(round(region.shape[0] * scale)))
        return cv2.resize(region, (width, height), interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        return None

    def __enter__(self) -> "CuCIMReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
