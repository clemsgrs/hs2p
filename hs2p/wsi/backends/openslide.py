from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hs2p.wsi.geometry import compute_level_spacings


def _pad_canvas(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


class OpenSlideReader:
    def __init__(self, path: str | Path, *, spacing_override: float | None = None):
        try:
            import openslide
        except ImportError as exc:
            raise ImportError(
                "openslide-python is required for the openslide backend. "
                "Install it with: pip install openslide-python"
            ) from exc

        self._slide = openslide.OpenSlide(str(path))
        self.native_spacing = self._extract_spacing()
        self._spacing = (
            float(spacing_override) if spacing_override is not None else self.native_spacing
        )
        self._level_dimensions = [
            (int(width), int(height)) for width, height in self._slide.level_dimensions
        ]
        self._level_downsamples = [
            (float(value), float(value)) for value in self._slide.level_downsamples
        ]
        self._spacings = compute_level_spacings(
            level0_spacing_um=self.native_spacing,
            level_downsamples=self._level_downsamples,
        )

    def _extract_spacing(self) -> float:
        properties = self._slide.properties
        mpp_x = properties.get("openslide.mpp-x")
        if mpp_x is not None:
            return float(mpp_x)
        objective = properties.get("openslide.objective-power")
        if objective is not None:
            objective_value = float(objective)
            if objective_value > 0:
                return 10.0 / objective_value
        raise ValueError(
            "Unable to infer slide spacing from OpenSlide metadata. "
            "Provide spacing_at_level_0 or use a slide with valid spacing metadata."
        )

    @property
    def backend_name(self) -> str:
        return "openslide"

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
        return self._slide.level_count

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return list(self._level_dimensions)

    @property
    def level_downsamples(self) -> list[tuple[float, float]]:
        return list(self._level_downsamples)

    def read_level(self, level: int) -> np.ndarray:
        width, height = self._level_dimensions[level]
        return self.read_region((0, 0), level, (width, height), pad_missing=True)

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        pad_missing: bool = True,
    ) -> np.ndarray:
        width, height = int(size[0]), int(size[1])
        if not pad_missing:
            return np.array(
                self._slide.read_region(
                    (int(location[0]), int(location[1])),
                    int(level),
                    (width, height),
                ).convert("RGB")
            )

        canvas = _pad_canvas(width, height)
        if width <= 0 or height <= 0:
            return canvas

        level_width, level_height = self._level_dimensions[level]
        downsample = float(self._level_downsamples[level][0])
        x_level = int(np.floor(location[0] / downsample))
        y_level = int(np.floor(location[1] / downsample))
        x1 = max(x_level, 0)
        y1 = max(y_level, 0)
        x2 = min(x_level + width, level_width)
        y2 = min(y_level + height, level_height)
        if x2 <= x1 or y2 <= y1:
            return canvas

        read_location = (int(round(x1 * downsample)), int(round(y1 * downsample)))
        read_width = x2 - x1
        read_height = y2 - y1
        region = self._slide.read_region(read_location, int(level), (read_width, read_height))
        region_rgb = np.array(region.convert("RGB"))
        paste_x = x1 - x_level
        paste_y = y1 - y_level
        canvas[paste_y : paste_y + read_height, paste_x : paste_x + read_width] = region_rgb
        return canvas

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        return np.array(self._slide.get_thumbnail(size).convert("RGB"))

    def close(self) -> None:
        self._slide.close()

    def __enter__(self) -> OpenSlideReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

