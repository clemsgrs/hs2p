
from pathlib import Path
from typing import Any

import numpy as np

from hs2p.wsi.backends.common import paste_region, resolve_padded_read_bounds
from hs2p.wsi.geometry import compute_level_spacings


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
            float(spacing_override)
            if spacing_override is not None
            else self.native_spacing
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
        region = self._slide.read_region((0, 0), int(level), (width, height))
        return np.array(region.convert("RGB"))

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

        region = self._slide.read_region(
            bounds.read_location,
            int(level),
            (int(read_width), int(read_height)),
        )
        return paste_region(
            bounds.canvas,
            np.array(region.convert("RGB")),
            paste_offset=bounds.paste_offset,
        )

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        return np.array(self._slide.get_thumbnail(size).convert("RGB"))

    def close(self) -> None:
        self._slide.close()

    def __enter__(self) -> "OpenSlideReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
