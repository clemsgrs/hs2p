from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

class ASAPReader:
    def __init__(self, path: str | Path, *, spacing_override: float | None = None):
        try:
            import wholeslidedata as wsd
        except ImportError as exc:
            raise ImportError(
                "wholeslidedata is required for the asap backend. "
                "Install it with: pip install wholeslidedata"
            ) from exc

        self._path = Path(path)
        self._wsi = wsd.WholeSlideImage(self._path, backend="asap")
        self.native_spacing = float(self._wsi.spacings[0])
        self._spacing = (
            float(spacing_override) if spacing_override is not None else self.native_spacing
        )
        self._spacings = list(float(value) for value in self._wsi.spacings)
        self._level_dimensions = [
            (int(width), int(height)) for width, height in self._wsi.shapes
        ]
        self._level_downsamples = [
            (float(value), float(value)) for value in self._wsi.downsamplings
        ]

    @property
    def backend_name(self) -> str:
        return "asap"

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
        return np.asarray(self._wsi.get_slide(spacing=self._spacings[level]))

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        pad_missing: bool = True,
    ) -> np.ndarray:
        # wholeslidedata already pads out-of-bounds ASAP reads; keep the
        # shared reader signature for protocol compatibility.
        return np.asarray(
            self._wsi.get_patch(
                int(location[0]),
                int(location[1]),
                int(size[0]),
                int(size[1]),
                spacing=float(self._spacings[level]),
                center=False,
            )
        )

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        level = self.level_count - 1
        arr = self.read_level(level)
        target_width = max(1, int(size[0]))
        target_height = max(1, int(size[1]))
        if arr.shape[1] == target_width and arr.shape[0] == target_height:
            return arr
        import cv2

        return cv2.resize(
            arr,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> ASAPReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
