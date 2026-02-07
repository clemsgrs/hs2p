from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PyramidSpec:
    spacings: list[float]
    levels: list[np.ndarray]


class FakePyramidWSI:
    def __init__(self, spec: PyramidSpec):
        self.spacings = spec.spacings
        self._levels = spec.levels
        self.shapes = tuple((arr.shape[1], arr.shape[0]) for arr in self._levels)
        self.downsamplings = tuple(self.shapes[0][0] / shape[0] for shape in self.shapes)

    def _best_level(self, spacing: float) -> int:
        return int(np.argmin([abs(s - spacing) for s in self.spacings]))

    def get_slide(self, spacing: float) -> np.ndarray:
        return self._levels[self._best_level(spacing)]

    def get_patch(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: float,
        center: bool = False,
    ) -> np.ndarray:
        del center
        level = self._best_level(spacing)
        downsample = self.downsamplings[level]
        arr = self._levels[level]

        x_level = int(round(x / downsample))
        y_level = int(round(y / downsample))

        patch = np.zeros((height, width, arr.shape[2]), dtype=arr.dtype)
        src = arr[y_level : y_level + height, x_level : x_level + width, :]
        patch[: src.shape[0], : src.shape[1], :] = src
        return patch


class FakeWSDFactory:
    def __init__(self, slide_spec: PyramidSpec, mask_spec: PyramidSpec):
        self.slide_spec = slide_spec
        self.mask_spec = mask_spec

    def __call__(self, path: Path, backend: str = "asap") -> FakePyramidWSI:
        del backend
        name = str(path).lower()
        if "mask" in name:
            return FakePyramidWSI(self.mask_spec)
        return FakePyramidWSI(self.slide_spec)


def make_slide_spec() -> PyramidSpec:
    slide_l0 = np.ones((32, 32, 3), dtype=np.uint8) * 180
    slide_l1 = slide_l0[::2, ::2, :]
    return PyramidSpec(spacings=[1.0, 2.0], levels=[slide_l0, slide_l1])


def make_mask_spec(mask_l0: np.ndarray) -> PyramidSpec:
    mask_l0 = mask_l0.astype(np.uint8)
    mask_l1 = mask_l0[::2, ::2, :]
    return PyramidSpec(spacings=[2.0, 4.0], levels=[mask_l0, mask_l1])
