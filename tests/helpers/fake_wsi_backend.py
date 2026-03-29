from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PyramidSpec:
    spacings: list[float]
    levels: list[np.ndarray]


class FakePyramidWSI:
    def __init__(self, spec: PyramidSpec, backend_name: str = "synthetic"):
        self.spacings = spec.spacings
        self._levels = spec.levels
        self.shapes = tuple((arr.shape[1], arr.shape[0]) for arr in self._levels)
        self.downsamplings = tuple(
            self.shapes[0][0] / shape[0] for shape in self.shapes
        )
        self.native_spacing = float(self.spacings[0])
        self._backend_name = backend_name

    def _best_level(self, spacing: float) -> int:
        return int(np.argmin([abs(s - spacing) for s in self.spacings]))

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.shapes[0]

    @property
    def spacing(self) -> float:
        return self.native_spacing

    @property
    def level_count(self) -> int:
        return len(self._levels)

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return list(self.shapes)

    @property
    def level_downsamples(self) -> list[tuple[float, float]]:
        return [(float(ds), float(ds)) for ds in self.downsamplings]

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

    def read_level(self, level: int) -> np.ndarray:
        return self._levels[level]

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        pad_missing: bool = True,
    ) -> np.ndarray:
        del pad_missing
        return self.get_patch(
            x=int(location[0]),
            y=int(location[1]),
            width=int(size[0]),
            height=int(size[1]),
            spacing=self.spacings[level],
            center=False,
        )

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        arr = self._levels[-1]
        return arr[: int(size[1]), : int(size[0])]

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeWSDFactory:
    def __init__(self, slide_spec: PyramidSpec, mask_spec: PyramidSpec):
        self.slide_spec = slide_spec
        self.mask_spec = mask_spec

    def __call__(
        self,
        path: Path,
        backend: str = "asap",
        spacing_override: float | None = None,
        gpu_decode: bool = False,
    ) -> FakePyramidWSI:
        del spacing_override, gpu_decode
        name = str(path).lower()
        if "mask" in name:
            return FakePyramidWSI(self.mask_spec, backend_name=backend)
        return FakePyramidWSI(self.slide_spec, backend_name=backend)


def make_slide_spec() -> PyramidSpec:
    slide_l0 = np.ones((32, 32, 3), dtype=np.uint8) * 180
    slide_l1 = slide_l0[::2, ::2, :]
    return PyramidSpec(spacings=[1.0, 2.0], levels=[slide_l0, slide_l1])


def make_mask_spec(mask_l0: np.ndarray) -> PyramidSpec:
    mask_l0 = mask_l0.astype(np.uint8)
    mask_l1 = mask_l0[::2, ::2, :]
    return PyramidSpec(spacings=[2.0, 4.0], levels=[mask_l0, mask_l1])
