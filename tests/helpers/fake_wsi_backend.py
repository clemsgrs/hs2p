from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hs2p.wsi.backends.common import make_white_canvas


@dataclass
class PyramidSpec:
    spacings: list[float]
    levels: list[np.ndarray]


class FakePyramidWSI:
    def __init__(self, spec: PyramidSpec):
        self.spacings = spec.spacings
        self._levels = spec.levels
        self.shapes = tuple((arr.shape[1], arr.shape[0]) for arr in self._levels)
        self.downsamplings = tuple(
            self.shapes[0][0] / shape[0] for shape in self.shapes
        )

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

        patch = make_white_canvas(width, height)
        src = arr[y_level : y_level + height, x_level : x_level + width, :]
        patch[: src.shape[0], : src.shape[1], :] = src
        return patch


class FakeSlideReader:
    def __init__(self, path: Path, spec: PyramidSpec, *, backend_name: str = "asap"):
        self.path = Path(path)
        self._backend_name = backend_name
        self._spec = spec
        self.spacings = list(spec.spacings)
        self.level_dimensions = [
            (int(arr.shape[1]), int(arr.shape[0])) for arr in spec.levels
        ]
        self.level_downsamples = [
            (
                self.level_dimensions[0][0] / float(width),
                self.level_dimensions[0][1] / float(height),
            )
            for width, height in self.level_dimensions
        ]
        self.dimensions = self.level_dimensions[0]
        self.spacing = float(self.spacings[0])
        self.level_count = len(self.level_dimensions)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def read_level(self, level: int) -> np.ndarray:
        return np.asarray(self._spec.levels[level])

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        downsample = float(self.level_downsamples[level][0])
        arr = np.asarray(self._spec.levels[level])
        x_level = int(round(location[0] / downsample))
        y_level = int(round(location[1] / downsample))
        width = int(size[0])
        height = int(size[1])
        patch = make_white_canvas(width, height)
        src = arr[y_level : y_level + height, x_level : x_level + width, :]
        patch[: src.shape[0], : src.shape[1], :] = src
        return patch

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        del size
        return self.read_level(self.level_count - 1)

    def close(self) -> None:
        return None

    def __enter__(self) -> "FakeSlideReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()


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


class FakeReaderFactory:
    def __init__(self, slide_spec: PyramidSpec, mask_spec: PyramidSpec):
        self.slide_spec = slide_spec
        self.mask_spec = mask_spec

    def __call__(
        self,
        path: Path,
        backend: str = "asap",
        *,
        spacing_override: float | None = None,
        gpu_decode: bool = False,
    ) -> FakeSlideReader:
        del spacing_override, gpu_decode
        name = str(path).lower()
        spec = self.mask_spec if "mask" in name else self.slide_spec
        return FakeSlideReader(Path(path), spec, backend_name=backend)


def make_slide_spec() -> PyramidSpec:
    slide_l0 = np.ones((32, 32, 3), dtype=np.uint8) * 180
    slide_l1 = slide_l0[::2, ::2, :]
    return PyramidSpec(spacings=[1.0, 2.0], levels=[slide_l0, slide_l1])


def make_mask_spec(mask_l0: np.ndarray) -> PyramidSpec:
    mask_l0 = mask_l0.astype(np.uint8)
    mask_l1 = mask_l0[::2, ::2, :]
    return PyramidSpec(spacings=[2.0, 4.0], levels=[mask_l0, mask_l1])
