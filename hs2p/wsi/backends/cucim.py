import faulthandler
import importlib
import os
import threading
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from hs2p.wsi.backends.common import (
    paste_region,
    resolve_level0_spacing,
    resolve_padded_read_bounds,
)
from hs2p.wsi.geometry import compute_level_spacings


class _suppress_native_stderr:
    """Context manager that silences native (C/C++) stderr output.

    cuCIM's cuFile/cuInit layers print warnings like ``cuInit Failed`` and
    ``cuFile initialization failed`` directly to file-descriptor 2.  No
    Python-level logging filter can catch these, so we temporarily point
    fd 2 at ``/dev/null`` and restore it on exit.

    A process that dies of a signal in here, as cuFile's assertion aborts it,
    would leave nothing on stderr, faulthandler output included (#278). So
    while fd 2 is redirected, faulthandler writes to a duplicate of the real
    stderr: a fatal signal is named there, with the Python traceback. The
    native message itself still goes to ``/dev/null``.

    Contexts nest (the batched streaming reads wrap the reader's own) and may
    overlap across threads, so the redirect is shared: the outermost context
    saves the real stderr and the last one to exit restores it. An inner
    context never takes ``/dev/null`` for the real stderr.
    """

    _lock = threading.Lock()
    _depth = 0
    _stderr_fd = -1
    _devnull_fd = -1
    _faulthandler_was_enabled = False

    def __enter__(self):
        cls = _suppress_native_stderr
        with cls._lock:
            if cls._depth == 0:
                cls._faulthandler_was_enabled = faulthandler.is_enabled()
                cls._devnull_fd = os.open(os.devnull, os.O_WRONLY)
                cls._stderr_fd = os.dup(2)
                faulthandler.enable(file=cls._stderr_fd, all_threads=True)
                os.dup2(cls._devnull_fd, 2)
            cls._depth += 1
        return self

    def __exit__(self, *exc):
        cls = _suppress_native_stderr
        with cls._lock:
            cls._depth -= 1
            if cls._depth == 0:
                os.dup2(cls._stderr_fd, 2)
                if cls._faulthandler_was_enabled:
                    # faulthandler cannot report the file it was enabled on: put it
                    # back on fd 2, where ``python -X faulthandler`` enables it.
                    faulthandler.enable(file=2, all_threads=True)
                else:
                    faulthandler.disable()
                os.close(cls._stderr_fd)
                os.close(cls._devnull_fd)
        return False


CUCIM_SUPPORTED_SUFFIXES = {".svs", ".tif", ".tiff"}

# Importing cuCIM loads NVIDIA cuFile, which starts a thread pool for each visible GPU and
# aborts the process (``CUFileThreadPoolWorker::run(): Assertion `0' failed``) when one GPU
# is too full for a worker to get a CUDA context (#278). Slide reads never go through
# cuFile: cuCIM's TIFF reader uses ``pread``. The shipped config holds one key,
# ``{"execution": {"max_io_threads": 0}}``, which turns the pool off; cuFile applies a
# partial config over its defaults. A ``CUFILE_ENV_PATH_JSON`` the user set is kept.
CUFILE_CONFIG_PATH = Path(__file__).with_name("cufile.json")


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


def _extract_spacing_from_metadata(metadata: dict[str, Any]) -> float | None:
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
    return None


class CuCIMReader:
    def __init__(
        self,
        path: str | Path,
        *,
        spacing_override: float | None = None,
        gpu_decode: bool = False,
        require_spacing: bool = True,
    ):
        if gpu_decode:
            os.environ["ENABLE_CUSLIDE2"] = "1"
        # cuFile reads the variable once, when cuCIM's library loads: set it before the
        # import, the only one in hs2p. It is process-wide (see CUFILE_CONFIG_PATH).
        os.environ.setdefault("CUFILE_ENV_PATH_JSON", str(CUFILE_CONFIG_PATH))
        with _suppress_native_stderr():
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
        self._spacing = resolve_level0_spacing(
            path=path,
            backend=self.backend_name,
            native_spacing=self.native_spacing,
            spacing_override=spacing_override,
            require_spacing=require_spacing,
        )
        # A source opened without spacing (``require_spacing=False``) has no level spacings.
        self._spacings = (
            []
            if self._spacing is None
            else compute_level_spacings(
                level0_spacing_um=self._spacing,
                level_downsamples=self._level_downsamples,
            )
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

    # Both calls into ``CuImage.read_region`` name every argument as an explicit keyword
    # and never expand a dict: with cuCIM 26.06 on Python 3.11.0rc1, a
    # ``read_region(**kwargs)`` call segfaults the process on its eighth single read or
    # its second or third batched read (#281). An argument hs2p does not set gets
    # cuCIM's own default, ``num_workers=0`` and ``device="cpu"``.
    def _read_region(self, location, size, *, level: int, num_workers: int = 1):
        with _suppress_native_stderr():
            return self._slide.read_region(
                location=(int(location[0]), int(location[1])),
                size=(int(size[0]), int(size[1])),
                level=int(level),
                num_workers=int(num_workers) if num_workers > 1 else 0,
                device="cuda" if self._gpu_decode else "cpu",
            )

    def read_level(self, level: int) -> np.ndarray:
        dims = self._level_dimensions[level]
        return _as_rgb_uint8(self._read_region((0, 0), dims, level=level))

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
        return paste_region(
            canvas=bounds.canvas, region=region, paste_offset=bounds.paste_offset
        )

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

        with _suppress_native_stderr():
            regions = self._slide.read_region(
                location=[(int(x), int(y)) for x, y in locations],
                size=requested_size,
                level=int(level),
                num_workers=max(1, int(num_workers or 1)),
                device="cuda" if self._gpu_decode else "cpu",
            )
        for bounds, region in zip(bounds_per_location, regions):
            arr = _as_rgb_uint8(region)
            yield paste_region(
                canvas=bounds.canvas, region=arr, paste_offset=bounds.paste_offset
            )

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
