from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class CuImageReader:
    """Thin wrapper around cucim.CuImage that handles lazy opening and gpu_decode.

    Encapsulates:
    - setting ENABLE_CUSLIDE2=1 before importing cucim when gpu_decode=True
    - lazy CuImage construction (open on first read_region call)
    - passing device="cuda" to read_region with a TypeError fallback for older
      cucim versions that do not support the device kwarg
    """

    def __init__(self, image_path: Path, *, gpu_decode: bool = False):
        self._image_path = str(image_path)
        self._gpu_decode = gpu_decode
        self._cu_image = None

    def _ensure_open(self):
        if self._cu_image is not None:
            return
        if self._gpu_decode:
            os.environ["ENABLE_CUSLIDE2"] = "1"
        try:
            cucim = importlib.import_module("cucim")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "cucim is required for GPU-accelerated tile reading. "
                "Install it with: pip install cucim-cuXX (where XX matches your CUDA version)"
            ) from exc
        self._cu_image = cucim.CuImage(self._image_path)

    def read_region(self, locations, size, *, level: int, num_workers: int):
        self._ensure_open()
        kwargs: dict = {
            "level": level,
            "num_workers": max(1, num_workers),
        }
        if self._gpu_decode:
            kwargs["device"] = "cuda"
        try:
            return self._cu_image.read_region(locations, size, **kwargs)
        except TypeError:
            kwargs.pop("device", None)
            return self._cu_image.read_region(locations, size, **kwargs)
