from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

import numpy as np

from hs2p.wsi.backends import (
    ASAPReader,
    CuCIMReader,
    OpenSlideReader,
    VIPSReader,
    supports_cucim_path,
    supports_vips_path,
)
from hs2p.wsi.geometry import LevelSelection, select_level, select_level_for_downsample

AUTO_BACKEND = "auto"


@runtime_checkable
class SlideReader(Protocol):
    @property
    def backend_name(self) -> str: ...

    @property
    def dimensions(self) -> tuple[int, int]: ...

    @property
    def spacing(self) -> float: ...

    @property
    def spacings(self) -> list[float]: ...

    @property
    def level_count(self) -> int: ...

    @property
    def level_dimensions(self) -> list[tuple[int, int]]: ...

    @property
    def level_downsamples(self) -> list[tuple[float, float]]: ...

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        pad_missing: bool = True,
    ) -> np.ndarray: ...

    def read_level(self, level: int) -> np.ndarray: ...

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray: ...

    def close(self) -> None: ...

    def __enter__(self) -> SlideReader: ...
    def __exit__(self, *args: Any) -> None: ...


@runtime_checkable
class BatchRegionReader(SlideReader, Protocol):
    def read_regions(
        self,
        locations: list[tuple[int, int]],
        level: int,
        size: tuple[int, int],
        *,
        num_workers: int | None = None,
        pad_missing: bool = False,
    ) -> Iterable[np.ndarray]: ...


@dataclass(frozen=True)
class BackendSelection:
    backend: str
    reason: str | None = None
    tried: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BackendSpec:
    name: str
    opener: Callable[..., SlideReader]
    supports_path: Callable[[str | Path], bool]


def _supports_all_paths(path: str | Path) -> bool:
    del path
    return True


def _open_asap(path: str | Path, *, spacing_override: float | None = None, gpu_decode: bool = False) -> SlideReader:
    del gpu_decode
    return ASAPReader(path, spacing_override=spacing_override)


def _open_openslide(
    path: str | Path, *, spacing_override: float | None = None, gpu_decode: bool = False
) -> SlideReader:
    del gpu_decode
    return OpenSlideReader(path, spacing_override=spacing_override)


def _open_cucim(
    path: str | Path, *, spacing_override: float | None = None, gpu_decode: bool = False
) -> SlideReader:
    return CuCIMReader(path, spacing_override=spacing_override, gpu_decode=gpu_decode)


def _open_vips(
    path: str | Path, *, spacing_override: float | None = None, gpu_decode: bool = False
) -> SlideReader:
    del gpu_decode
    return VIPSReader(path, spacing_override=spacing_override)


_BACKENDS: dict[str, _BackendSpec] = {
    "cucim": _BackendSpec("cucim", _open_cucim, supports_cucim_path),
    "vips": _BackendSpec("vips", _open_vips, supports_vips_path),
    "openslide": _BackendSpec("openslide", _open_openslide, _supports_all_paths),
    "asap": _BackendSpec("asap", _open_asap, _supports_all_paths),
}

AUTO_BACKEND_ORDER = ("cucim", "vips", "openslide", "asap")


def open_slide(
    path: str | Path,
    backend: str = AUTO_BACKEND,
    *,
    spacing_override: float | None = None,
    gpu_decode: bool = False,
) -> SlideReader:
    backend = (backend or AUTO_BACKEND).strip().lower()
    if backend == AUTO_BACKEND:
        selection = resolve_backend(backend, wsi_path=Path(path))
        backend = selection.backend
    spec = _BACKENDS.get(backend)
    if spec is None:
        available = ", ".join(["auto", *_BACKENDS.keys()])
        raise ValueError(f"Unknown backend: '{backend}'. Available: {available}")
    return spec.opener(
        path,
        spacing_override=spacing_override,
        gpu_decode=gpu_decode,
    )


def _normalize_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path))


@lru_cache(maxsize=256)
def _backend_can_open_slide(
    *,
    wsi_path: str,
    mask_path: str | None,
    backend: str,
) -> bool:
    spec = _BACKENDS.get(backend)
    if spec is None:
        return False
    if not spec.supports_path(wsi_path):
        return False
    if mask_path is not None and not spec.supports_path(mask_path):
        return False
    try:
        slide = spec.opener(wsi_path)
        slide.close()
        if mask_path is not None:
            mask = spec.opener(mask_path)
            mask.close()
        return True
    except Exception:
        return False


def resolve_backend(
    requested_backend: str,
    *,
    wsi_path: Path,
    mask_path: Path | None = None,
) -> BackendSelection:
    requested_backend = (requested_backend or AUTO_BACKEND).strip().lower()
    if requested_backend != AUTO_BACKEND:
        return BackendSelection(
            backend=requested_backend,
            tried=(requested_backend,),
        )

    normalized_wsi_path = _normalize_path(wsi_path)
    normalized_mask_path = _normalize_path(mask_path)
    tried: list[str] = []
    reasons: list[str] = []
    for backend in AUTO_BACKEND_ORDER:
        spec = _BACKENDS[backend]
        if not spec.supports_path(wsi_path):
            reasons.append(
                f"{backend} skipped for unsupported slide suffix {wsi_path.suffix.lower() or '<none>'}"
            )
            continue
        if mask_path is not None and not spec.supports_path(mask_path):
            reasons.append(
                f"{backend} skipped for unsupported mask suffix {mask_path.suffix.lower() or '<none>'}"
            )
            continue
        tried.append(backend)
        if _backend_can_open_slide(
            wsi_path=normalized_wsi_path,
            mask_path=normalized_mask_path,
            backend=backend,
        ):
            selected_reason = f"selected {backend} for auto backend"
            if reasons:
                selected_reason = "; ".join([*reasons, selected_reason])
            return BackendSelection(
                backend=backend,
                reason=selected_reason,
                tried=tuple(tried),
            )
        reasons.append(f"{backend} could not open the slide or mask")

    raise RuntimeError(
        f"Unable to open {wsi_path} with any supported backend (tried: {', '.join(tried) or 'none'})"
    )


__all__ = [
    "AUTO_BACKEND",
    "AUTO_BACKEND_ORDER",
    "BackendSelection",
    "BatchRegionReader",
    "LevelSelection",
    "SlideReader",
    "open_slide",
    "resolve_backend",
    "select_level",
    "select_level_for_downsample",
]
