from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import wholeslidedata as wsd

from hs2p.wsi.reader import open_slide, select_level, select_level_for_downsample

AUTO_BACKEND = "auto"
PREFERRED_BACKENDS = ("cucim", "asap", "openslide")
CUCIM_SUPPORTED_SUFFIXES = {".svs", ".tif", ".tiff"}


@dataclass(frozen=True)
class BackendSelection:
    backend: str
    reason: str | None = None
    tried: tuple[str, ...] = ()


def _normalize_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path))


def coerce_wsd_path(path: Path | str, *, backend: str) -> Path | str:
    """Return a path object compatible with the requested WSD backend.

    CuCIM-backed WSD opens require plain strings, while the other backends
    accept pathlib objects.
    """
    if backend == "cucim":
        return str(path)
    return Path(path)


def _is_cucim_supported_format(wsi_path: Path) -> bool:
    suffix = wsi_path.suffix.lower()
    return suffix in CUCIM_SUPPORTED_SUFFIXES


@lru_cache(maxsize=256)
def _backend_can_open_slide(
    *,
    wsi_path: str,
    mask_path: str | None,
    backend: str,
) -> bool:
    try:
        wsd.WholeSlideImage(coerce_wsd_path(wsi_path, backend=backend), backend=backend)
        if mask_path is not None:
            wsd.WholeSlideImage(
                coerce_wsd_path(mask_path, backend=backend),
                backend=backend,
            )
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
            reason=None,
            tried=(requested_backend,),
        )

    normalized_wsi_path = _normalize_path(wsi_path)
    normalized_mask_path = _normalize_path(mask_path)
    tried: list[str] = []
    reasons: list[str] = []

    if _is_cucim_supported_format(wsi_path):
        tried.append("cucim")
        if _backend_can_open_slide(
            wsi_path=normalized_wsi_path,
            mask_path=normalized_mask_path,
            backend="cucim",
        ):
            return BackendSelection(
                backend="cucim",
                reason="selected CuCIM for auto backend",
                tried=tuple(tried),
            )
        reasons.append("CuCIM could not open the slide or mask")
    else:
        reasons.append(
            f"CuCIM skipped for unsupported slide suffix {wsi_path.suffix.lower() or '<none>'}"
        )

    for backend in PREFERRED_BACKENDS[1:]:
        tried.append(backend)
        if _backend_can_open_slide(
            wsi_path=normalized_wsi_path,
            mask_path=normalized_mask_path,
            backend=backend,
        ):
            reason = "; ".join(reasons + [f"selected {backend} for auto backend"])
            return BackendSelection(
                backend=backend,
                reason=reason,
                tried=tuple(tried),
            )
        reasons.append(f"{backend} could not open the slide or mask")

    raise RuntimeError(
        f"Unable to open {wsi_path} with any supported backend (tried: {', '.join(tried) or 'none'})"
    )


__all__ = [
    "AUTO_BACKEND",
    "BackendSelection",
    "coerce_wsd_path",
    "open_slide",
    "resolve_backend",
    "select_level",
    "select_level_for_downsample",
    "_backend_can_open_slide",
]
