from __future__ import annotations

from pathlib import Path

from hs2p.wsi.reader import (
    AUTO_BACKEND,
    BackendSelection,
    open_slide,
    resolve_backend,
    select_level,
    select_level_for_downsample,
)


def coerce_wsd_path(path: Path | str, *, backend: str) -> Path | str:
    """Return a path object compatible with the requested WSD backend.

    CuCIM-backed WSD opens require plain strings, while the other backends
    accept pathlib objects.
    """
    if backend == "cucim":
        return str(path)
    return Path(path)


__all__ = [
    "AUTO_BACKEND",
    "BackendSelection",
    "coerce_wsd_path",
    "open_slide",
    "resolve_backend",
    "select_level",
    "select_level_for_downsample",
]
