
from pathlib import Path

from hs2p.wsi.reader import (
    AUTO_BACKEND,
    BackendSelection,
    ResolvedBackends,
    open_mask_reader,
    open_slide,
    resolve_backend,
    resolve_backends,
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
    "ResolvedBackends",
    "coerce_wsd_path",
    "open_mask_reader",
    "open_slide",
    "resolve_backend",
    "resolve_backends",
    "select_level",
    "select_level_for_downsample",
]
