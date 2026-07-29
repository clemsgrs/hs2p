
import warnings
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
AUTO_BACKEND_ORDER = ("cucim", "vips", "openslide", "asap")


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
    ) -> np.ndarray: ...

    def read_level(self, level: int) -> np.ndarray: ...
    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray: ...
    def close(self) -> None: ...
    def __enter__(self) -> "SlideReader": ...
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
    ) -> Iterable[np.ndarray]: ...


@dataclass(frozen=True)
class BackendSelection:
    backend: str
    reason: str | None = None
    tried: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedBackends:
    """Slide- and mask-role backend resolution, keeping requested and resolved values apart.

    The single seam every read path shares (#163): the slide backend is resolved from the
    slide path alone and the mask backend from the mask path alone — neither role's
    openability probe influences the other. When a slide has no source mask, ``mask`` is
    ``None`` and both mask-provenance fields are ``None``: a maskless run never resolves or
    validates mask-backend availability.
    """

    slide: BackendSelection
    mask: BackendSelection | None
    requested_slide_backend: str
    requested_mask_backend: str | None

    @property
    def slide_backend(self) -> str:
        return self.slide.backend

    @property
    def mask_backend(self) -> str | None:
        return None if self.mask is None else self.mask.backend


def resolve_backends(
    *,
    requested_slide_backend: str,
    requested_mask_backend: str | None,
    wsi_path: str | Path,
    mask_path: str | Path | None = None,
    slide_spacing_override: float | None = None,
) -> ResolvedBackends:
    """Resolve slide and mask backends independently from their own paths.

    Slide ``auto`` and mask ``auto`` share the same openability-only selection policy
    (:func:`resolve_backend`); an explicit backend is authoritative and returned without a
    probe. A slide with no ``mask_path`` resolves only the slide role.
    """
    requested_slide = (requested_slide_backend or AUTO_BACKEND).strip().lower()
    slide_selection = resolve_backend(
        requested_slide,
        wsi_path=Path(wsi_path),
        spacing_override=slide_spacing_override,
    )
    if mask_path is None:
        return ResolvedBackends(
            slide=slide_selection,
            mask=None,
            requested_slide_backend=requested_slide,
            requested_mask_backend=None,
        )
    requested_mask = (
        requested_mask_backend if requested_mask_backend is not None else AUTO_BACKEND
    )
    requested_mask = (requested_mask or AUTO_BACKEND).strip().lower()
    mask_selection = resolve_backend(requested_mask, wsi_path=Path(mask_path))
    return ResolvedBackends(
        slide=slide_selection,
        mask=mask_selection,
        requested_slide_backend=requested_slide,
        requested_mask_backend=requested_mask,
    )


@dataclass(frozen=True)
class _BackendSpec:
    name: str
    opener: Callable[..., SlideReader]
    supports_path: Callable[[str | Path], bool]


def _supports_all_paths(path: str | Path) -> bool:
    del path
    return True


def _open_asap(
    path: str | Path,
    *,
    spacing_override: float | None = None,
) -> SlideReader:
    return ASAPReader(path, spacing_override=spacing_override)


def _open_openslide(
    path: str | Path,
    *,
    spacing_override: float | None = None,
) -> SlideReader:
    return OpenSlideReader(path, spacing_override=spacing_override)


def _open_cucim(
    path: str | Path,
    *,
    spacing_override: float | None = None,
    gpu_decode: bool = False,
) -> SlideReader:
    return CuCIMReader(path, spacing_override=spacing_override, gpu_decode=gpu_decode)


def _open_vips(
    path: str | Path,
    *,
    spacing_override: float | None = None,
) -> SlideReader:
    return VIPSReader(path, spacing_override=spacing_override)


_BACKENDS: dict[str, _BackendSpec] = {
    "cucim": _BackendSpec("cucim", _open_cucim, supports_cucim_path),
    "asap": _BackendSpec("asap", _open_asap, _supports_all_paths),
    "openslide": _BackendSpec("openslide", _open_openslide, _supports_all_paths),
    "vips": _BackendSpec("vips", _open_vips, supports_vips_path),
}


def open_slide(
    path: str | Path,
    backend: str = AUTO_BACKEND,
    *,
    spacing_override: float | None = None,
    gpu_decode: bool = False,
) -> SlideReader:
    backend = (backend or AUTO_BACKEND).strip().lower()
    if backend == AUTO_BACKEND:
        selection = resolve_backend(
            backend,
            wsi_path=Path(path),
            spacing_override=spacing_override,
        )
        backend = selection.backend
    spec = _BACKENDS.get(backend)
    if spec is None:
        available = ", ".join(["auto", *_BACKENDS.keys()])
        raise ValueError(f"Unknown backend: '{backend}'. Available: {available}")
    if backend == "cucim":
        return spec.opener(path, spacing_override=spacing_override, gpu_decode=gpu_decode)
    return spec.opener(path, spacing_override=spacing_override)


def _normalize_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path))


@lru_cache(maxsize=256)
def _backend_can_open_source(
    *,
    source_path: str,
    companion_path: str | None,
    backend: str,
    spacing_override: float | None = None,
) -> bool:
    spec = _BACKENDS.get(backend)
    if spec is None:
        return False
    if not spec.supports_path(source_path):
        return False
    if companion_path is not None and not spec.supports_path(companion_path):
        return False
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"^Slide spacing override conflict:",
                category=UserWarning,
            )
            source = spec.opener(source_path, spacing_override=spacing_override)
            source.close()
            if companion_path is not None:
                companion = spec.opener(companion_path)
                companion.close()
        return True
    except Exception:
        return False


def resolve_backend(
    requested_backend: str,
    *,
    wsi_path: Path,
    mask_path: Path | None = None,
    spacing_override: float | None = None,
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

    display_names = {"cucim": "cuCIM", "vips": "VIPS"}
    for backend in AUTO_BACKEND_ORDER:
        spec = _BACKENDS[backend]
        display_name = display_names.get(backend, backend)
        unsupported_path = next(
            (
                path
                for path in (normalized_wsi_path, normalized_mask_path)
                if path is not None and not spec.supports_path(path)
            ),
            None,
        )
        if unsupported_path is not None:
            suffix = Path(unsupported_path).suffix.lower() or "<none>"
            reasons.append(
                f"{display_name} skipped for unsupported path suffix {suffix}"
            )
            continue
        tried.append(backend)
        if _backend_can_open_source(
            source_path=normalized_wsi_path,
            companion_path=normalized_mask_path,
            backend=backend,
            spacing_override=spacing_override,
        ):
            reason = "; ".join(
                reasons + [f"selected {display_name} for auto backend"]
            )
            return BackendSelection(
                backend=backend,
                reason=reason,
                tried=tuple(tried),
            )
        reasons.append(f"{display_name} could not open the source")

    raise RuntimeError(
        f"Unable to open {wsi_path} with any supported backend (tried: {', '.join(tried) or 'none'})"
    )


def open_mask_reader(
    mask_path: str | Path, *, mask_backend: str = AUTO_BACKEND
) -> tuple[SlideReader, str]:
    """Open a source mask through its own resolved backend, with actionable failures.

    Resolves the mask backend from the mask path alone (``auto`` probes openability; a concrete
    name is authoritative) and opens the reader, both inside one ``try`` so any failure —
    resolution *or* open — is reraised with context naming the mask path and the requested
    backend (and the resolved backend when it got that far). This is the centralized mask-open
    seam for the visualization/overlay and :class:`~hs2p.wsi.wsi.WSI` attached-mask paths (#163),
    mirroring :func:`hs2p.tiling.mask._raise_mask_decode_error`: a ``ValueError`` cause reraises
    as ``ValueError``, anything else as ``RuntimeError``.

    Returns ``(reader, resolved_backend)``.
    """
    resolved = mask_backend
    try:
        resolved = resolve_backend(mask_backend, wsi_path=Path(mask_path)).backend
        reader = open_slide(mask_path, backend=resolved)
    except Exception as error:
        message = (
            f"Mask open failed for path={Path(mask_path)} with backend={resolved} "
            f"(requested={mask_backend}): {error}. "
            "Select another mask backend or verify the mask file."
        )
        if isinstance(error, ValueError):
            raise ValueError(message) from error
        raise RuntimeError(message) from error
    return reader, resolved


__all__ = [
    "AUTO_BACKEND",
    "AUTO_BACKEND_ORDER",
    "BackendSelection",
    "BatchRegionReader",
    "LevelSelection",
    "ResolvedBackends",
    "SlideReader",
    "open_mask_reader",
    "open_slide",
    "resolve_backend",
    "resolve_backends",
    "select_level",
    "select_level_for_downsample",
]
