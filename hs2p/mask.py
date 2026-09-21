"""First-class source masks: closed label semantics, alignment, and validated reads."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hs2p.configs.resolvers import validate_pixel_mapping
from hs2p.wsi.geometry import compute_level_spacings, select_level_for_spacing_read
from hs2p.wsi.reader import AUTO_BACKEND, SlideReader, open_slide, resolve_backend
from hs2p.wsi.types import pixel_values

MAX_LABEL_ID = 255
# Decoding a native mask level larger than this can exhaust memory (256 Mpx is already
# 256 MB for uint8, more for wider dtypes and backend buffers), so reads fail fast
# instead. A fixed safety invariant, not configuration; the same cap as the legacy
# ``hs2p.tiling.mask.MAX_MASK_READ_PX``.
MAX_MASK_READ_PX = 256_000_000
# Level selection treats a level within 1% of the requested spacing as exact,
# independently of ``tiling.params.tolerance``.
LEVEL_SPACING_TOLERANCE = 0.01
# File spacing is only cross-checked against the dimension-ratio spacing.
SPACING_WARN_THRESHOLD = 0.01
SPACING_FAIL_THRESHOLD = 0.05

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class TissueLabels:
    """Binary tissue semantics: the explicit background and tissue label IDs."""

    background: int
    tissue: int

    def __post_init__(self) -> None:
        for name, value in (("background", self.background), ("tissue", self.tissue)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"TissueLabels {name} must be an integer label ID, got {value!r}"
                )
            if value < 0 or value > MAX_LABEL_ID:
                raise ValueError(
                    f"TissueLabels {name}={value} is outside the supported label "
                    f"range [0, {MAX_LABEL_ID}]"
                )
        if self.background == self.tissue:
            raise ValueError(
                "TissueLabels background and tissue must be distinct IDs, "
                f"both are {self.tissue}"
            )

    @property
    def ids(self) -> frozenset[int]:
        return frozenset({self.background, self.tissue})


@dataclass(frozen=True, kw_only=True)
class AnnotationLabels:
    """Annotation semantics: the complete label-name to label-IDs mapping.

    Built from the configuration ``pixel_mapping`` shape and stored normalized, every
    label owning a tuple of IDs.
    """

    pixel_mapping: dict[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        validate_pixel_mapping(self.pixel_mapping)
        object.__setattr__(
            self,
            "pixel_mapping",
            {
                label: tuple(int(value) for value in pixel_values(entry))
                for label, entry in self.pixel_mapping.items()
            },
        )

    @property
    def ids(self) -> frozenset[int]:
        return frozenset(
            value for values in self.pixel_mapping.values() for value in values
        )


def _max_scale(reference_size: int, mask_size: int) -> float:
    """Largest scale at which ``mask_size`` is within one pixel of ``reference_size / scale``."""
    return reference_size / (mask_size - 1) if mask_size > 1 else math.inf


@dataclass(frozen=True, eq=False)
class MaskRead:
    """One validated read: 2-D ``uint8`` labels plus the mask level and spacing read."""

    labels: np.ndarray
    read_level: int
    read_spacing_um: float


class Mask:
    """An externally supplied, source-backed mask with one closed label semantics.

    Owns one open reader for ``path``: ``auto`` resolves the backend from the mask path
    alone, a concrete backend is authoritative. The source opens without requiring
    spacing metadata, so flat PNG/JPEG and untagged TIFF masks are supported.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        labels: TissueLabels | AnnotationLabels,
        backend: str = AUTO_BACKEND,
    ) -> None:
        if not isinstance(labels, (TissueLabels, AnnotationLabels)):
            raise ValueError(
                "Mask labels must be TissueLabels or AnnotationLabels, "
                f"got {type(labels).__name__}"
            )
        self._path = Path(path)
        self._labels = labels
        self._backend = (backend or AUTO_BACKEND).strip().lower()
        try:
            self._backend = resolve_backend(
                self._backend, wsi_path=self._path, require_spacing=False
            ).backend
            self._reader: SlideReader | None = open_slide(
                self._path, self._backend, require_spacing=False
            )
        except Exception as error:
            message = (
                f"Mask open failed for path={self._path} with "
                f"backend={self._backend}: {error}"
            )
            if isinstance(error, ValueError):
                raise ValueError(message) from error
            raise RuntimeError(message) from error

    @property
    def path(self) -> Path:
        return self._path

    @property
    def labels(self) -> TissueLabels | AnnotationLabels:
        return self._labels

    @property
    def backend(self) -> str:
        """The concrete backend that opened this mask."""
        return self._backend

    def _require_reader(self) -> SlideReader:
        if self._reader is None:
            raise ValueError(f"Mask is closed: path={self._path}")
        return self._reader

    def align_to(
        self,
        *,
        reference_spacing_um: float,
        reference_dimensions: tuple[int, int],
    ) -> AlignedMask:
        """Bind this mask to a reference level-0 grid it must fully cover.

        The mask-to-reference dimension ratio is authoritative: it must describe one
        scale on both axes, within one mask level-0 pixel of rounding per axis, and it
        defines the effective mask spacing. File spacing, when present, is only
        cross-checked (warn from 1%, fail above 5%); a mask without spacing metadata is
        guarded by the shape check alone.
        """
        reader = self._require_reader()
        reference_spacing_um = float(reference_spacing_um)
        if not math.isfinite(reference_spacing_um) or reference_spacing_um <= 0:
            raise ValueError(
                "reference_spacing_um must be a finite positive value, "
                f"got {reference_spacing_um!r}"
            )
        reference_width, reference_height = (int(v) for v in reference_dimensions)
        if reference_width <= 0 or reference_height <= 0:
            raise ValueError(
                "reference_dimensions must be positive, "
                f"got {reference_width}x{reference_height}"
            )

        mask_width, mask_height = (int(v) for v in reader.level_dimensions[0])
        effective_spacing_um = reference_spacing_um * reference_width / mask_width
        file_spacing_um = reader.native_spacing
        context = (
            f"path={self._path} with backend={self._backend}: "
            f"mask dimensions {mask_width}x{mask_height}, "
            f"reference dimensions {reference_width}x{reference_height}, "
            f"effective spacing {effective_spacing_um:.4f} um/px"
        )
        if file_spacing_um is not None:
            context += f", file spacing {float(file_spacing_um):.4f} um/px"

        # One scale ``s`` must satisfy |mask - reference / s| <= 1 on both axes.
        scale_min = max(
            reference_width / (mask_width + 1), reference_height / (mask_height + 1)
        )
        scale_max = min(
            _max_scale(reference_width, mask_width),
            _max_scale(reference_height, mask_height),
        )
        if scale_min > scale_max:
            raise ValueError(
                f"Mask alignment failed for {context}. The mask does not cover the "
                "reference canvas at one scale; masks must span the full reference."
            )

        if file_spacing_um is not None:
            difference = (
                abs(float(file_spacing_um) - effective_spacing_um)
                / effective_spacing_um
            )
            if difference > SPACING_FAIL_THRESHOLD:
                raise ValueError(
                    f"Mask alignment failed for {context}. File spacing differs from "
                    f"the effective spacing by {difference:.1%}, above the "
                    f"{SPACING_FAIL_THRESHOLD:.0%} limit."
                )
            if difference >= SPACING_WARN_THRESHOLD:
                logger.warning(
                    "Mask spacing disagreement for %s. File spacing differs from the "
                    "effective spacing by %s; the effective spacing is used and "
                    "recorded.",
                    context,
                    f"{difference:.1%}",
                )

        return AlignedMask(
            mask=self,
            reference_spacing_um=reference_spacing_um,
            reference_dimensions=(reference_width, reference_height),
            level_spacings_um=tuple(
                compute_level_spacings(
                    level0_spacing_um=effective_spacing_um,
                    level_downsamples=reader.level_downsamples,
                )
            ),
        )

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self) -> Mask:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


@dataclass(frozen=True, eq=False)
class AlignedMask:
    """A mask bound to a reference level-0 grid; reads fail once the mask is closed."""

    mask: Mask
    reference_spacing_um: float
    reference_dimensions: tuple[int, int]
    # Effective spacing of every mask level, derived from the dimension ratio.
    level_spacings_um: tuple[float, ...]

    def read_full(
        self,
        *,
        target_spacing_um: float,
        target_dimensions: tuple[int, int],
    ) -> MaskRead:
        """Read the full canvas as ``target_dimensions`` labels at ``target_spacing_um``.

        Reads the coarsest mask level not meaningfully coarser than the target (level 0
        when none is fine enough), validates the native decode, then nearest-neighbor
        resizes it to the exact requested dimensions.
        """
        mask = self.mask
        reader = mask._require_reader()
        target_width, target_height = (int(v) for v in target_dimensions)
        if target_width <= 0 or target_height <= 0:
            raise ValueError(
                f"target_dimensions must be positive, got {target_width}x{target_height}"
            )
        selection = select_level_for_spacing_read(
            requested_spacing_um=float(target_spacing_um),
            level0_spacing_um=self.level_spacings_um[0],
            level_downsamples=reader.level_downsamples,
            tolerance=LEVEL_SPACING_TOLERANCE,
            content_kind="label",
        )
        level = selection.level
        width, height = (int(v) for v in reader.level_dimensions[level])
        context = f"path={mask.path} with backend={mask.backend}"
        if width * height > MAX_MASK_READ_PX:
            raise ValueError(
                f"Mask read refused for {context}: the level selected for "
                f"{float(target_spacing_um):.4f} um/px is level {level} at "
                f"{width}x{height} ({width * height / 1e6:.0f} Mpx), exceeding the "
                f"{MAX_MASK_READ_PX / 1e6:.0f} Mpx read cap. The mask likely lacks a "
                "pyramid level near that spacing; regenerate it as a multi-resolution "
                "pyramidal TIFF."
            )
        try:
            native = np.asarray(reader.read_region((0, 0), level, (width, height)))
        except Exception as error:
            message = f"Mask decode failed for {context} at level {level}: {error}"
            if isinstance(error, ValueError):
                raise ValueError(message) from error
            raise RuntimeError(message) from error

        try:
            labels = _validated_labels(native, declared_ids=mask.labels.ids)
        except ValueError as error:
            raise ValueError(
                f"Mask read produced invalid labels for {context} at level {level}: "
                f"{error}"
            ) from error
        if labels.shape != (target_height, target_width):
            labels = cv2.resize(
                labels, (target_width, target_height), interpolation=cv2.INTER_NEAREST
            )
        # A read-only view: the decode may share memory with a backend-owned buffer.
        labels = labels.view()
        labels.flags.writeable = False
        return MaskRead(
            labels=labels,
            read_level=level,
            read_spacing_um=selection.read_spacing_um,
        )


def _validated_labels(native: np.ndarray, *, declared_ids: frozenset[int]) -> np.ndarray:
    """Return one native decode as 2-D ``uint8`` labels, or raise ``ValueError``.

    Runs before any resampling, so an invalid value can never be dropped by
    nearest-neighbor downsampling and escape validation.
    """
    if not np.issubdtype(native.dtype, np.integer):
        raise ValueError(f"expected an integer dtype, got {native.dtype}")
    if native.ndim == 3:
        first = native[..., 0]
        if any(
            not np.array_equal(first, native[..., channel])
            for channel in range(1, native.shape[-1])
        ):
            raise ValueError(
                "channels differ; a label mask must be single-channel or replicate "
                "one channel"
            )
        native = first
    if native.ndim != 2:
        raise ValueError(f"expected a 2-D label raster, got shape {native.shape}")
    min_value, max_value = int(native.min()), int(native.max())
    if min_value < 0 or max_value > MAX_LABEL_ID:
        raise ValueError(
            f"label values span [{min_value}, {max_value}], outside [0, {MAX_LABEL_ID}]"
        )
    labels = np.ascontiguousarray(native, dtype=np.uint8)
    # A 256-bin histogram finds the present IDs without sorting the whole raster.
    present = np.flatnonzero(cv2.calcHist([labels], [0], None, [256], [0, 256]))
    undeclared = sorted(set(present.tolist()) - declared_ids)
    if undeclared:
        raise ValueError(
            f"undeclared label IDs {undeclared}; declared {sorted(declared_ids)}"
        )
    return labels


__all__ = ["AlignedMask", "AnnotationLabels", "Mask", "MaskRead", "TissueLabels"]
