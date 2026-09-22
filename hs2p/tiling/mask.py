from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import cv2
import numpy as np

from hs2p.configs import SegmentationConfig
from hs2p.mask import AnnotationLabels, Mask, TissueLabels
from hs2p.segmentation import segment_tissue_image
from hs2p.tiling.contours import _normalize_level_downsamples
from hs2p.tiling.result import ResolvedAnnotationMasks, ResolvedTissueMask, Sam2Thumbnail
from hs2p.wsi.reader import AUTO_BACKEND, select_level, select_level_for_downsample
from hs2p.wsi.types import PixelMapping

DEFAULT_SAM2_THUMBNAIL_SPACING_UM = 8.0
DEFAULT_SAM2_THUMBNAIL_TOLERANCE = 0.05
logger = logging.getLogger(__name__)


def tissue_labels_from_pixel_mapping(
    pixel_mapping: PixelMapping | None = None,
) -> TissueLabels:
    """Tissue semantics declared by a configuration ``pixel_mapping``: its ``background``
    and ``tissue`` IDs, defaulting to 0 and 1."""
    pixel_mapping = pixel_mapping or {}
    return TissueLabels(
        background=pixel_mapping.get("background", 0),
        tissue=pixel_mapping.get("tissue", 1),
    )


def open_tissue_mask(
    mask_path: str | Path | None,
    *,
    pixel_mapping: PixelMapping | None = None,
    backend: str = AUTO_BACKEND,
) -> AbstractContextManager[Mask | None]:
    """Context manager over the tissue :class:`~hs2p.mask.Mask` a configuration-driven
    caller opens for ``mask_path``; yields ``None`` when there is no source mask."""
    if mask_path is None:
        return nullcontext()
    return Mask(
        path=mask_path,
        labels=tissue_labels_from_pixel_mapping(pixel_mapping),
        backend=backend,
    )


def open_annotation_mask(
    mask_path: str | Path | None,
    *,
    pixel_mapping: PixelMapping | None,
    backend: str = AUTO_BACKEND,
) -> AbstractContextManager[Mask | None]:
    """Context manager over the annotation :class:`~hs2p.mask.Mask` for ``mask_path``,
    declaring the full ``pixel_mapping`` vocabulary; yields ``None`` without a source mask
    or a mapping."""
    if mask_path is None or pixel_mapping is None:
        return nullcontext()
    return Mask(
        path=mask_path,
        labels=AnnotationLabels(pixel_mapping=pixel_mapping),
        backend=backend,
    )


def _read_binary_tissue_mask(
    *, mask: Mask, slide, seg_level: int
) -> tuple[np.ndarray, int, float]:
    """Read ``mask`` on the slide's ``seg_level`` grid as a 255 (tissue) / 0 raster, with
    the mask level and effective spacing that were read."""
    if not isinstance(mask.labels, TissueLabels):
        raise ValueError(
            f"A tissue mask must declare TissueLabels, got {type(mask.labels).__name__}"
        )
    seg_spacing_um = float(slide.spacing) * float(
        _normalize_level_downsamples(slide.level_downsamples)[seg_level]
    )
    read = mask.align_to(
        reference_spacing_um=float(slide.spacing),
        reference_dimensions=slide.level_dimensions[0],
    ).read_full(
        target_spacing_um=seg_spacing_um,
        target_dimensions=slide.level_dimensions[seg_level],
    )
    tissue_mask = np.where(read.labels == mask.labels.tissue, 255, 0).astype(np.uint8)
    return tissue_mask, read.read_level, read.read_spacing_um


def load_precomputed_tissue_mask(
    *,
    mask_path: str | Path,
    slide,
    seg_level: int,
    tissue_value: int,
    background_value: int = 0,
    mask_backend: str | None = None,
) -> tuple[np.ndarray, int, float]:
    """Path-based tissue loader kept until the legacy mask interfaces are removed; opens a
    :class:`~hs2p.mask.Mask` and delegates to the same read as :func:`resolve_tissue_mask`."""
    with Mask(
        path=mask_path,
        labels=TissueLabels(background=background_value, tissue=tissue_value),
        backend=mask_backend if mask_backend is not None else AUTO_BACKEND,
    ) as mask:
        return _read_binary_tissue_mask(mask=mask, slide=slide, seg_level=seg_level)


def prepare_sam2_thumbnail(
    *,
    slide,
    target_spacing_um: float = DEFAULT_SAM2_THUMBNAIL_SPACING_UM,
    tolerance: float = DEFAULT_SAM2_THUMBNAIL_TOLERANCE,
) -> Sam2Thumbnail:
    normalized_downsamples = _normalize_level_downsamples(slide.level_downsamples)
    level_sel = select_level(
        requested_spacing_um=float(target_spacing_um),
        level0_spacing_um=float(slide.spacing),
        level_downsamples=[
            (float(downsample), float(downsample))
            for downsample in normalized_downsamples
        ],
        tolerance=float(tolerance),
    )
    seg_size = slide.level_dimensions[level_sel.level]
    seg_image = np.asarray(slide.read_region((0, 0), level_sel.level, seg_size))
    if seg_image.ndim != 3:
        raise ValueError(
            f"Expected SAM2 thumbnail to be RGB, got array with shape {seg_image.shape}"
        )
    if level_sel.is_within_tolerance:
        return Sam2Thumbnail(
            image=seg_image,
            seg_level=level_sel.level,
            seg_spacing_um=float(level_sel.read_spacing_um),
            source_spacing_um=float(level_sel.read_spacing_um),
            resized=False,
        )

    target_width = max(
        1,
        int(round(float(slide.dimensions[0]) * float(slide.spacing) / float(target_spacing_um))),
    )
    target_height = max(
        1,
        int(round(float(slide.dimensions[1]) * float(slide.spacing) / float(target_spacing_um))),
    )
    if target_width != int(seg_image.shape[1]) or target_height != int(seg_image.shape[0]):
        interpolation = (
            cv2.INTER_AREA
            if target_width < int(seg_image.shape[1]) or target_height < int(seg_image.shape[0])
            else cv2.INTER_CUBIC
        )
        seg_image = cv2.resize(
            seg_image,
            (target_width, target_height),
            interpolation=interpolation,
        )
    return Sam2Thumbnail(
        image=seg_image,
        seg_level=level_sel.level,
        seg_spacing_um=float(target_spacing_um),
        source_spacing_um=float(level_sel.read_spacing_um),
        resized=True,
    )


def resolve_tissue_mask(
    *,
    slide,
    sample_id: str | None = None,
    tissue_method: str | None = None,
    mask: Mask | None = None,
    sthresh: int = 8,
    sthresh_up: int = 255,
    mthresh: int = 7,
    close: int = 4,
    seg_downsample: int = 64,
    sam2_checkpoint_path: str | Path | None = None,
    sam2_config_path: str | Path | None = None,
    sam2_device: str = "cpu",
    requested_mask_backend: str | None = None,
) -> ResolvedTissueMask:
    """Resolve the slide's tissue mask: from ``mask`` (an open tissue
    :class:`~hs2p.mask.Mask`, whose lifetime the caller owns) when given, otherwise by
    segmenting with ``tissue_method``.

    ``requested_mask_backend`` is caller-owned provenance; the mask only knows the concrete
    backend it opened, which is recorded as the request when none is supplied.
    """
    if mask is not None:
        normalized_downsamples = _normalize_level_downsamples(slide.level_downsamples)
        seg_level = select_level_for_downsample(
            float(seg_downsample),
            [(float(ds), float(ds)) for ds in normalized_downsamples],
        )
        seg_spacing_um = float(slide.spacing) * float(normalized_downsamples[seg_level])
        tissue_mask, mask_level, mask_spacing_um = _read_binary_tissue_mask(
            mask=mask,
            slide=slide,
            seg_level=seg_level,
        )
        if not np.any(tissue_mask):
            logger.warning(
                "Empty precomputed tissue mask: sample_id=%s mask_path=%s backend=%s "
                "mask_level=%s tissue_value=%s. The mask backend returned a valid mask "
                "containing no configured tissue value; the mask may be genuinely empty or "
                "incorrectly decoded — verify it is intentionally empty, select another mask "
                "backend, or regenerate the mask.",
                sample_id if sample_id is not None else "<unknown>",
                mask.path,
                mask.backend,
                mask_level,
                mask.labels.tissue,
            )
        return ResolvedTissueMask(
            tissue_mask=tissue_mask,
            tissue_method="precomputed_mask",
            requested_seg_downsample=int(seg_downsample),
            seg_downsample=max(1, int(round(seg_spacing_um / float(slide.spacing)))),
            seg_level=seg_level,
            seg_spacing_um=seg_spacing_um,
            mask_path=mask.path,
            tissue_mask_tissue_value=mask.labels.tissue,
            mask_level=mask_level,
            mask_spacing_um=mask_spacing_um,
            mask_backend=mask.backend,
            requested_mask_backend=(
                requested_mask_backend
                if requested_mask_backend is not None
                else mask.backend
            ),
        )

    if not tissue_method:
        raise ValueError(
            "tissue_method is required when no precomputed tissue mask is provided"
        )

    if str(tissue_method).lower() == "sam2":
        thumbnail = prepare_sam2_thumbnail(slide=slide)
        seg_image = thumbnail.image
        seg_level = thumbnail.seg_level
        seg_spacing_um = thumbnail.seg_spacing_um
    else:
        normalized_downsamples = _normalize_level_downsamples(slide.level_downsamples)
        seg_level = select_level_for_downsample(
            float(seg_downsample),
            [(float(ds), float(ds)) for ds in normalized_downsamples],
        )
        seg_spacing_um = float(slide.spacing) * float(normalized_downsamples[seg_level])
        seg_size = slide.level_dimensions[seg_level]
        seg_image = np.asarray(slide.read_region((0, 0), seg_level, seg_size))
    effective_downsample = max(1, int(round(seg_spacing_um / float(slide.spacing))))

    segmentation_config = SegmentationConfig(
        method=tissue_method,
        downsample=effective_downsample,
        sthresh=sthresh,
        sthresh_up=sthresh_up,
        mthresh=mthresh,
        close=close,
        sam2_checkpoint_path=(
            Path(sam2_checkpoint_path) if sam2_checkpoint_path is not None else None
        ),
        sam2_config_path=(
            Path(sam2_config_path) if sam2_config_path is not None else None
        ),
        sam2_device=sam2_device,
    )
    mask = segment_tissue_image(
        seg_image,
        config=segmentation_config,
    )
    return ResolvedTissueMask(
        tissue_mask=mask,
        tissue_method=segmentation_config.method,
        requested_seg_downsample=int(seg_downsample),
        seg_downsample=effective_downsample,
        seg_level=seg_level,
        seg_spacing_um=seg_spacing_um,
    )


def _read_annotation_label_mask(
    *, mask: Mask, slide, seg_level: int
) -> tuple[np.ndarray, int, float]:
    """Read ``mask`` on the slide's ``seg_level`` grid as its validated label raster, with
    the mask level and effective spacing that were read."""
    if not isinstance(mask.labels, AnnotationLabels):
        raise ValueError(
            "An annotation mask must declare AnnotationLabels, "
            f"got {type(mask.labels).__name__}"
        )
    seg_spacing_um = float(slide.spacing) * float(
        _normalize_level_downsamples(slide.level_downsamples)[seg_level]
    )
    read = mask.align_to(
        reference_spacing_um=float(slide.spacing),
        reference_dimensions=slide.level_dimensions[0],
    ).read_full(
        target_spacing_um=seg_spacing_um,
        target_dimensions=slide.level_dimensions[seg_level],
    )
    return read.labels, read.read_level, read.read_spacing_um


def load_annotation_label_mask(
    *,
    mask_path: str | Path,
    slide,
    seg_level: int,
    valid_values: set[int],
    mask_backend: str | None = None,
) -> tuple[np.ndarray, int, float]:
    """Path-based annotation loader kept until the legacy mask interfaces are removed; opens
    a :class:`~hs2p.mask.Mask` declaring each of ``valid_values`` as its own label and
    delegates to the same read as :func:`resolve_annotation_masks`."""
    with Mask(
        path=mask_path,
        labels=AnnotationLabels(
            pixel_mapping={str(value): int(value) for value in sorted(valid_values)}
        ),
        backend=mask_backend if mask_backend is not None else AUTO_BACKEND,
    ) as mask:
        return _read_annotation_label_mask(mask=mask, slide=slide, seg_level=seg_level)


def _configured_pixel_mapping(labels: AnnotationLabels) -> PixelMapping:
    """The configuration shape of ``labels``: one value, or the list a label merges."""
    return {
        name: list(values) if len(values) > 1 else values[0]
        for name, values in labels.pixel_mapping.items()
    }


def resolve_annotation_masks(
    *,
    slide,
    mask: Mask,
    seg_downsample: int = 64,
    tissue_method: str = "precomputed_mask",
    requested_mask_backend: str | None = None,
) -> ResolvedAnnotationMasks:
    """Read ``mask`` (an open annotation :class:`~hs2p.mask.Mask`, whose lifetime the caller
    owns) into one binary mask per declared label at ``seg_downsample``.

    The annotation counterpart of :func:`resolve_tissue_mask`'s precomputed path. The
    mask's :class:`~hs2p.mask.AnnotationLabels` map each class name to the mask values it
    owns; one binary mask (255 foreground / 0 background) is produced for every label, the
    union of its values. Every value the raster holds must be declared (the read rejects
    undeclared IDs), so a value reserved for unannotated pixels is declared like any other
    class — just given no coverage threshold, so it is never sampled. Which classes get
    sampled is decided downstream by the sampling spec (``min_coverage`` thresholds).

    ``requested_mask_backend`` is caller-owned provenance; the mask only knows the concrete
    backend it opened, which is recorded as the request when none is supplied.
    """
    normalized_downsamples = _normalize_level_downsamples(slide.level_downsamples)
    seg_level = select_level_for_downsample(
        float(seg_downsample),
        [(float(ds), float(ds)) for ds in normalized_downsamples],
    )
    seg_spacing_um = float(slide.spacing) * float(normalized_downsamples[seg_level])
    labels, mask_level, mask_spacing_um = _read_annotation_label_mask(
        mask=mask, slide=slide, seg_level=seg_level
    )
    masks = {
        name: np.where(np.isin(labels, values), 255, 0).astype(np.uint8)
        for name, values in mask.labels.pixel_mapping.items()
    }
    return ResolvedAnnotationMasks(
        masks=masks,
        tissue_method=tissue_method,
        requested_seg_downsample=int(seg_downsample),
        seg_downsample=max(1, int(round(seg_spacing_um / float(slide.spacing)))),
        seg_level=seg_level,
        seg_spacing_um=seg_spacing_um,
        pixel_mapping=_configured_pixel_mapping(mask.labels),
        mask_path=mask.path,
        mask_level=mask_level,
        mask_spacing_um=mask_spacing_um,
        mask_backend=mask.backend,
        requested_mask_backend=(
            requested_mask_backend
            if requested_mask_backend is not None
            else mask.backend
        ),
    )


__all__ = [
    "load_annotation_label_mask",
    "load_precomputed_tissue_mask",
    "prepare_sam2_thumbnail",
    "resolve_annotation_masks",
    "resolve_tissue_mask",
]
