from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hs2p.configs import SegmentationConfig
from hs2p.segmentation import segment_tissue_image
from hs2p.tiling.contours import _normalize_level_downsamples
from hs2p.tiling.result import ResolvedTissueMask, Sam2Thumbnail
from hs2p.wsi.reader import open_slide, select_level, select_level_for_downsample

DEFAULT_SAM2_THUMBNAIL_SPACING_UM = 8.0
DEFAULT_SAM2_THUMBNAIL_TOLERANCE = 0.05


def _reduce_mask_channels(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        return np.asarray(mask[..., 0])
    return np.asarray(mask)


def _is_discrete_binary_mask(mask: np.ndarray, *, tissue_value: int) -> bool:
    values = np.unique(mask.astype(np.int64, copy=False))
    non_tissue = [int(value) for value in values.tolist() if int(value) != tissue_value]
    return len(values) <= 2 and len(non_tissue) <= 1


def _read_exact_tiff_mask_level(mask_path: str | Path, *, mask_level: int) -> np.ndarray:
    path = Path(mask_path)
    with Image.open(path) as image:
        n_frames = int(getattr(image, "n_frames", 1))
        if mask_level >= n_frames:
            raise ValueError(
                f"Mask level {mask_level} is unavailable in TIFF mask {path} (n_frames={n_frames})"
            )
        image.seek(mask_level)
        return np.asarray(image)


def _read_mask_level(
    *,
    mask_path: str | Path,
    mask_slide,
    mask_level: int,
    tissue_value: int,
) -> np.ndarray:
    mask_size = mask_slide.level_dimensions[mask_level]
    backend_mask = mask_slide.read_region((0, 0), mask_level, mask_size)
    backend_mask = _reduce_mask_channels(backend_mask)
    if _is_discrete_binary_mask(backend_mask, tissue_value=tissue_value):
        return backend_mask.astype(np.uint8, copy=False)

    suffix = Path(mask_path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        exact_mask = _reduce_mask_channels(
            _read_exact_tiff_mask_level(mask_path, mask_level=mask_level)
        )
        if _is_discrete_binary_mask(exact_mask, tissue_value=tissue_value):
            return exact_mask.astype(np.uint8, copy=False)
    values = np.unique(backend_mask.astype(np.int64, copy=False))
    raise ValueError(
        "Precomputed tissue mask read produced non-discrete labels "
        f"at level {mask_level}: {values.tolist()[:16]}"
    )


def _select_mask_level(
    *,
    mask_slide,
    requested_spacing_um: float,
) -> tuple[int, float]:
    read_spacings = [
        float(mask_slide.spacing) * float(downsample[0] if isinstance(downsample, tuple) else downsample)
        for downsample in mask_slide.level_downsamples
    ]
    level = int(np.argmin([abs(spacing_um - requested_spacing_um) for spacing_um in read_spacings]))
    spacing_um = read_spacings[level]
    while level > 0 and spacing_um > requested_spacing_um:
        level -= 1
        spacing_um = read_spacings[level]
    return level, spacing_um


def load_precomputed_tissue_mask(
    *,
    mask_path: str | Path,
    slide,
    seg_level: int,
    tissue_value: int,
) -> tuple[np.ndarray, int, float]:
    mask_slide = open_slide(mask_path, backend=slide.backend_name)
    try:
        seg_size = slide.level_dimensions[seg_level]
        seg_spacing_um = float(slide.spacing) * float(
            _normalize_level_downsamples(slide.level_downsamples)[seg_level]
        )
        mask_level, mask_spacing_um = _select_mask_level(
            mask_slide=mask_slide,
            requested_spacing_um=seg_spacing_um,
        )
        raw_mask = _read_mask_level(
            mask_path=mask_path,
            mask_slide=mask_slide,
            mask_level=mask_level,
            tissue_value=tissue_value,
        )
    finally:
        mask_slide.close()

    if raw_mask.shape[:2] != (int(seg_size[1]), int(seg_size[0])):
        raw_mask = cv2.resize(
            raw_mask.astype(np.uint8, copy=False),
            (int(seg_size[0]), int(seg_size[1])),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = np.where(raw_mask == tissue_value, 255, 0).astype(np.uint8)
    return mask, mask_level, mask_spacing_um


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
    tissue_method: str | None = None,
    tissue_mask_path: str | Path | None,
    tissue_mask_tissue_value: int = 1,
    sthresh: int = 8,
    sthresh_up: int = 255,
    mthresh: int = 7,
    close: int = 4,
    seg_downsample: int = 64,
    sam2_checkpoint_path: str | Path | None = None,
    sam2_config_path: str | Path | None = None,
    sam2_device: str = "cpu",
) -> ResolvedTissueMask:
    if tissue_mask_path is not None:
        normalized_downsamples = _normalize_level_downsamples(slide.level_downsamples)
        seg_level = select_level_for_downsample(
            float(seg_downsample),
            [(float(ds), float(ds)) for ds in normalized_downsamples],
        )
        seg_spacing_um = float(slide.spacing) * float(normalized_downsamples[seg_level])
        mask, mask_level, mask_spacing_um = load_precomputed_tissue_mask(
            mask_path=tissue_mask_path,
            slide=slide,
            seg_level=seg_level,
            tissue_value=int(tissue_mask_tissue_value),
        )
        return ResolvedTissueMask(
            tissue_mask=mask,
            tissue_method="precomputed_mask",
            seg_downsample=int(seg_downsample),
            seg_level=seg_level,
            seg_spacing_um=seg_spacing_um,
            mask_path=tissue_mask_path,
            tissue_mask_tissue_value=int(tissue_mask_tissue_value),
            mask_level=mask_level,
            mask_spacing_um=mask_spacing_um,
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
        effective_downsample = max(1, int(round(seg_spacing_um / float(slide.spacing))))
    else:
        normalized_downsamples = _normalize_level_downsamples(slide.level_downsamples)
        seg_level = select_level_for_downsample(
            float(seg_downsample),
            [(float(ds), float(ds)) for ds in normalized_downsamples],
        )
        seg_spacing_um = float(slide.spacing) * float(normalized_downsamples[seg_level])
        seg_size = slide.level_dimensions[seg_level]
        seg_image = np.asarray(slide.read_region((0, 0), seg_level, seg_size))
        effective_downsample = int(seg_downsample)

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
        seg_downsample=effective_downsample,
        seg_level=seg_level,
        seg_spacing_um=seg_spacing_um,
    )


__all__ = [
    "load_precomputed_tissue_mask",
    "prepare_sam2_thumbnail",
    "resolve_tissue_mask",
]
