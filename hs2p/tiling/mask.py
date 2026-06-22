from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hs2p.configs import SegmentationConfig
from hs2p.segmentation import segment_tissue_image
from hs2p.tiling.contours import _normalize_level_downsamples
from hs2p.tiling.result import ResolvedAnnotationMasks, ResolvedTissueMask, Sam2Thumbnail
from hs2p.wsi.reader import open_slide, select_level, select_level_for_downsample

DEFAULT_SAM2_THUMBNAIL_SPACING_UM = 8.0
DEFAULT_SAM2_THUMBNAIL_TOLERANCE = 0.05


def _reduce_mask_channels(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        return np.asarray(mask[..., 0])
    return np.asarray(mask)


def _as_discrete_label_array(mask: np.ndarray) -> np.ndarray:
    """Return ``mask`` as the smallest cv2-resize-safe unsigned integer dtype that
    preserves every label value (``uint8`` or ``uint16``).

    Discrete masks must never be silently narrowed: downcasting a validated ``uint16``
    label raster to ``uint8`` would wrap any value above 255 modulo 256, making a class
    vanish or merge into the class that owns the wrapped value. Labels are non-negative
    class ids, so we reject negatives and values past the supported ``uint16`` range
    rather than corrupting them.
    """
    arr = np.asarray(mask)
    if arr.size == 0:
        return arr.astype(np.uint8, copy=False)
    min_value = int(arr.min())
    max_value = int(arr.max())
    if min_value < 0:
        raise ValueError(f"Mask contains negative label value {min_value}")
    if max_value <= np.iinfo(np.uint8).max:
        return arr.astype(np.uint8, copy=False)
    if max_value <= np.iinfo(np.uint16).max:
        return arr.astype(np.uint16, copy=False)
    raise ValueError(
        f"Mask label value {max_value} exceeds the supported uint16 range"
    )


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


def _is_label_subset(mask: np.ndarray, *, valid_values: set[int]) -> bool:
    values = {int(value) for value in np.unique(mask.astype(np.int64, copy=False)).tolist()}
    return values <= valid_values


def _read_discrete_mask_level(
    *,
    mask_path: str | Path,
    mask_slide,
    mask_level: int,
    is_discrete,
    label: str,
) -> np.ndarray:
    """Read a mask level, preferring the backend read but falling back to an exact TIFF read
    when the backend's (possibly interpolated) downsampled level is not label-discrete.

    ``is_discrete`` is the predicate that decides acceptability — a single-tissue-value check
    for tissue masks, or a class-value-subset check for multi-class annotation masks.
    """
    mask_size = mask_slide.level_dimensions[mask_level]
    backend_mask = _reduce_mask_channels(mask_slide.read_region((0, 0), mask_level, mask_size))
    if is_discrete(backend_mask):
        return _as_discrete_label_array(backend_mask)

    suffix = Path(mask_path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        exact_mask = _reduce_mask_channels(
            _read_exact_tiff_mask_level(mask_path, mask_level=mask_level)
        )
        if is_discrete(exact_mask):
            return _as_discrete_label_array(exact_mask)
    values = np.unique(backend_mask.astype(np.int64, copy=False))
    raise ValueError(
        f"{label} read produced non-discrete labels "
        f"at level {mask_level}: {values.tolist()[:16]}"
    )


def _read_mask_level(
    *,
    mask_path: str | Path,
    mask_slide,
    mask_level: int,
    tissue_value: int,
) -> np.ndarray:
    return _read_discrete_mask_level(
        mask_path=mask_path,
        mask_slide=mask_slide,
        mask_level=mask_level,
        is_discrete=lambda mask: _is_discrete_binary_mask(mask, tissue_value=tissue_value),
        label="Precomputed tissue mask",
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
            _as_discrete_label_array(raw_mask),
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
            requested_seg_downsample=int(seg_downsample),
            seg_downsample=max(1, int(round(seg_spacing_um / float(slide.spacing)))),
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


def _read_label_mask_at_seg(
    *,
    mask_path: str | Path,
    slide,
    seg_level: int,
    valid_values: set[int],
    backend: str,
) -> tuple[np.ndarray, int, float]:
    """Read the label raster at the mask level nearest the slide's ``seg_level`` spacing,
    nearest-resampled to the seg grid, using the requested ``backend``."""
    mask_slide = open_slide(mask_path, backend=backend)
    try:
        seg_size = slide.level_dimensions[seg_level]
        seg_spacing_um = float(slide.spacing) * float(
            _normalize_level_downsamples(slide.level_downsamples)[seg_level]
        )
        mask_level, mask_spacing_um = _select_mask_level(
            mask_slide=mask_slide,
            requested_spacing_um=seg_spacing_um,
        )
        raw_mask = _read_discrete_mask_level(
            mask_path=mask_path,
            mask_slide=mask_slide,
            mask_level=mask_level,
            is_discrete=lambda mask: _is_label_subset(mask, valid_values=valid_values),
            label="Annotation mask",
        )
    finally:
        mask_slide.close()

    if raw_mask.shape[:2] != (int(seg_size[1]), int(seg_size[0])):
        raw_mask = cv2.resize(
            _as_discrete_label_array(raw_mask),
            (int(seg_size[0]), int(seg_size[1])),
            interpolation=cv2.INTER_NEAREST,
        )
    return _as_discrete_label_array(raw_mask), mask_level, mask_spacing_um


def load_annotation_label_mask(
    *,
    mask_path: str | Path,
    slide,
    seg_level: int,
    valid_values: set[int],
) -> tuple[np.ndarray, int, float]:
    """Read a multi-class annotation label raster resampled to the slide's ``seg_level`` grid.

    Mirrors :func:`load_precomputed_tissue_mask` but preserves every class index (no
    binarization) and validates against the annotation pixel-value set rather than a single
    tissue value. Resampling is nearest-neighbor — any averaging would invent class indices.

    Robustness guard: some backends mis-decode intermediate levels of a compressed
    single-channel (minisblack) pyramid and return an all-background level (observed with
    cucim + LZW/deflate masks, where openslide decodes the same level correctly). We retry via
    openslide and prefer that result when it recovers labels — costless when the primary read
    already carries labels, and safe for genuinely-empty masks (openslide would agree).

    The degenerate read shows up two ways depending on the label vocabulary: if the background
    value is declared, the all-background level reads as a valid (empty) mask; if it is not
    (e.g. ``{tumor: 1, stroma: 2}`` with no ``0``), the discreteness guard *rejects* the
    all-zero level and raises. Both must trigger the openslide retry, so the primary read is
    guarded and its error only surfaces if openslide fails to recover.
    """
    backend_name = str(getattr(slide, "backend_name", "")).lower()
    primary_error: Exception | None = None
    raw_mask = mask_level = mask_spacing_um = None
    try:
        raw_mask, mask_level, mask_spacing_um = _read_label_mask_at_seg(
            mask_path=mask_path,
            slide=slide,
            seg_level=seg_level,
            valid_values=valid_values,
            backend=slide.backend_name,
        )
    except Exception as exc:
        primary_error = exc

    degenerate = primary_error is not None or not raw_mask.any()
    fallback = None
    if degenerate and backend_name != "openslide":
        try:
            fallback = _read_label_mask_at_seg(
                mask_path=mask_path,
                slide=slide,
                seg_level=seg_level,
                valid_values=valid_values,
                backend="openslide",
            )
        except Exception:
            fallback = None
    if fallback is not None:
        fb_mask, fb_level, fb_spacing = fallback
        # When the primary read *raised*, any validated fallback is acceptable — including a
        # legitimately all-zero mask (a genuinely empty raster, or a label mapped to 0 now
        # that no value is reserved). The ``.any()`` gate applies only to the all-background
        # *successful*-primary recovery heuristic, where we prefer openslide solely when it
        # actually recovers labels rather than swapping one empty read for another.
        if primary_error is not None or fb_mask.any():
            return fb_mask, fb_level, fb_spacing

    if primary_error is not None:
        raise primary_error
    return raw_mask, mask_level, mask_spacing_um


def resolve_annotation_masks(
    *,
    slide,
    mask_path: str | Path,
    pixel_mapping: dict[str, int],
    seg_downsample: int = 64,
    tissue_method: str = "precomputed_mask",
) -> ResolvedAnnotationMasks:
    """Read an annotation mask into one binary mask per declared label at ``seg_downsample``.

    The multi-label producer that ``build_per_annotation_tiling_results`` consumes but that
    was missing from the codebase — the annotation counterpart of
    :func:`resolve_tissue_mask`'s precomputed path. ``pixel_mapping`` maps class name to the
    integer pixel value in the mask; one binary mask (255 foreground / 0 background) is
    produced for **every** entry — no label name is special. ``pixel_mapping`` is the user's
    own vocabulary; *which* of these classes get sampled is decided downstream by the sampling
    spec (``min_coverage`` thresholds), not here.

    There is no reserved ``background`` label: every pixel value present in the raster must be
    declared (the read-time discreteness guard rejects undeclared values), so if the raster
    reserves a value for unannotated pixels, declare it like any other class — just give it no
    coverage threshold and it will never be sampled.
    """
    if mask_path is None:
        raise ValueError("resolve_annotation_masks requires a mask_path")

    normalized_downsamples = _normalize_level_downsamples(slide.level_downsamples)
    seg_level = select_level_for_downsample(
        float(seg_downsample),
        [(float(ds), float(ds)) for ds in normalized_downsamples],
    )
    seg_spacing_um = float(slide.spacing) * float(normalized_downsamples[seg_level])
    valid_values = {int(value) for value in pixel_mapping.values()}
    raw_mask, mask_level, mask_spacing_um = load_annotation_label_mask(
        mask_path=mask_path,
        slide=slide,
        seg_level=seg_level,
        valid_values=valid_values,
    )
    masks = {
        name: np.where(raw_mask == int(value), np.uint8(255), np.uint8(0)).astype(np.uint8)
        for name, value in pixel_mapping.items()
    }
    return ResolvedAnnotationMasks(
        masks=masks,
        tissue_method=tissue_method,
        requested_seg_downsample=int(seg_downsample),
        seg_downsample=max(1, int(round(seg_spacing_um / float(slide.spacing)))),
        seg_level=seg_level,
        seg_spacing_um=seg_spacing_um,
        pixel_mapping=dict(pixel_mapping),
        mask_path=mask_path,
        mask_level=mask_level,
        mask_spacing_um=mask_spacing_um,
    )


__all__ = [
    "load_annotation_label_mask",
    "load_precomputed_tissue_mask",
    "prepare_sam2_thumbnail",
    "resolve_annotation_masks",
    "resolve_tissue_mask",
]
