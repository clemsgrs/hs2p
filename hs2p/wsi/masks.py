
import cv2
import numpy as np


def read_label_at_spacing(
    wsi,
    requested_spacing_um: float,
    *,
    tolerance: float,
) -> np.ndarray:
    """Read a multi-class label raster at ``requested_spacing_um``, preserving class ids.

    Resamples with **nearest-neighbor** (any averaging interpolation would invent
    class indices) via :meth:`hs2p.wsi.wsi.WSI.read_full_at_spacing`, then recovers a
    single-channel integer raster. Slide backends return RGB (a single-channel label
    is replicated across channels); this collapses it back to one channel, asserting
    the channels agree so a genuine colour image fails loud rather than silently
    keeping only its red channel.

    Args:
        wsi: An :class:`hs2p.wsi.wsi.WSI` (or any object exposing
            ``read_full_at_spacing``).
        requested_spacing_um: Target spacing (µm/px).
        tolerance: Relative spacing tolerance for the level match.

    Returns:
        A 2-D integer ``np.ndarray`` of class indices at the requested spacing.
    """
    arr = wsi.read_full_at_spacing(
        requested_spacing_um, tolerance=tolerance, interpolation="nearest"
    )
    return _collapse_label_raster(arr)


def read_label_region_at_spacing(
    wsi,
    location: tuple[int, int],
    requested_spacing_um: float,
    size: tuple[int, int],
    *,
    tolerance: float,
) -> np.ndarray:
    """Read a multi-class label *region* at ``requested_spacing_um``, preserving class ids.

    The region counterpart of :func:`read_label_at_spacing` (for annotation-sampled ROIs):
    nearest-neighbor resample via :meth:`hs2p.wsi.wsi.WSI.read_region_at_spacing`, then
    collapse the channel-replicated raster back to a single-channel integer mask.

    Args:
        location: ``(x, y)`` top-left in level-0 pixel space.
        size: Output ``(width, height)`` in pixels at ``requested_spacing_um``.
    """
    arr = wsi.read_region_at_spacing(
        location,
        requested_spacing_um,
        size,
        tolerance=tolerance,
        interpolation="nearest",
    )
    return _collapse_label_raster(arr)


def _collapse_label_raster(arr: np.ndarray) -> np.ndarray:
    """Recover a 2-D integer class-index raster from a (possibly RGB-replicated) read."""
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels >= 3 and not (
            np.array_equal(arr[..., 0], arr[..., 1])
            and np.array_equal(arr[..., 1], arr[..., 2])
        ):
            raise ValueError(
                "label raster has non-identical RGB channels; expected a "
                "single-channel class-index mask (channel-replicated by the backend), "
                "not a colour image."
            )
        arr = arr[..., 0]
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"label raster must have an integer dtype (class indices), got {arr.dtype}."
        )
    return arr


def normalize_tissue_mask(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    return (mask_arr > 0).astype(np.uint8)


def mask_level_downsamples(mask_obj) -> list[tuple[float, float]]:
    if not hasattr(mask_obj, "level_downsamples"):
        raise AttributeError("Mask object must expose level_downsamples")
    return list(mask_obj.level_downsamples)


def read_aligned_mask(
    *,
    mask_obj,
    slide_spacing: float,
    slide_dimensions: tuple[int, int],
) -> np.ndarray:
    mask_spacings = list(mask_obj.spacings)
    level_downsamples = mask_level_downsamples(mask_obj)
    mask_spacing_at_level_0 = mask_spacings[0]
    mask_downsample = slide_spacing / mask_spacing_at_level_0
    mask_level = int(np.argmin([abs(ds[0] - mask_downsample) for ds in level_downsamples]))
    mask_spacing = mask_spacings[mask_level]
    scale = slide_spacing / mask_spacing
    while scale < 1 and mask_level > 0:
        mask_level -= 1
        mask_spacing = mask_spacings[mask_level]
        scale = slide_spacing / mask_spacing

    mask_arr = mask_obj.read_level(mask_level)
    target_width, target_height = slide_dimensions
    return cv2.resize(
        np.asarray(mask_arr, dtype=np.uint8),
        (int(target_width), int(target_height)),
        interpolation=cv2.INTER_NEAREST,
    )


def pad_array_to_shape(
    arr: np.ndarray, *, target_width: int, target_height: int
) -> np.ndarray:
    if arr.shape[1] == target_width and arr.shape[0] == target_height:
        return arr
    if arr.shape[1] > target_width or arr.shape[0] > target_height:
        raise ValueError(
            "Cannot pad an array to a smaller shape; expected the target canvas to be at least as large as the source"
        )
    if arr.ndim == 2:
        padded = np.zeros((target_height, target_width), dtype=arr.dtype)
        padded[: arr.shape[0], : arr.shape[1]] = arr
        return padded
    padded = np.zeros((target_height, target_width, arr.shape[2]), dtype=arr.dtype)
    padded[: arr.shape[0], : arr.shape[1], :] = arr
    return padded


def extract_padded_crop(
    arr: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    if arr.ndim == 2:
        crop = np.zeros((height, width), dtype=arr.dtype)
        src = arr[y : y + height, x : x + width]
        crop[: src.shape[0], : src.shape[1]] = src
        return crop
    crop = np.zeros((height, width, arr.shape[2]), dtype=arr.dtype)
    src = arr[y : y + height, x : x + width, :]
    crop[: src.shape[0], : src.shape[1], :] = src
    return crop


def compose_overlay_mask_from_annotations(
    *,
    annotation_mask: dict[str, np.ndarray],
    pixel_mapping: dict[str, int],
) -> np.ndarray:
    mask = np.full_like(
        normalize_tissue_mask(annotation_mask["tissue"]),
        fill_value=pixel_mapping.get("background", 0),
        dtype=np.uint8,
    )
    for annotation, label_value in pixel_mapping.items():
        if annotation == "background":
            continue
        if annotation not in annotation_mask:
            continue
        mask[normalize_tissue_mask(annotation_mask[annotation]) > 0] = label_value
    return mask
