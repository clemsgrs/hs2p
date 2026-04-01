
import cv2
import numpy as np


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
