import numpy as np

from hs2p.wsi.types import PixelMapping, pixel_values


def normalize_tissue_mask(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    return (mask_arr > 0).astype(np.uint8)


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
    pixel_mapping: PixelMapping,
) -> np.ndarray:
    # A label merging several raw values is painted with its first (representative) value.
    mask = np.full_like(
        normalize_tissue_mask(annotation_mask["tissue"]),
        fill_value=pixel_values(pixel_mapping.get("background", 0))[0],
        dtype=np.uint8,
    )
    for annotation, entry in pixel_mapping.items():
        if annotation == "background":
            continue
        if annotation not in annotation_mask:
            continue
        mask[normalize_tissue_mask(annotation_mask[annotation]) > 0] = pixel_values(entry)[0]
    return mask
