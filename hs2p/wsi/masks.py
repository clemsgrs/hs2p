from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hs2p.wsi.geometry import select_level, select_level_for_downsample
from hs2p.wsi.reader import SlideReader


def _mask_spacing(mask_reader) -> float:
    spacing = getattr(mask_reader, "spacing", None)
    if spacing is not None:
        return float(spacing)
    return float(mask_reader.spacings[0])


def _mask_level_downsamples(mask_reader) -> list[tuple[float, float]]:
    if hasattr(mask_reader, "level_downsamples"):
        return list(mask_reader.level_downsamples)
    if hasattr(mask_reader, "downsamplings"):
        return [(float(ds), float(ds)) for ds in mask_reader.downsamplings]
    raise AttributeError("mask_reader must expose level_downsamples or downsamplings")


def _read_mask_level(mask_reader, level: int) -> np.ndarray:
    if hasattr(mask_reader, "read_level"):
        return mask_reader.read_level(level)
    return mask_reader.get_slide(mask_reader.spacings[level])


def normalize_tissue_mask(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    return (mask_arr > 0).astype(np.uint8)


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


def read_aligned_mask(
    *,
    mask_reader: SlideReader,
    slide_spacing: float,
    slide_dimensions: tuple[int, int],
) -> np.ndarray:
    mask_downsample = slide_spacing / _mask_spacing(mask_reader)
    mask_level = select_level_for_downsample(
        requested_downsample=mask_downsample,
        level_downsamples=_mask_level_downsamples(mask_reader),
    )
    mask_spacing = mask_reader.spacings[mask_level]
    scale = slide_spacing / mask_spacing
    while scale < 1 and mask_level > 0:
        mask_level -= 1
        mask_spacing = mask_reader.spacings[mask_level]
        scale = slide_spacing / mask_spacing

    mask_arr = _read_mask_level(mask_reader, mask_level)
    target_width, target_height = slide_dimensions
    return cv2.resize(
        mask_arr.astype(np.uint8),
        (int(target_width), int(target_height)),
        interpolation=cv2.INTER_NEAREST,
    )


def load_annotation_masks(
    *,
    mask_reader: SlideReader,
    mask_path: Path,
    segment_params,
    sampling_spec,
    seg_level: int,
    seg_spacing: float,
) -> dict[str, np.ndarray]:
    mask_downsample = seg_spacing / _mask_spacing(mask_reader)
    mask_level = select_level_for_downsample(
        requested_downsample=mask_downsample,
        level_downsamples=_mask_level_downsamples(mask_reader),
    )
    mask_spacing = mask_reader.spacings[mask_level]
    scale = seg_spacing / mask_spacing
    while scale < 1 and mask_level > 0:
        mask_level -= 1
        mask_spacing = mask_reader.spacings[mask_level]
        scale = seg_spacing / mask_spacing

    mask = _read_mask_level(mask_reader, mask_level)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    known_values = set(sampling_spec.pixel_mapping.values())
    if not set(np.unique(mask).tolist()).issubset(known_values):
        with Image.open(mask_path) as mask_img:
            mask_img.seek(mask_level)
            mask = np.array(mask_img)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

    height, width = mask.shape
    mask = cv2.resize(
        mask.astype(np.uint8),
        (int(round(width / scale, 0)), int(round(height / scale, 0))),
        interpolation=cv2.INTER_NEAREST,
    )

    background = sampling_spec.pixel_mapping["background"]
    annotation_mask = {"tissue": (mask != background).astype("uint8") * segment_params.sthresh_up}
    for annotation, val in sampling_spec.pixel_mapping.items():
        if annotation == "background":
            continue
        annotation_mask[annotation] = (mask == val).astype("uint8") * segment_params.sthresh_up
    return annotation_mask
