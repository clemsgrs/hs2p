from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hs2p.configs import SegmentationConfig

DEFAULT_SAM2_MODEL_REPO = "AtlasAnalyticsLab/AtlasPatch"
DEFAULT_SAM2_MODEL_FILENAME = "model.pth"


def segment_tissue_image(
    image: np.ndarray,
    *,
    config: SegmentationConfig,
) -> np.ndarray:
    """Return a uint8 binary tissue mask with the same height/width as ``image``."""
    image = _normalize_rgb_image(image)
    method = str(config.method).lower()
    if method == "hsv":
        mask = _segment_hsv(
            image,
            sthresh_up=int(config.sthresh_up),
        )
    elif method == "otsu":
        mask = _segment_otsu(
            image,
            mthresh=int(config.mthresh),
            sthresh_up=int(config.sthresh_up),
        )
    elif method == "threshold":
        mask = _segment_threshold(
            image,
            sthresh=int(config.sthresh),
            sthresh_up=int(config.sthresh_up),
            mthresh=int(config.mthresh),
        )
    elif method == "sam2":
        mask = _segment_sam2(
            image,
            checkpoint_path=config.sam2_checkpoint_path,
            config_path=config.sam2_config_path,
            device=str(config.sam2_device),
            input_size=int(config.sam2_input_size),
            mask_threshold=float(config.sam2_mask_threshold),
            sthresh_up=int(config.sthresh_up),
        )
    else:
        raise ValueError(
            f"Unknown tissue segmentation method '{config.method}'. "
            "Available: hsv, otsu, threshold, sam2"
        )

    if int(config.close) > 0:
        kernel = np.ones((int(config.close), int(config.close)), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(np.uint8, copy=False)


def _normalize_rgb_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected an RGB image with shape (H, W, C), got {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[2] != 3:
        raise ValueError(f"Expected 3 or 4 channels, got {arr.shape[2]}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _segment_hsv(
    image: np.ndarray,
    *,
    sthresh_up: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array((90, 8, 103)),
        np.array((180, 255, 255)),
    )
    if sthresh_up != 255:
        mask = np.where(mask > 0, int(sthresh_up), 0).astype(np.uint8)
    return mask


def _segment_otsu(
    image: np.ndarray,
    *,
    mthresh: int,
    sthresh_up: int,
) -> np.ndarray:
    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]
    blurred = cv2.medianBlur(saturation, int(mthresh))
    _, mask = cv2.threshold(
        blurred,
        0,
        int(sthresh_up),
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return mask


def _segment_threshold(
    image: np.ndarray,
    *,
    sthresh: int,
    sthresh_up: int,
    mthresh: int,
) -> np.ndarray:
    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]
    blurred = cv2.medianBlur(saturation, int(mthresh))
    _, mask = cv2.threshold(
        blurred,
        int(sthresh),
        int(sthresh_up),
        cv2.THRESH_BINARY,
    )
    return mask


def _segment_sam2(
    image: np.ndarray,
    *,
    checkpoint_path: Path | None,
    config_path: Path | None,
    device: str,
    input_size: int,
    mask_threshold: float,
    sthresh_up: int,
) -> np.ndarray:
    predictor = _build_sam2_predictor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
        input_size=input_size,
        mask_threshold=mask_threshold,
    )
    mask = predictor.predict_mask(image)
    return np.where(mask > 0, int(sthresh_up), 0).astype(np.uint8)


def _build_sam2_predictor(
    *,
    checkpoint_path: Path | None,
    config_path: Path | None,
    device: str,
    input_size: int,
    mask_threshold: float,
) -> "_Sam2Predictor":
    return _Sam2Predictor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
        input_size=input_size,
        mask_threshold=mask_threshold,
    )


class _Sam2Predictor:
    def __init__(
        self,
        *,
        checkpoint_path: Path | None,
        config_path: Path | None,
        device: str,
        input_size: int,
        mask_threshold: float,
    ) -> None:
        if config_path is None:
            raise ValueError("sam2_config_path is required when method='sam2'")
        if input_size <= 0:
            raise ValueError(f"sam2_input_size must be > 0, got {input_size}")
        self.device = _validate_sam2_device(device)
        self.input_size = int(input_size)
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self._predictor = self._load_predictor(
            checkpoint_path=self.checkpoint_path,
            config_path=Path(config_path),
            device=self.device,
            mask_threshold=float(mask_threshold),
        )

    def _resolve_checkpoint_path(self, checkpoint_path: Path | None) -> Path:
        if checkpoint_path is not None:
            resolved = Path(checkpoint_path)
            if not resolved.exists():
                raise FileNotFoundError(f"SAM2 checkpoint not found: {resolved}")
            return resolved

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Automatic SAM2 checkpoint download requires huggingface-hub. "
                "Install hs2p with the 'sam2' extra or provide sam2_checkpoint_path."
            ) from exc

        try:
            downloaded = hf_hub_download(
                repo_id=DEFAULT_SAM2_MODEL_REPO,
                filename=DEFAULT_SAM2_MODEL_FILENAME,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to download the default SAM2 checkpoint from "
                f"{DEFAULT_SAM2_MODEL_REPO}: {exc}"
            ) from exc
        return Path(downloaded)

    def _load_predictor(
        self,
        *,
        checkpoint_path: Path,
        config_path: Path,
        device: str,
        mask_threshold: float,
    ) -> Any:
        try:
            import torch
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "SAM2 tissue segmentation requires optional dependencies. "
                "Install hs2p with the 'sam2' extra and the SAM2 package."
            ) from exc

        if not config_path.exists():
            raise FileNotFoundError(f"SAM2 config not found: {config_path}")

        conf = OmegaConf.load(str(config_path))
        model_cfg: Any = conf.get("model", conf)
        model = instantiate(model_cfg)
        predictor = SAM2ImagePredictor(model, mask_threshold=float(mask_threshold))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        predictor.model.load_state_dict(checkpoint["model"], strict=True)
        predictor.model.to(device).eval()
        return predictor

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        resized, original_shape = _resize_image_for_sam2(image, input_size=self.input_size)
        self._predictor.set_image(resized)
        height, width = resized.shape[:2]
        bbox = np.array([0, 0, width, height], dtype=np.float32)
        masks, _, _ = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox,
            multimask_output=False,
            return_logits=False,
        )
        mask = np.asarray(masks[0], dtype=np.float32)
        if mask.shape[:2] != original_shape:
            mask = _resize_mask_from_sam2(mask, target_shape=original_shape)
        return (mask > 0).astype(np.uint8)


def _resize_image_for_sam2(
    image: np.ndarray,
    *,
    input_size: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    original_shape = (int(image.shape[0]), int(image.shape[1]))
    if original_shape == (input_size, input_size):
        return image, original_shape
    resized = cv2.resize(
        image,
        (int(input_size), int(input_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized, original_shape


def _resize_mask_from_sam2(
    mask: np.ndarray,
    *,
    target_shape: tuple[int, int],
) -> np.ndarray:
    resized = cv2.resize(
        mask.astype(np.float32),
        (int(target_shape[1]), int(target_shape[0])),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized


def _validate_sam2_device(device: str) -> str:
    dev = str(device).strip().lower()
    if dev == "cpu":
        return dev
    if dev == "cuda" or dev.startswith("cuda:"):
        if dev == "cuda":
            return dev
        suffix = dev.split("cuda:", 1)[1]
        if suffix and suffix.isdigit():
            return dev
    allowed = ", ".join(["cpu", "cuda", "cuda:<index>"])
    raise ValueError(f"Invalid SAM2 device '{device}'. Expected one of {allowed}.")
