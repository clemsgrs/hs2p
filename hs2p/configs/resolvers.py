from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hs2p.wsi.types import CoordinateSelectionStrategy, SamplingSpec

from .loader import default_config
from .models import FilterConfig, PreviewConfig, SegmentationConfig, TilingConfig


def resolve_tiling_config(cfg: Any) -> TilingConfig:
    return TilingConfig(
        target_spacing_um=cfg.tiling.params.target_spacing_um,
        target_tile_size_px=cfg.tiling.params.target_tile_size_px,
        tolerance=cfg.tiling.params.tolerance,
        overlap=cfg.tiling.params.overlap,
        tissue_threshold=cfg.tiling.params.tissue_threshold,
        use_padding=cfg.tiling.params.use_padding,
        backend=(getattr(cfg.tiling, "backend", None) or "auto"),
    )


def resolve_segmentation_config(cfg: Any) -> SegmentationConfig:
    return SegmentationConfig(**dict(cfg.tiling.seg_params))


def resolve_filter_config(cfg: Any) -> FilterConfig:
    return FilterConfig(**dict(cfg.tiling.filter_params))


def resolve_preview_config(cfg: Any) -> PreviewConfig:
    preview_cfg = cfg.tiling.preview
    default_preview = default_config.tiling.preview
    return PreviewConfig(
        save_mask_preview=bool(cfg.save_previews),
        save_tiling_preview=bool(cfg.save_previews),
        downsample=int(preview_cfg.downsample),
        mask_overlay_color=tuple(
            int(v)
            for v in getattr(
                preview_cfg,
                "mask_overlay_color",
                default_preview.mask_overlay_color,
            )
        ),
        mask_overlay_alpha=float(
            getattr(
                preview_cfg,
                "mask_overlay_alpha",
                default_preview.mask_overlay_alpha,
            )
        ),
    )


def resolve_read_coordinates_from(cfg: Any) -> Path | None:
    value = cfg.tiling.read_coordinates_from
    if not value:
        return None
    return Path(value)


def resolve_sampling_strategy(cfg: Any) -> str:
    if cfg.tiling.sampling_params.independent_sampling:
        return CoordinateSelectionStrategy.INDEPENDENT_SAMPLING
    return CoordinateSelectionStrategy.JOINT_SAMPLING


def build_default_sampling_spec(tiling: TilingConfig) -> SamplingSpec:
    return SamplingSpec(
        pixel_mapping={"background": 0, "tissue": 1},
        color_mapping={"background": None, "tissue": None},
        tissue_percentage={
            "background": None,
            "tissue": tiling.tissue_threshold,
        },
        active_annotations=("tissue",),
    )


def _merge_sampling_mapping(
    entries: Any,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    if entries is None:
        return None
    if isinstance(entries, dict):
        return {str(key): value for key, value in entries.items()}
    try:
        iterator = list(entries)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a mapping or a list of mappings") from exc
    merged: dict[str, Any] = {}
    for entry in iterator:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{field_name} must be a mapping or a list of single-entry mappings"
            )
        for key, value in entry.items():
            merged[str(key)] = value
    return merged


def validate_color_mapping(
    *,
    pixel_mapping: dict[str, int],
    color_mapping: dict[str, Sequence[int] | None],
) -> None:
    missing_annotations = sorted(set(pixel_mapping.keys()) - set(color_mapping.keys()))
    if missing_annotations:
        raise ValueError(
            "color_mapping is missing annotation keys required by pixel_mapping: "
            + ", ".join(missing_annotations)
        )
    unexpected_annotations = sorted(set(color_mapping.keys()) - set(pixel_mapping.keys()))
    if unexpected_annotations:
        raise ValueError(
            "color_mapping has unexpected annotation keys: "
            + ", ".join(unexpected_annotations)
        )

    for annotation, color in color_mapping.items():
        if color is None:
            continue
        if isinstance(color, (str, bytes)):
            raise ValueError(
                f"color_mapping['{annotation}'] must be None or a length-3 RGB sequence"
            )
        if not isinstance(color, Sequence) or len(color) != 3:
            raise ValueError(
                f"color_mapping['{annotation}'] must be None or a length-3 RGB sequence"
            )
        if any(
            (not isinstance(c, (int, np.integer)) or c < 0 or c > 255) for c in color
        ):
            raise ValueError(
                f"color_mapping['{annotation}'] must contain integers in [0, 255]"
            )


def resolve_sampling_spec(
    cfg: Any,
    *,
    tiling: TilingConfig,
) -> SamplingSpec:
    sampling_config = getattr(cfg.tiling, "sampling_params", None)
    if sampling_config is None:
        return build_default_sampling_spec(tiling)

    pixel_mapping = _merge_sampling_mapping(
        getattr(sampling_config, "pixel_mapping", None),
        field_name="pixel_mapping",
    )
    tissue_percentage = _merge_sampling_mapping(
        getattr(sampling_config, "tissue_percentage", None),
        field_name="tissue_percentage",
    )
    if pixel_mapping is None:
        raise ValueError("sampling pixel_mapping is required when sampling config is provided")
    if tissue_percentage is None:
        raise ValueError(
            "sampling tissue_percentage is required when sampling config is provided"
        )
    if "background" not in pixel_mapping:
        raise ValueError("sampling pixel_mapping must include a 'background' label")
    missing_threshold_labels = sorted(
        set(tissue_percentage.keys()) - set(pixel_mapping.keys())
    )
    if missing_threshold_labels:
        raise ValueError(
            "sampling tissue_percentage references unknown labels: "
            + ", ".join(missing_threshold_labels)
        )
    color_mapping = _merge_sampling_mapping(
        getattr(sampling_config, "color_mapping", None),
        field_name="color_mapping",
    )
    if color_mapping is not None:
        validate_color_mapping(
            pixel_mapping=pixel_mapping,
            color_mapping=color_mapping,
        )

    return SamplingSpec(
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        tissue_percentage=tissue_percentage,
        active_annotations=tuple(
            annotation
            for annotation, pct in tissue_percentage.items()
            if annotation in pixel_mapping and pct is not None and annotation != "background"
        ),
    )
