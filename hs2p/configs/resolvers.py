from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hs2p.wsi.types import CoordinateSelectionStrategy, SamplingSpec

from .models import FilterConfig, PreviewConfig, SegmentationConfig, TilingConfig


def resolve_tiling_config(cfg: Any) -> TilingConfig:
    min_coverage = _merge_sampling_mapping(
        cfg.tiling.masks.min_coverage,
        field_name="min_coverage",
    )
    return TilingConfig(
        requested_spacing_um=cfg.tiling.params.requested_spacing_um,
        requested_tile_size_px=cfg.tiling.params.requested_tile_size_px,
        tolerance=cfg.tiling.params.tolerance,
        overlap=cfg.tiling.params.overlap,
        tissue_threshold=float(min_coverage["tissue"]),
        independent_sampling=bool(cfg.tiling.independent_sampling),
        backend=cfg.tiling.backend,
    )


def resolve_segmentation_config(cfg: Any) -> SegmentationConfig:
    return SegmentationConfig(**dict(cfg.tiling.seg_params))


def resolve_filter_config(cfg: Any) -> FilterConfig:
    return FilterConfig(**dict(cfg.tiling.filter_params))


def resolve_preview_config(cfg: Any) -> PreviewConfig:
    preview_cfg = cfg.tiling.preview
    return PreviewConfig(
        save_mask_preview=bool(preview_cfg.save),
        save_tiling_preview=bool(preview_cfg.save),
        downsample=int(preview_cfg.downsample),
        mask_overlay_color=tuple(int(v) for v in preview_cfg.mask_overlay_color),
        mask_overlay_alpha=float(preview_cfg.mask_overlay_alpha),
    )


def resolve_read_coordinates_from(cfg: Any) -> Path | None:
    value = cfg.tiling.read_coordinates_from
    if not value:
        return None
    return Path(value)


def resolve_sampling_strategy(cfg: Any) -> str:
    """Derive selection strategy from tiling.independent_sampling."""
    independent = bool(getattr(cfg.tiling, "independent_sampling", False))
    if independent:
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
    # Try new masks config first
    masks_cfg = getattr(cfg.tiling, "masks", None)
    if masks_cfg is not None:
        return _resolve_sampling_spec_from_masks(masks_cfg, tiling=tiling)

    # Fall back to legacy sampling_params key
    sampling_config = getattr(cfg.tiling, "sampling_params", None)
    if sampling_config is None:
        return build_default_sampling_spec(tiling)
    return _resolve_sampling_spec_from_sampling_params(sampling_config, tiling=tiling)


def _resolve_sampling_spec_from_masks(masks_cfg: Any, *, tiling: TilingConfig) -> SamplingSpec:
    pixel_mapping = _merge_sampling_mapping(
        getattr(masks_cfg, "pixel_mapping", None),
        field_name="pixel_mapping",
    )
    min_coverage = _merge_sampling_mapping(
        getattr(masks_cfg, "min_coverage", None),
        field_name="min_coverage",
    )
    if pixel_mapping is None:
        raise ValueError("masks.pixel_mapping is required")
    if min_coverage is None:
        raise ValueError("masks.min_coverage is required")
    if "background" not in pixel_mapping:
        raise ValueError("masks.pixel_mapping must include a 'background' label")
    missing_coverage_labels = sorted(set(min_coverage.keys()) - set(pixel_mapping.keys()))
    if missing_coverage_labels:
        raise ValueError(
            "masks.min_coverage references unknown labels: "
            + ", ".join(missing_coverage_labels)
        )
    colors = _merge_sampling_mapping(
        getattr(masks_cfg, "colors", None),
        field_name="colors",
    )
    if colors is not None:
        validate_color_mapping(pixel_mapping=pixel_mapping, color_mapping=colors)

    return SamplingSpec(
        pixel_mapping=pixel_mapping,
        color_mapping=colors,
        tissue_percentage=min_coverage,
        active_annotations=tuple(
            annotation
            for annotation, pct in min_coverage.items()
            if annotation in pixel_mapping and pct is not None and annotation != "background"
        ),
    )


def _resolve_sampling_spec_from_sampling_params(
    sampling_config: Any,
    *,
    tiling: TilingConfig,
) -> SamplingSpec:
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
