from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from hs2p.wsi.types import CoordinateOutputMode, CoordinateSelectionStrategy, SamplingSpec

from .models import FilterConfig, PreviewConfig, SegmentationConfig, TilingConfig


def resolve_tiling_config(cfg: Any) -> TilingConfig:
    min_coverage = dict(
        _merge_sampling_mapping(cfg.tiling.masks.min_coverage, field_name="min_coverage")
        or {}
    )
    return TilingConfig(
        requested_spacing_um=cfg.tiling.params.requested_spacing_um,
        requested_tile_size_px=cfg.tiling.params.requested_tile_size_px,
        tolerance=cfg.tiling.params.tolerance,
        overlap=cfg.tiling.params.overlap,
        min_coverage=min_coverage,
        independent_sampling=bool(cfg.tiling.independent_sampling),
        backend=cfg.tiling.backend,
    )


def require_tissue_fraction(tiling: TilingConfig) -> float:
    """Tissue-coverage threshold for the binary tissue-tiling path.

    Used only where binary tissue tiling actually gates on the threshold; annotation
    sampling gates on per-class coverage and never calls this. A missing ``tissue`` entry
    is an error rather than a silent ``0.0`` (which would keep every tile and disable
    tissue filtering). An explicit ``0.0`` is honoured as a deliberate opt-out.
    """
    value = tiling.min_coverage.get("tissue")
    if value is None:
        raise ValueError(
            "tiling.masks.min_coverage.tissue is required for tissue tiling. Set it (e.g. "
            "0.01), or sample specific annotations instead of binary tissue."
        )
    return float(value)


def resolve_segmentation_config(cfg: Any) -> SegmentationConfig:
    return SegmentationConfig(**dict(cfg.tiling.seg_params))


def resolve_filter_config(cfg: Any) -> FilterConfig:
    return FilterConfig(**dict(cfg.tiling.filter_params))


def resolve_preview_config(cfg: Any) -> PreviewConfig:
    preview_cfg = cfg.tiling.preview
    return PreviewConfig(
        save_mask_preview=bool(preview_cfg.save_mask_preview),
        save_tiling_preview=bool(preview_cfg.save_tiling_preview),
        downsample=int(preview_cfg.downsample),
        tissue_contour_color=tuple(int(v) for v in preview_cfg.tissue_contour_color),
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


def resolve_output_mode(cfg: Any) -> str:
    """Derive the coordinate output mode from ``tiling.masks.output_mode``.

    ``per_annotation`` (default) writes one coordinate artifact per sampled class;
    ``merged`` writes one merged per-slide artifact (the union of tiles passing any
    class threshold).
    """
    masks_cfg = getattr(cfg.tiling, "masks", None)
    raw = getattr(masks_cfg, "output_mode", None) if masks_cfg is not None else None
    value = str(raw or CoordinateOutputMode.PER_ANNOTATION).lower()
    valid = {CoordinateOutputMode.PER_ANNOTATION, CoordinateOutputMode.MERGED}
    if value not in valid:
        raise ValueError(
            f"tiling.masks.output_mode must be one of {sorted(valid)}, got {value!r}"
        )
    return value


def resolve_sampling_request(
    cfg: Any,
    *,
    tiling: TilingConfig,
) -> tuple[SamplingSpec | None, str | None, str | None]:
    """Resolve the annotation-sampling invocation for the CLI tiling entrypoint.

    Returns ``(sampling, selection_strategy, output_mode)`` when the masks config has been
    customized away from the shipped default, else ``(None, None, None)`` so the caller takes
    the binary tissue-tiling path. The signal is configuration, not label names: a spec equal
    to :func:`build_default_sampling_spec` (the untouched default ``{background, tissue}``
    setup) is binary tissue tiling; any change to the label vocabulary or the sampled set opts
    into multi-label annotation sampling. No label name is reserved at this boundary.
    """
    spec = resolve_sampling_spec(cfg, tiling=tiling)
    if not spec.active_annotations:
        return None, None, None
    # Compare against the default skeleton's constants, not a built default spec: an
    # annotation-only config (no ``tissue`` threshold) must not trip the tissue requirement
    # baked into build_default_sampling_spec just to detect the untouched default.
    is_untouched_default = dict(spec.pixel_mapping) == _DEFAULT_PIXEL_MAPPING and set(
        spec.active_annotations
    ) == set(_DEFAULT_ACTIVE_ANNOTATIONS)
    if is_untouched_default:
        return None, None, None
    return spec, resolve_sampling_strategy(cfg), resolve_output_mode(cfg)


_DEFAULT_PIXEL_MAPPING = {"background": 0, "tissue": 1}
_DEFAULT_ACTIVE_ANNOTATIONS = ("tissue",)


def build_default_sampling_spec(tiling: TilingConfig) -> SamplingSpec:
    return SamplingSpec(
        pixel_mapping=dict(_DEFAULT_PIXEL_MAPPING),
        color_mapping={"background": None, "tissue": None},
        tissue_percentage={
            "background": None,
            "tissue": require_tissue_fraction(tiling),
        },
        active_annotations=_DEFAULT_ACTIVE_ANNOTATIONS,
    )


def _merge_sampling_mapping(
    entries: Any,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    if entries is None:
        return None
    if isinstance(entries, Mapping):
        return {str(key): value for key, value in entries.items()}
    try:
        iterator = list(entries)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a mapping or a list of mappings") from exc
    merged: dict[str, Any] = {}
    for entry in iterator:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{field_name} must be a mapping or a list of single-entry mappings"
            )
        for key, value in entry.items():
            merged[str(key)] = value
    return merged


def validate_pixel_mapping(pixel_mapping: dict[str, int]) -> None:
    """Validate the annotation ``pixel_mapping`` up front, before any slide is opened.

    ``pixel_mapping`` is the user's own label vocabulary — no name is reserved. Each label
    must map to a distinct, non-negative integer within the supported ``uint16`` range so the
    read path can split classes by exact value without collisions or silent wrapping (see
    :func:`hs2p.tiling.mask._as_discrete_label_array`).

    Label *names* become on-disk path components (``output_dir/tiles/<label>/...`` for
    per-annotation artifacts), so they must be safe single path components — no separators,
    no ``.``/``..`` — to keep artifacts inside the run's output directory.
    """
    seen: dict[int, str] = {}
    for annotation, value in pixel_mapping.items():
        if (
            not annotation
            or annotation in {".", ".."}
            or any(ch in annotation for ch in ("/", "\\", "\x00"))
        ):
            raise ValueError(
                f"pixel_mapping label {annotation!r} must be a safe path component "
                "(non-empty, no '/'\\ separators, not '.' or '..')"
            )
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"pixel_mapping['{annotation}'] must be an integer label value, got {value!r}"
            )
        ivalue = int(value)
        if ivalue < 0 or ivalue > 65535:
            raise ValueError(
                f"pixel_mapping['{annotation}']={ivalue} is outside the supported "
                "label range [0, 65535]"
            )
        if ivalue in seen:
            raise ValueError(
                "pixel_mapping values must be unique: "
                f"'{annotation}' and '{seen[ivalue]}' both map to {ivalue}"
            )
        seen[ivalue] = annotation


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


def _drop_null_labels(
    pixel_mapping: dict[str, Any],
    *companions: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any] | None]]:
    """Drop labels removed via the null-to-drop idiom (pixel value set to null) and filter the
    companion mappings (min_coverage, colors) to the surviving labels.

    Configs are deep-merged over the default ``{background:0, tissue:1}`` mapping, so shadowing
    a default value isn't enough to remove a default label — e.g. reusing value ``1`` for a new
    class collides with the default ``tissue:1``. Nulling a label's pixel value removes it
    entirely (mirroring how ``min_coverage.<label>: null`` drops a class from sampling), and the
    label is also stripped from the companion mappings so the cross-mapping checks stay coherent.
    """
    removed = {name for name, value in pixel_mapping.items() if value is None}
    kept = {name: value for name, value in pixel_mapping.items() if value is not None}
    filtered = [
        None if m is None else {k: v for k, v in m.items() if k not in removed}
        for m in companions
    ]
    return kept, filtered


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
    colors = _merge_sampling_mapping(
        getattr(masks_cfg, "colors", None),
        field_name="colors",
    )
    pixel_mapping, (min_coverage, colors) = _drop_null_labels(
        pixel_mapping, min_coverage, colors
    )
    validate_pixel_mapping(pixel_mapping)
    missing_coverage_labels = sorted(set(min_coverage.keys()) - set(pixel_mapping.keys()))
    if missing_coverage_labels:
        raise ValueError(
            "masks.min_coverage references unknown labels: "
            + ", ".join(missing_coverage_labels)
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
            if annotation in pixel_mapping and pct is not None
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
    color_mapping = _merge_sampling_mapping(
        getattr(sampling_config, "color_mapping", None),
        field_name="color_mapping",
    )
    pixel_mapping, (tissue_percentage, color_mapping) = _drop_null_labels(
        pixel_mapping, tissue_percentage, color_mapping
    )
    validate_pixel_mapping(pixel_mapping)
    missing_threshold_labels = sorted(
        set(tissue_percentage.keys()) - set(pixel_mapping.keys())
    )
    if missing_threshold_labels:
        raise ValueError(
            "sampling tissue_percentage references unknown labels: "
            + ", ".join(missing_threshold_labels)
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
            if annotation in pixel_mapping and pct is not None
        ),
    )
