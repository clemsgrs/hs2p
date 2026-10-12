from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from hs2p.fileops import validate_annotation_name
from hs2p.wsi.types import CoordinateSelectionStrategy, SamplingSpec

from .keys import reject_unknown_config_keys
from .models import (
    DEFAULT_MASK_SAMPLING,
    FilterConfig,
    MaskSamplingConfig,
    PreviewConfig,
    SegmentationConfig,
    TilingConfig,
    validate_pixel_mapping,
)
from .values import construct_config


def resolve_tiling_config(cfg: Any) -> TilingConfig:
    reject_unknown_config_keys(cfg.tiling, path="tiling")
    params = _checked_section(cfg.tiling.params, path="tiling.params")
    return construct_config(
        TilingConfig,
        {
            "tiling.params.requested_spacing_um": params.requested_spacing_um,
            "tiling.params.requested_tile_size_px": params.requested_tile_size_px,
            "tiling.params.tolerance": params.tolerance,
            "tiling.params.overlap": params.overlap,
            "tiling.backend": cfg.tiling.backend,
            "tiling.mask_backend": getattr(cfg.tiling, "mask_backend", "auto"),
        },
    )


def _checked_section(section: Any, *, path: str) -> Any:
    """Check a section as the resolver read it, and return it.

    The check on the whole ``tiling`` section sees an interpolated section
    (``params: ${...}`` in an unresolved ``DictConfig``) only as its ``${...}`` string.
    Reading the section dereferences that interpolation anyway, so checking what was read
    catches unknown keys it brings in without evaluating anything the resolver does not.
    """
    if section is not None:
        reject_unknown_config_keys(section, path=path)
    return section


def require_tissue_fraction(masks: MaskSamplingConfig) -> float:
    """Tissue-coverage threshold for the binary tissue-tiling path.

    Used only where binary tissue tiling actually gates on the threshold; annotation
    sampling gates on per-class coverage and never calls this. A missing ``tissue`` entry
    is an error rather than a silent ``0.0`` (which would keep every tile and disable
    tissue filtering). An explicit ``0.0`` is honoured as a deliberate opt-out.
    """
    value = masks.min_coverage.get("tissue")
    if value is None:
        raise ValueError(
            "tiling.masks.min_coverage.tissue is required for tissue tiling. Set it (e.g. "
            "0.01), or sample specific annotations instead of binary tissue."
        )
    return float(value)


def resolve_segmentation_config(cfg: Any) -> SegmentationConfig:
    reject_unknown_config_keys(cfg.tiling.seg_params, path="tiling.seg_params")
    return construct_config(
        SegmentationConfig,
        {
            f"tiling.seg_params.{key}": value
            for key, value in dict(cfg.tiling.seg_params).items()
        },
    )


def resolve_filter_config(cfg: Any) -> FilterConfig:
    reject_unknown_config_keys(cfg.tiling.filter_params, path="tiling.filter_params")
    return construct_config(
        FilterConfig,
        {
            f"tiling.filter_params.{key}": value
            for key, value in dict(cfg.tiling.filter_params).items()
        },
    )


def resolve_preview_config(cfg: Any) -> PreviewConfig:
    preview_cfg = cfg.tiling.preview
    reject_unknown_config_keys(preview_cfg, path="tiling.preview")
    return construct_config(
        PreviewConfig,
        {
            "tiling.preview.save_mask_preview": preview_cfg.save_mask_preview,
            "tiling.preview.save_tiling_preview": preview_cfg.save_tiling_preview,
            "tiling.preview.downsample": preview_cfg.downsample,
            "tiling.preview.tissue_contour_color": preview_cfg.tissue_contour_color,
            "tiling.preview.mask_overlay_alpha": preview_cfg.mask_overlay_alpha,
        },
    )


def resolve_read_coordinates_from(cfg: Any) -> Path | None:
    value = cfg.tiling.read_coordinates_from
    if not value:
        return None
    return Path(value)


_SELECTION_STRATEGIES = {
    "joint": CoordinateSelectionStrategy.JOINT_SAMPLING,
    "independent": CoordinateSelectionStrategy.INDEPENDENT_SAMPLING,
}


def is_binary_tissue_tiling(masks: MaskSamplingConfig) -> bool:
    """Whether ``masks`` asks for binary tissue tiling rather than annotation sampling.

    The signal is configuration, not label names: the shipped ``{background, tissue}``
    vocabulary sampling only ``tissue`` (or sampling nothing) is binary tissue tiling; any
    change to the label vocabulary or the sampled set opts into multi-label annotation
    sampling. Colors, output mode and strategy do not change the decision.
    """
    if not masks.min_coverage:
        return True
    return masks.pixel_mapping == DEFAULT_MASK_SAMPLING.pixel_mapping and set(
        masks.min_coverage
    ) == set(DEFAULT_MASK_SAMPLING.min_coverage)


def resolve_sampling_request(
    *,
    masks: MaskSamplingConfig,
) -> tuple[SamplingSpec | None, str | None, str | None]:
    """Resolve how ``masks`` is tiled.

    Returns ``(sampling, selection_strategy, output_mode)`` for annotation sampling, else
    ``(None, None, None)`` so the caller takes the binary tissue-tiling path (see
    :func:`is_binary_tissue_tiling`).
    """
    if is_binary_tissue_tiling(masks):
        return None, None, None
    sampling = SamplingSpec(
        pixel_mapping=dict(masks.pixel_mapping),
        color_mapping=None if masks.colors is None else dict(masks.colors),
        tissue_percentage=dict(masks.min_coverage),
        active_annotations=tuple(masks.min_coverage),
    )
    return sampling, _SELECTION_STRATEGIES[masks.strategy], masks.output_mode


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


def validate_sampling_spec(sampling: SamplingSpec) -> None:
    """Validate every annotation declaration accepted by direct Python sampling APIs."""
    validate_pixel_mapping(sampling.pixel_mapping)
    _validate_annotation_names(
        sampling.active_annotations,
        sampling.tissue_percentage,
        sampling.color_mapping,
    )


def _validate_annotation_names(*declarations: Any) -> None:
    for declaration in declarations:
        if declaration is None:
            continue
        for annotation in declaration:
            validate_annotation_name(annotation)


def resolve_mask_sampling_config(cfg: Any) -> MaskSamplingConfig:
    """Build the :class:`MaskSamplingConfig` declared by ``tiling.masks``.

    The YAML spellings are resolved here: each label map may be a mapping or a list of
    single-entry mappings, a label whose pixel value is null is dropped from every map
    (configs deep-merge over the shipped ``{background, tissue}`` block, so this is how a
    default label is removed), and a null coverage leaves that label unsampled. A config
    without a ``masks`` section gets :data:`DEFAULT_MASK_SAMPLING`.
    """
    reject_unknown_config_keys(cfg.tiling, path="tiling")
    masks_cfg = _checked_section(
        getattr(cfg.tiling, "masks", None), path="tiling.masks"
    )
    if masks_cfg is None:
        return DEFAULT_MASK_SAMPLING
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
    _validate_annotation_names(pixel_mapping, min_coverage, colors)
    pixel_mapping, (min_coverage, colors) = _drop_null_labels(
        pixel_mapping, min_coverage, colors
    )
    values: dict[str, Any] = {
        "tiling.masks.pixel_mapping": _plain_pixel_lists(pixel_mapping),
        "tiling.masks.colors": colors,
        "tiling.masks.min_coverage": {
            label: coverage
            for label, coverage in min_coverage.items()
            if coverage is not None
        },
    }
    for name in ("output_mode", "strategy"):
        value = getattr(masks_cfg, name, None)
        if value is not None:
            values[f"tiling.masks.{name}"] = value
    return construct_config(MaskSamplingConfig, values)


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


def _plain_pixel_lists(pixel_mapping: dict[str, Any]) -> dict[str, Any]:
    """Turn list-valued entries (an OmegaConf ``ListConfig`` when read from YAML) into plain
    lists. Scalar entries pass through untouched, so scalar-only mappings resolve as before."""
    return {
        name: list(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else value
        for name, value in pixel_mapping.items()
    }
