# Python API

The Python API is the best entrypoint when you want to integrate `hs2p` into your own pipeline instead of driving it through the CLI. The current public surface is split into:

- high-level orchestration in `hs2p.api`
- the canonical in-memory result model in `hs2p.preprocessing`

## Main public types

- `SlideSpec`
  - Identifies one slide via `sample_id`, `image_path`, and optional `mask_path`
  - `SlideSpec` stays generic because it is shared across tiling and sampling internals
  - A finite positive `spacing_at_level_0` is the effective level-0 spacing for every
    backend, including when native metadata is missing. Every effective pyramid spacing
    is this value multiplied by the backend's level downsample.
  - If valid native metadata disagrees with the override beyond tight floating-point
    equivalence, the selected backend emits one warning with the slide path, native
    value, supplied value, and backend. Missing metadata is rescued without a conflict
    warning.
- `TilingConfig`
  - Requested backend, spacing, tile size, overlap, padding, and minimum tissue fraction
- `SegmentationConfig`
  - Tissue-segmentation settings used before coordinate extraction
- `FilterConfig`
  - Contour filtering plus optional coarse tile QC for white-space, black-space, grayspace, and blur
- `PreviewConfig`
  - Batch preview toggles, preview downsample, and mask preview styling
- `TilingResult`
  - Canonical in-memory result model from `hs2p.preprocessing`
- `TileGeometry`
  - Canonical geometry container with `x`, `y`, `tissue_fractions`, and `tile_index`
- `TilingArtifacts`
  - Lightweight record of saved artifact paths and optional preview/tar outputs

## Canonical result contract

`TilingResult` is the only supported tiling-result model. Downstream code should use:

- `x`
- `y`
- `tissue_fractions`
- `tile_index`
- `requested_tile_size_px`
- `requested_spacing_um`
- `read_tile_size_px`
- `read_spacing_um`
- `tile_size_lv0`
- `step_px_lv0`
- `min_tissue_fraction`

## Single-slide tiling

```python
from pathlib import Path

from hs2p import (
    FilterConfig,
    SegmentationConfig,
    SlideSpec,
    TilingConfig,
    save_tiling_result,
    tile_slide,
)

result = tile_slide(
    SlideSpec(
        sample_id="slide-1",
        image_path=Path("/data/slide-1.tif"),
        mask_path=Path("/data/slide-1-tissue-mask.tif"),
    ),
    tiling=TilingConfig(
        backend="openslide",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
    ),
    segmentation=SegmentationConfig(method="hsv", downsample=64),
    filtering=FilterConfig(ref_tile_size=224, a_t=4, a_h=2),
)

artifacts = save_tiling_result(result, output_dir=Path("output"))
```

Use `tile_slide()` when you want an in-memory result for one slide.

## Batch tiling

Use `tile_slides()` when you want to process multiple slides.   
Results will be automatically written do disk.

```python
from pathlib import Path

from hs2p import (
    FilterConfig,
    PreviewConfig,
    SegmentationConfig,
    SlideSpec,
    TilingConfig,
    tile_slides,
)

slides = [
    SlideSpec(
        sample_id="slide-1",
        image_path=Path("/data/slide-1.tif"),
        mask_path=Path("/data/slide-1-tissue-mask.tif"),
    ),
    SlideSpec(sample_id="slide-2", image_path=Path("/data/slide-2.tif")),
]

artifacts = tile_slides(
    slides,
    tiling=TilingConfig(
        backend="auto",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
    ),
    segmentation=SegmentationConfig(method="hsv", downsample=64),
    filtering=FilterConfig(ref_tile_size=224, a_t=4, a_h=2),
    preview=PreviewConfig(
        save_mask_preview=True,
        save_tiling_preview=True,
        downsample=32,
        tissue_contour_color=(37, 94, 59),
        mask_overlay_alpha=0.5,
    ),
    output_dir=Path("output"),
    num_workers=4,
)
```

`tile_slides()` attempts every requested slide independently. If one or more
slides fail, it returns the `TilingArtifacts` for successful slides and emits
one `BatchPartialFailureWarning` after the batch has finished. The warning names
every failed slide and its reason. Full per-slide errors and tracebacks remain
available in `output/process_list.csv`.

An all-success batch returns every artifact without emitting this warning.

Pass `read_coordinates_from=Path("saved/tiles")` to reuse compatible
`{sample_id}.coordinates.*` files without recomputing coordinates. This does not mark the
slide as already processed: `save_tiles=True` still writes the TAR and its manifest sidecar,
and `PreviewConfig(save_tiling_preview=True)` still renders a preview for non-empty
coordinates. Outputs that are not enabled are not created.

When `save_mask_preview=True`, `tile_slides()` writes `preview/mask/{sample_id}.jpg`
as a contour-only slide preview. The outer tissue boundary uses evergreen
`#255E3B`, while hole contours use coral `#F26B3A`. `tissue_contour_color`
controls the outer border color for this preview path. `mask_overlay_alpha`
does not affect this contour-only path.

The sampling preview path still uses the multi-label filled-mask renderer when
pixel and color mappings are provided.

## Saving and loading artifacts

```python
from hs2p import load_tiling_result

loaded = load_tiling_result(
    coordinates_npz_path=artifacts[0].coordinates_npz_path,
    coordinates_meta_path=artifacts[0].coordinates_meta_path,
)
```

## Preview helpers

- `write_tiling_preview(result=..., output_dir=..., downsample=...)`
- `overlay_mask_on_slide(...)`

The lower-level WSI helpers use a single public mask name:

- `extract_coordinates(..., mask_path=...)`
- `sample_coordinates(..., mask_path=...)`
- `filter_coordinates(..., mask_path=...)`

Internally, the shared coordinate engine still uses a generic `mask_path`.

## Backend selection

`TilingConfig.backend` (slide reader) and `TilingConfig.mask_backend` (source-mask reader)
each support:

- `auto`
- `pil`
- `cucim`
- `vips`
- `openslide`
- `asap`

`auto` classifies each input by suffix. `.png`, `.jpg`, and `.jpeg`
(case-insensitive) select only PIL; a flat-raster open or size failure is final.
Other inputs use the unchanged `cucim -> vips -> openslide -> asap`
openability chain, which never considers PIL. Both fields reject null and unknown
values when the `TilingConfig` is constructed. An explicit backend remains
authoritative.

The slide backend is resolved from the slide path and the mask backend from the source-mask
path — independently, with the same format-aware `auto` policy (no label-semantics
inspection and no retry after selection). An explicit mask backend applies to every source-mask
read: precomputed tissue masks, annotation masks, the low-level readers
(`resolve_tissue_mask`, `resolve_annotation_masks`, `load_precomputed_tissue_mask`,
`load_annotation_label_mask`, all of which accept a `mask_backend`), `overlay_mask_on_slide`,
`WSI(..., mask_path=..., mask_backend=...)`, and deferred preview reads. A slide with no source
mask never resolves or validates mask-backend availability, and its mask provenance is null.

A low-level mask reader called **without** a `mask_backend` (omitted or `None`) resolves the
mask backend independently from the mask path via `auto` — it never inherits the slide's
backend — and records `requested_mask_backend == "auto"`. The high-level pipeline is unaffected
because it always passes an explicit resolved `mask_backend`. `TilingConfig` is keyword-only, so
every field (including `mask_backend`) must be passed by name.

`tiling.backend="pil"` is the input reader for flat rasters. It is distinct from
`speed.jpeg_backend="pil"`, which selects Pillow only as the JPEG encoder for
saved tile TARs.

Opening a source mask that the selected backend cannot decode fails with actionable context
naming the mask path and the requested backend (and the resolved backend when known) rather than
surfacing a raw codec error — for example
`Mask open failed for path=... with backend=<resolved> (requested=<requested>): <error>. Select
another mask backend or verify the mask file.` The remedy is to set `mask_backend` explicitly.
For a flat raster that `auto` assigns to PIL, the error instead asks the caller
only to verify the file; it does not recommend another backend.

`TilingResult`, `TilingArtifacts`, the tiling metadata, and `process_list.csv` record both the
requested and resolved slide and mask backends separately (`requested_backend` / `backend` and
`requested_mask_backend` / `mask_backend`); requested values are provenance only. Resume
compares the resolved slide backend and, when a source mask exists, the resolved mask backend,
and does not reject artifacts merely because a requested value differs (including `requested_backend`).
