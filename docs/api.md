# Python API

Use `tile_slide()` to compute coordinates in memory and `tile_slides()` to process a
batch and save its artifacts. Both are available from `hs2p`. Install an
[input reader](cli.md#backends) before running the examples.

## Tile, save, and reload one slide

```python
from pathlib import Path

from hs2p import (
    FilterConfig,
    SegmentationConfig,
    SlideSpec,
    TilingConfig,
    load_tiling_result,
    save_tiling_result,
    tile_slide,
)

result = tile_slide(
    SlideSpec(sample_id="slide-1", image_path=Path("/data/slide-1.tif")),
    tiling=TilingConfig(
        backend="auto",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        min_coverage={"tissue": 0.1},
    ),
    segmentation=SegmentationConfig(method="hsv", downsample=64),
    filtering=FilterConfig(ref_tile_size=224, a_t=4, a_h=2),
)

artifact = save_tiling_result(result, output_dir=Path("output"))
loaded = load_tiling_result(
    coordinates_npz_path=artifact.coordinates_npz_path,
    coordinates_meta_path=artifact.coordinates_meta_path,
)
```

`TilingConfig` is keyword-only. `min_coverage={"tissue": 0.1}` requires at least 10%
tissue coverage per tile. See the [artifact reference](artifacts.md) for saved paths,
coordinate units, and metadata.

To use a precomputed tissue mask, set `SlideSpec.mask_path`. Its tissue label is `1`;
the mask is aligned to the slide automatically. You may omit `segmentation` when a
mask is supplied. A slide without a mask requires a `SegmentationConfig`.

If native spacing is missing or incorrect, set `SlideSpec.spacing_at_level_0` to a
finite positive value in µm/px. It becomes the effective level-0 spacing for every
backend; pyramid spacings follow the reader's level downsamples. A conflicting valid
native value produces one warning identifying the slide, backend, and both values.
Missing native spacing is rescued without a conflict warning.

## Process a batch

```python
from pathlib import Path

from hs2p import PreviewConfig, SegmentationConfig, SlideSpec, TilingConfig, tile_slides

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
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        min_coverage={"tissue": 0.1},
    ),
    segmentation=SegmentationConfig(method="hsv", downsample=64),
    preview=PreviewConfig(save_mask_preview=True, save_tiling_preview=True, downsample=32),
    output_dir=Path("output"),
    num_workers=4,
)
```

Sample IDs must be unique within a batch. `tile_slides()` attempts each slide
independently and returns a list of `TilingArtifacts` for successful slides. If any
slides fail, it emits one `BatchPartialFailureWarning` after the batch, naming each
failed slide and its reason. `output/process_list.csv` retains errors and tracebacks.

Pass `save_tiles=True` to export tile JPEGs in TAR files with manifest sidecars.
Pass `resume=True` to reuse completed compatible outputs, or
`read_coordinates_from=Path("saved/tiles")` to reuse coordinates while creating the
currently requested TARs and tiling previews. Mask previews are not regenerated
from reused coordinates. Validation and recovery behavior is
described in [Resume and validation](artifacts.md#resume-and-validation).

Annotation-aware sampling is available through the `sampling` argument on both
functions (`SamplingSpec` from `hs2p.wsi`). With per-annotation output, `tile_slide()`
returns a dictionary of `TilingResult` objects keyed by annotation. Batch sampling
currently rejects `resume`, `read_coordinates_from`, and `save_tiles`. See the
[CLI guide](cli.md) for annotation selection and output modes.

## Results and configuration

| Type | Purpose |
| --- | --- |
| `SlideSpec` | Slide identity, optional mask, and optional level-0 spacing override. |
| `TilingConfig` | Reader selection, requested spacing and tile size, overlap, tolerance, and minimum coverage. |
| `SegmentationConfig` | Tissue-segmentation method and settings. |
| `FilterConfig` | Contour filtering and optional white-space, black-space, grayspace, and blur QC. |
| `PreviewConfig` | Batch preview toggles, downsample, and mask styling. |
| `TilingResult` | In-memory geometry, processing settings, and provenance. |
| `TileGeometry` | Coordinate arrays and the geometry used to read tiles. |
| `TilingArtifacts` | Saved paths, tile count, and backend provenance. |

These types are available from `hs2p`; lower-level preprocessing helpers are exposed
through `hs2p.preprocessing`.

`TilingResult` exposes the arrays `x`, `y`, `tissue_fractions`, and `tile_index`.
Coordinates are level-0 pixel origins. Use `requested_tile_size_px` and
`requested_spacing_um` for the output request; `read_tile_size_px` and
`read_spacing_um` describe the source crop. `tile_size_lv0` is its level-0 footprint,
`step_px_lv0` is the grid stride, and `min_tissue_fraction` is the coverage threshold.

## Previews

`write_tiling_preview(result=..., output_dir=..., downsample=...)` writes a coordinate
preview and returns its path, or `None` for an empty result. Batch tiling provides
the same output through `PreviewConfig(save_tiling_preview=True)`.

`save_mask_preview=True` writes `preview/mask/{sample_id}.jpg`. Tissue tiling shows
contours: evergreen `#255E3B` outer borders and coral `#F26B3A` hole borders.
`tissue_contour_color` changes the outer border; `mask_overlay_alpha` has no effect
on this contour-only preview. Sampling uses filled label masks with its pixel and
color mappings; it also supports a tiling preview for each non-empty coordinate
output. `overlay_mask_on_slide()` is the lower-level overlay helper.

## Reader selection

`TilingConfig.backend` selects the slide reader and `mask_backend` selects the
source-mask reader independently. See [Backends](cli.md#backends) for supported
readers, automatic selection, and decoding errors. This selection also applies to
deferred mask-preview reads. `jpeg_backend` on `tile_slides()` selects the TAR JPEG
encoder separately.

The lower-level helpers `resolve_tissue_mask`, `resolve_annotation_masks`,
`load_precomputed_tissue_mask`, and `load_annotation_label_mask` accept
`mask_backend`. Omitting it or passing `None` selects `auto` from the mask path and
records `requested_mask_backend="auto"`; it does not inherit the slide reader.
The high-level API passes the resolved reader explicitly. A slide without a source
mask does not check mask-reader availability, and its mask provenance is null.

Results, saved metadata, and process-list rows record requested and resolved readers
as `requested_backend` / `backend` and `requested_mask_backend` / `mask_backend`.
Resume compares resolved readers; a change in the requested value alone does not
invalidate compatible artifacts.
