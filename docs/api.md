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

Every public constructor is keyword-only
([ADR 0003](adr/0003-keyword-only-constructors.md)). `min_coverage={"tissue": 0.1}` requires at least 10%
tissue coverage per tile. See the [artifact reference](artifacts.md) for saved paths,
coordinate units, and metadata.

To use a precomputed tissue mask, set `SlideSpec.mask_path`. Its background label is
`0` and its tissue label `1`; any other value in the raster fails the read. The mask may
be a pyramidal TIFF or a flat PNG/JPEG and is aligned to the slide by its dimensions
(see [Source masks](#source-masks)). You may omit `segmentation` when a mask is
supplied. A slide without a mask requires a `SegmentationConfig`.

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

## Source masks

`hs2p.Mask` opens an externally supplied label raster (pyramidal TIFF, flat PNG/JPEG,
untagged TIFF) together with the meaning of its pixels. `TissueLabels(background=...,
tissue=...)` declares a binary mask; `AnnotationLabels(pixel_mapping=...)` declares the
complete name-to-values mapping in the configuration's `pixel_mapping` shape. A `Mask`
owns one reader and is context-managed; `close()` is idempotent and any read through it,
or through a view aligned from it, fails afterwards with `ValueError("Mask is closed ...")`.
No spacing metadata is required: a flat PNG works as is.

```python
from hs2p import Mask, TissueLabels

# A 6x4 px PNG covering a 12x8 px slide read at 0.5 µm/px.
with Mask(
    path="/data/slide-1-tissue-mask.png",
    labels=TissueLabels(background=0, tissue=1),
) as mask:
    print(mask.backend)  # "pil"
    aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(12, 8))
    full = aligned.read_full(target_spacing_um=2.0, target_dimensions=(3, 2))
    region = aligned.read_region(
        location=(4, 2), target_spacing_um=1.0, target_dimensions=(2, 2)
    )

full.labels            # 2x3 uint8 array holding only 0 and 1
full.read_level        # 0
full.read_spacing_um   # 1.0: the slide's 0.5 µm/px times 12 / 6
```

`tests/test_mask.py::test_documented_flat_png_example_runs` executes this example.

**Backend.** `backend="auto"` (the default) resolves the reader from the mask path alone
and never inherits the slide reader; a concrete backend is authoritative. `Mask.backend`
is the concrete reader that opened the file. Open failures raise `ValueError` or
`RuntimeError` naming the path and backend.

**Alignment.** `align_to(reference_spacing_um=..., reference_dimensions=...)` binds the
mask to a slide's level-0 grid, which the mask must cover in full from a shared origin.
The mask-to-slide dimension ratio is authoritative: both axes must agree on one scale
within one mask pixel of rounding, and the effective mask spacing is
`reference_spacing_um * reference_width / mask_width`. When the file carries a spacing
tag it is only cross-checked: within 1% of the effective spacing nothing is logged, from
1% to 5% one warning names both values, and above 5% alignment fails. A mask whose shape
does not fit the slide at one scale fails instead of being stretched. A mask without
spacing metadata is guarded by the shape check only. The rule and its thresholds are
recorded in [ADR 0004](adr/0004-first-class-mask.md).

**Reads.** `read_full(target_spacing_um=..., target_dimensions=...)` returns the whole
canvas; `read_region(location=..., target_spacing_um=..., target_dimensions=...)` returns
a window whose `location` is the top-left `(x, y)` in the slide's level-0 pixels, the same
convention as `WSI` reads, never the mask file's own pixels. Both read the mask level
nearest the target when it is within 1%, otherwise the coarsest level finer than the
target, and level 0 upsampled when none is fine enough; they resample with
nearest-neighbour and always return exactly `target_dimensions`. Every
decode is validated before resampling: integer dtype, values in `0..255`, identical
channels if any, and only declared IDs; the result is a read-only 2-D `uint8` array in
`MaskRead.labels`, with `read_level` and `read_spacing_um` (the effective spacing of the
level read). A region extending past the slide canvas raises `ValueError`; nothing is
padded. `dimensions_within_canvas(location=..., target_spacing_um=...,
target_dimensions=...)` returns how much of a region lies on the canvas under the same
rule, for callers that handle the overhang themselves. A native level or window above 256 Mpx is refused before decoding. A region
whose `target_spacing_um / reference_spacing_um` ratio is a simple fraction (1, 2, 1/2,
...) up to float noise (`SPACING_RATIO_RTOL`, 1e-6 relative) is sampled at that exact
fraction: a target spacing that differs from the reference spacing only by float32
rounding, as when a TIFF tag and decimal metadata disagree, still returns the native crop.

**Failures** are `ValueError` for invalid semantics, geometry, labels or requests and
`RuntimeError` for backend open and decode errors; messages start with `Mask open failed`,
`Mask alignment failed`, `Mask decode failed`, `Mask read produced invalid labels`,
`Mask read refused` or `Mask region read refused` and name the path and backend.

The lower-level resolvers `resolve_tissue_mask` and `resolve_annotation_masks` consume
an open `Mask` (declaring `TissueLabels` or `AnnotationLabels`, respectively) whose
lifetime the caller owns. `hs2p.tiling.mask.open_tissue_mask(path, pixel_mapping=...)`
and `open_annotation_mask(path, pixel_mapping=...)` build the labels from a configuration
`pixel_mapping` and return the context-managed mask. Pass `requested_mask_backend` to a
resolver to record what was asked for; otherwise the mask's concrete backend is recorded
as the request. `overlay_mask_on_slide` and `write_coordinate_preview` take the same
open `Mask`; `draw_grid_from_coordinates` takes an `AlignedMask`.

## Reader selection

`TilingConfig.backend` selects the slide reader and `mask_backend` selects the
source-mask reader independently. See [Backends](cli.md#backends) for supported
readers, automatic selection, and decoding errors. This selection also applies to
deferred mask-preview reads. `jpeg_backend` on `tile_slides()` selects the TAR JPEG
encoder separately. The high-level API opens each source mask with the resolved reader;
a slide without a source mask does not check mask-reader availability, and its mask
provenance is null.

Results, saved metadata, and process-list rows record requested and resolved readers
as `requested_backend` / `backend` and `requested_mask_backend` / `mask_backend`.
Resume compares resolved readers; a change in the requested value alone does not
invalidate compatible artifacts.
