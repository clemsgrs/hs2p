# Release Notes

## 5.0.0

hs2p 5.0 is a breaking release. Source masks are now first-class `hs2p.Mask` objects
([ADR 0004](adr/0004-first-class-mask.md)), and every public constructor and
multi-argument function is keyword-only
([ADR 0003](adr/0003-keyword-only-constructors.md)). Parent spec #167 is the
authoritative scope reference for the mask work. Configuration-driven workflows
(`tile_slide`, `tile_slides`, the CLI) keep their inputs; artifact metadata and
`process_list.csv` keep every field name, including `mask_level`, `mask_spacing_um`,
`mask_backend` and `requested_mask_backend`.

### First-class source masks

`Mask(path=..., labels=..., backend="auto")` opens a mask with its label semantics
(`TissueLabels(background=..., tissue=...)` or `AnnotationLabels(pixel_mapping=...)`),
`Mask.align_to(...)` binds it to a slide, and `AlignedMask.read_full(...)` /
`read_region(...)` return validated `uint8` labels with the level and effective spacing
read. Flat PNG/JPEG and untagged TIFF masks now work: no spacing metadata is needed. See
[Source masks](api.md#source-masks).

### Removed APIs and their replacements

| Removed in 5.0 | Replacement |
| --- | --- |
| `WSI(mask_path=..., mask_backend=...)`, `WSI.mask_reader`, `WSI.mask_path`, `WSI.mask_backend`, `WSI.requested_mask_backend` | `WSI` is image-only. Open the mask separately: `Mask(path=..., labels=..., backend=...)`. |
| `hs2p.wsi.reader.open_mask_reader`, `hs2p.wsi.backend.open_mask_reader` | `Mask(path=..., labels=..., backend="auto")`; `Mask.backend` is the resolved reader. |
| `hs2p.wsi.masks.read_aligned_mask` (also `hs2p.wsi`) | `Mask.align_to(reference_spacing_um=..., reference_dimensions=...).read_full(target_spacing_um=..., target_dimensions=...).labels` |
| `hs2p.wsi.masks.mask_level_downsamples` (also `hs2p.wsi`) | None. `AlignedMask.level_spacings_um` carries the effective per-level spacings. |
| `hs2p.wsi.masks.read_label_at_spacing` | `Mask.align_to(...).read_full(target_spacing_um=..., target_dimensions=...)` |
| `hs2p.wsi.masks.read_label_region_at_spacing` | `Mask.align_to(...).read_region(location=..., target_spacing_um=..., target_dimensions=...)`, with `location` in the **slide's** level-0 pixels, not the mask file's. |
| `hs2p.tiling.mask.load_precomputed_tissue_mask` (also `hs2p.preprocessing`) | `resolve_tissue_mask(slide=..., mask=Mask(path=..., labels=TissueLabels(...)))`, or `hs2p.tiling.mask.open_tissue_mask(path, pixel_mapping=...)` from a configuration. |
| `hs2p.tiling.mask.load_annotation_label_mask` (also `hs2p.preprocessing`) | `resolve_annotation_masks(slide=..., mask=Mask(path=..., labels=AnnotationLabels(...)))`, or `hs2p.tiling.mask.open_annotation_mask(path, pixel_mapping=...)`. |
| Private helpers `_resolve_mask_backend`, `_select_mask_level`, `_read_mask_level`, `_read_discrete_mask_level`, `_read_label_mask_at_seg`, `_reduce_mask_channels`, `_as_discrete_label_array`, `_mask_label_values`, `_is_discrete_binary_mask`, `_is_label_subset`, `_raise_mask_decode_error`, `_collapse_label_raster`, and `hs2p.tiling.mask.MAX_MASK_READ_PX` | Owned by `hs2p.mask` (`MAX_MASK_READ_PX` lives there). |

### Changed signatures

- `resolve_tissue_mask(*, slide, mask=None, ..., requested_mask_backend=None)`: the
  `tissue_mask_path`, `tissue_mask_tissue_value` and `mask_backend` arguments are gone;
  pass an open tissue `Mask` as `mask`. A `Mask` declaring `AnnotationLabels` is rejected.
- `resolve_annotation_masks(*, slide, mask, seg_downsample=64, tissue_method=...,
  requested_mask_backend=None)` (exported from `hs2p.api`): `mask_path`, `pixel_mapping`
  and `mask_backend` are gone; pass an open annotation `Mask` as `mask`. A `Mask`
  declaring `TissueLabels` is rejected. Open failures now come from the `Mask`
  constructor, not the resolver.
- `preprocess_slide(...)`: `tissue_mask_tissue_value=` is replaced by `pixel_mapping=`,
  from which `TissueLabels` are built (`background` and `tissue`, defaults 0 and 1).
- `overlay_mask_on_slide(...)`: `annotation_mask_path` and `mask_backend` are replaced by
  `mask` (an open `Mask`). `write_coordinate_preview(...)`: `mask_path` and `mask_backend`
  are replaced by `mask`. `draw_grid_from_coordinates(...)`: `mask` is an `AlignedMask`.
  `write_annotation_tiling_preview(...)`: `mask_path` is replaced by `mask`, whose
  lifetime the caller owns.
- New: `hs2p.tiling.mask.open_tissue_mask(mask_path, *, pixel_mapping=None, backend="auto")`
  and `open_annotation_mask(mask_path, *, pixel_mapping, backend="auto")` context managers.
- `open_slide`, `resolve_backend` and every backend reader constructor accept
  `require_spacing: bool = True`; with `False` a source without spacing metadata opens
  with `native_spacing = None`, `spacing = None` and `spacings = []`.

### Behavior changes

- **Alignment is a hard error.** A mask whose shape does not cover the slide at one scale
  (within one mask pixel per axis), or whose spacing tag differs from its
  dimension-derived spacing by more than 5%, fails with
  `ValueError: Mask alignment failed for path=... with backend=...: mask dimensions WxH,
  reference dimensions WxH, effective spacing X um/px[, file spacing Y um/px]. ...`
  instead of being stretched over the slide. In batch runs the slide fails and its
  `process_list.csv` row records the error.
- **Nominal spacing tags warn.** A tag 1–5% off the dimension-derived spacing logs one
  `WARNING` per alignment: `Mask spacing disagreement for path=... File spacing differs
  from the effective spacing by N%; the effective spacing is used and recorded.`
- **Tissue masks are validated against a closed vocabulary.** A tissue mask holding any
  value other than the declared background (default 0) and tissue (default 1) IDs fails
  with `ValueError: Mask read produced invalid labels for path=... with backend=... at
  level N: undeclared label IDs [...]; declared [0, 1]`. A `{1, 255}` mask used to pass
  with 255 treated as background. A multi-channel decode is accepted only when its
  channels are identical (the legacy path kept channel 0).
- **`mask_spacing_um` is the effective spacing.** Results, metadata and process rows now
  record the dimension-derived spacing of the level read, which can differ from the
  file's tag by up to 5%. For the repository fixture the shift is about `4e-6` relative.
- **Mask level selection uses the shared label selector at a fixed 1% tolerance** (the
  nearest level when within 1%, otherwise the coarsest level finer than the target, or
  level 0 upsampled), independent of `tiling.params.tolerance`, so `mask_level` can
  differ from 4.x for some pyramids.
- **Regional reads return exactly the requested dimensions**, sample by coordinate so a
  region equals the matching crop of a full read, and take `location` in the slide's
  level-0 pixels. A request beyond the slide canvas raises `ValueError` instead of being
  padded with 255. The legacy helper returned an unresized native window when the level
  was within tolerance and interpreted `location` in the mask file's pixels.
- **Previews validate like preprocessing.** Source-backed mask and tiling previews now
  reject undeclared or out-of-range values, refuse native reads above 256 Mpx, and align
  by dimension ratio (spacing-less masks work in previews). The silent `uint8` narrowing
  and reader leak of the legacy preview read are gone.
- **Error text.** Mask errors read `Mask open failed ...`, `Mask decode failed ...`,
  `Mask read produced invalid labels ...`, `Mask read refused ...`, `Mask region read
  refused ...`. The `Precomputed tissue mask decode failed ... Select another backend or
  regenerate the mask.` and `Annotation mask read produced non-discrete labels ...` texts
  are gone.
- `ResolvedAnnotationMasks.pixel_mapping` is rebuilt from the normalized labels: a
  single-value list such as `{"tumor": [1]}` comes back as `{"tumor": 1}`.
- **Untagged slides under VIPS raise.** A slide with no resolution metadata opened with
  `backend="vips"` (or resolved to VIPS by `auto`) now raises
  `Unable to infer slide spacing for path=... with backend=vips`, like every other
  backend, instead of opening at `1000 µm/px`. Pass `spacing_at_level_0`, or
  `require_spacing=False` where a spacing-less source is acceptable.

### Keyword-only constructors and functions

Positional construction of `SegmentationConfig`, `FilterConfig`, `PreviewConfig`,
`SlideSpec`, `TilingArtifacts`, `CompatibilitySpec`, `SamplingSpec`, `TilingResult`,
`TileGeometry`, `ContourResult`, `ResolvedTissueMask`, `ResolvedAnnotationMasks`,
`Sam2Thumbnail`, `LevelSelection`, `SpacingReadPlan` and `WSI` raises `TypeError`;
pass keywords. `TilingConfig` and the `hs2p.mask` classes were already keyword-only.

Positional calls to `WSI.get_tile`, `WSI.get_best_level_for_spacing`,
`WSI.read_region_at_spacing`, `compute_tile_coverage`, `overlay_mask_on_slide`,
`overlay_mask_on_tile`, `draw_grid`, `draw_grid_from_coordinates`, `promote_temp_file`,
`write_config`, `make_white_canvas` and `paste_region` (and the re-exported helpers
`_tiles_for_contour` and `_compute_tile_coverage`) raise `TypeError` (#212).
`read_region(location, level, size)` and `read_regions` keep the OpenSlide idiom.
Saved artifacts are unaffected.

### Downstream

- **soma** (`hs2p>=4.4.2`, no upper bound) breaks on 5.0: `soma/dense/reader.py` calls
  the removed `read_label_at_spacing` / `read_label_region_at_spacing`, and
  `soma/curation/segmentation_coverage.py` calls `resolve_annotation_masks` with the old
  signature. The replacement is `Mask.align_to(...).read_full(...)` / `.read_region(...)`
  with slide coordinates (soma currently passes mask-file coordinates, which misregisters
  whenever mask level 0 is coarser than slide level 0; `AlignedMask` removes that failure
  mode) and `resolve_annotation_masks(slide=..., mask=...)`. Its positional
  `WSI(Path(path), backend=...)` and `wsi.read_region_at_spacing(location, spacing, size,
  ...)` calls also need keywords.
- **slide2vec** uses no mask API and constructs every affected class with keywords; it is
  unaffected.

## 4.4.0 – 4.5.0

### Several mask values under one label

A `tiling.masks.pixel_mapping` entry may now be a list of raster values, e.g.
`tumor: [1, 2]`. The values are sampled as one class: its binary mask is their union, so
`min_coverage` applies to their summed coverage. `output_mode: merged` cannot express
this, because it is the union of tiles passing each label's own threshold. Lists must be
non-empty and no value may appear twice, within or across labels. Scalar entries are
unchanged and are not rewritten as lists.

### Faster tiling previews

Tiling previews (`tiling.preview.save_tiling_preview`) are now rendered by drawing the
grid directly on the loaded slide canvas when no annotation overlay is requested,
instead of cropping and re-pasting every tile. The overlay path is unchanged. The
preview stage also runs in up to `speed.num_workers` spawned processes rather
than threads, including when previews are materialized from reusable coordinate
artifacts. A single worker renders inline to avoid process startup and backend
re-import overhead. On dense prostatectomy slides (60k-190k tiles) this cuts the
preview stage from ~19 s to ~2 s per slide with 16 workers. Grid lines are now
complete: the previous per-tile paste erased the shared edge of already-drawn
neighbours when tiles did not fall on whole preview pixels.

### Format-aware automatic slide and mask backends

Flat `.png`, `.jpg`, and `.jpeg` slide or source-mask inputs now select the new
PIL reader directly under `auto` (case-insensitive). Pillow is a core dependency,
so ordinary flat benchmark images no longer require libvips or another native
WSI backend. Flat rasters have one level and require
`spacing_at_level_0`; PNG/JPEG density metadata is not interpreted as pathology
spacing. The input-reader setting `tiling.backend: pil` is separate from
`speed.jpeg_backend: pil`, which selects Pillow as the saved-tile JPEG encoder.

This routing is authoritative: corrupt, unsupported, or oversized flat rasters
fail through PIL without another backend probe or recommendation. PIL enforces
the project-owned `PIL_MAX_IMAGE_PIXELS` ceiling before pixel decoding,
independent of Pillow's mutable global ceiling. Explicit backend choices remain
authoritative.

Other inputs continue to probe slide and source-mask paths independently with
the same openability-only priority:

`cucim -> vips -> openslide -> asap`

PIL is never considered in that multi-resolution chain. Automatic selection
still stops at the first reader that opens a source; it does not inspect mask
labels or retry after a later decode failure.

Slide probes now receive `spacing_at_level_0`, allowing the override to rescue missing native
spacing metadata during automatic selection. Probe-time spacing-discrepancy warnings are
suppressed, while the selected reader continues to emit the single contextual warning.

This priority change can select a different resolved backend for an existing configuration
that uses `auto`. Because resume compatibility records the resolved slide and mask backends,
previous backend-dependent artifacts may be rejected and recomputed. Pin `backend` and
`mask_backend` explicitly when preserving the previous decoder is required.
