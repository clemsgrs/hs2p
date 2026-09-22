# ADR 0004: A first-class `Mask` domain object

- Status: Accepted
- Date: 2026-09-22
- Scope: parent spec #167 (child tickets #210, #190, #191, #192, #193, #194, #195, #218, #196)

## Context

Until 4.5, an externally supplied mask was a reader-shaped adjunct to a slide. `WSI` could
carry an attached mask reader, and three helpers (`_select_mask_level`, `read_aligned_mask`,
`read_label_at_spacing`) each selected a pyramid level their own way. The preview path
narrowed to `uint8` without range checks, validated no vocabulary, applied no read-size guard
and leaked readers. Every backend constructor required level-0 spacing, so the flat PNG/JPEG
masks that `auto` routes to PIL could not be opened at all. Nothing tied a mask path to the
meaning of its pixels.

## Definitions

- **Mask**: an externally supplied, source-backed label raster with one closed label
  semantics, its own backend and its own resolution. Generated HSV or SAM2 tissue arrays are
  not masks in this sense; they stay in-memory arrays.
- **Mask label ID**: an integer in `0..255` stored in the raster. Wider integer storage is
  accepted only when every decoded value stays in that range; values outside it, non-integer
  dtypes and colour rasters (channels that differ) are invalid.
- **Mask label semantics**: the closed vocabulary a mask is read under. `TissueLabels`
  declares the background and tissue IDs; `AnnotationLabels` declares the complete
  name-to-IDs mapping, where a label may own several IDs and no ID is claimed twice. Every
  decoded ID must be declared; a read may contain any subset of the declared IDs.
- **Mask alignment**: binding a mask to a reference level-0 grid (a slide's spacing and
  dimensions) that the mask must cover in full from a shared origin.
- **Aligned mask**: the result of alignment. It answers full-canvas and regional reads in
  the reference grid's level-0 pixel coordinates and reports, per read, the mask level and
  the effective spacing actually read.

## Decision

### The `Mask` / `AlignedMask` boundary

`hs2p.mask.Mask(*, path, labels, backend="auto")` owns source access: it resolves `auto`
from the mask path alone (never from the slide), opens one reader with
`require_spacing=False`, exposes the concrete backend it opened and closes the reader
through `close()` or a `with` block. It is the only way production code opens a source mask.

`Mask.align_to(*, reference_spacing_um, reference_dimensions)` validates geometry once and
returns an `AlignedMask`. `AlignedMask.read_full(...)` and `.read_region(...)` decode,
validate, and nearest-neighbour resample to the exact requested dimensions; both return a
`MaskRead` (read-only 2-D `uint8` labels, `read_level`, `read_spacing_um`). Reads through an
aligned view fail with `ValueError("Mask is closed ...")` once the parent mask closes.

`WSI` is image-only. Preprocessing resolvers, overlays and previews consume a `Mask` or an
`AlignedMask`; configuration-driven entry points construct and close masks internally.
Requested-backend provenance stays in orchestration and artifacts, outside `Mask`.

### Ratio-authoritative alignment

The mask-to-reference dimension ratio decides alignment; file spacing is only cross-checked.

- Shape check: one scale `s` must exist with `|mask_width - reference_width / s| <= 1` and
  `|mask_height - reference_height / s| <= 1`, i.e. one mask level-0 pixel of rounding per
  axis. Otherwise `ValueError` (the mask does not cover the reference at one scale).
- Effective level-0 spacing is `reference_spacing_um * reference_width / mask_width`;
  effective level spacings follow the mask's level downsamples. This value drives level
  selection and is what `read_spacing_um` (and so `mask_spacing_um`) records.
- Spacing cross-check, only when the file carries native spacing, with `d` the relative
  difference between the file's level-0 spacing and the effective spacing: `d < 1%` silent;
  `1% <= d <= 5%` one warning per `align_to`; `d > 5%` `ValueError`.
- A mask without native spacing skips the cross-check. Its only guard is the shape check.
  This is a deliberately weaker guarantee: a same-shaped but unrelated raster is not
  detected, and the effective spacing is inferred rather than confirmed.

This replaced the strict physical-extent check first proposed for #167 (mask spacing times
mask dimensions must equal the slide extent within one pixel). That rule rejected two classes
of valid, pixel-for-pixel registered masks: masks with no spacing metadata at all (flat
PNG/JPEG, untagged TIFF), and masks with nominal spacing tags such as `4.0 um` written for a
slide at `0.2431 um / 16 = 3.8896 um`, which differ by a few percent in physical extent but
register exactly. Under the ratio rule both align; the second also warns.

### Fixed thresholds

- **1% is noise.** Level selection reuses `select_level_for_spacing_read(content_kind="label",
  tolerance=0.01)`: a level within 1% of the requested spacing is read as exact, so float
  noise in a request never drops to a finer level. The same 1% is the silent band of the
  spacing cross-check.
- **5% is the ceiling.** It is the most a mask's spacing metadata may deviate from its
  dimension-derived spacing before alignment fails. Between 1% and 5% hs2p trusts the
  dimensions and tells the user the tag is off.
- The 256 Mpx native-read cap applies to every full level or region window before decoding.

None of these is configurable.

### The 5.0 cut

The legacy interfaces (`WSI` mask arguments and attributes, `open_mask_reader`,
`read_aligned_mask`, `mask_level_downsamples`, `read_label_at_spacing`,
`read_label_region_at_spacing`, `load_precomputed_tissue_mask`, `load_annotation_label_mask`,
and the private mask helpers) are removed in 5.0 without deprecated wrappers, and alignment
failures are hard errors from 5.0.0 with no warn-first period.

**Rejected: a compatibility facade.** Keeping the old functions as wrappers over `Mask`
would have preserved two coordinate conventions for regional reads (mask-file pixels versus
reference pixels), two level-selection results, and the silently-stretching alignment that
5.0 exists to remove. The wrappers could not be behaviour-preserving and correct at once, and
every downstream caller would still have to be audited. A documented break with explicit
replacements (see the [5.0 release notes](../release-notes.md)) is cheaper and safer.

### Non-goals (physical geometry)

`Mask` supports one geometry: a mask that covers the whole reference canvas from a shared
zero origin at one isotropic scale. Out of scope, now and as constraints on future changes:
anisotropic x/y spacing (the per-axis factors differ only by raster rounding); nonzero
origins, affine transforms or rotation; cropped or partial masks; reference-free reads;
implicit or configurable padding for out-of-canvas regions; an explicit `spacing_um`
override on `Mask` (addable later without a break because the constructor is keyword-only).

## Consequences

- Preprocessing, previews, full and regional reads share one alignment, one level selector,
  one validation and one read-size guard; a fix in `hs2p.mask` applies everywhere.
- Flat PNG/JPEG and untagged TIFF masks work. Under VIPS this required #218: untagged
  slides now raise instead of opening at `1000 um/px`.
- Masks that used to be stretched over a slide now fail; masks whose tag is 1–5% off now
  warn; `mask_spacing_um` can differ from the file's tag by that amount.
- `mask_level`, `mask_spacing_um`, `mask_backend` and `requested_mask_backend` keep their
  names and places in metadata and `process_list.csv`.
- ADR 0002's independent mask-backend decision stands; the `open_mask_reader` helper and the
  `WSI` attached-mask path it originally described no longer exist.
- Downstream: soma must move from `read_label_at_spacing` / `read_label_region_at_spacing`
  to `AlignedMask` reads with reference-grid coordinates and adopt the new
  `resolve_annotation_masks` signature. slide2vec uses no mask API.
