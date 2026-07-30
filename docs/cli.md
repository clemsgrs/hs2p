# CLI Guide

hs2p provides a single batch entrypoint:

```
hs2p /path/to/config.yaml [opts...]
```

The `mask_path` column in the input CSV is interpreted as a tissue mask.
Multi-label annotation sampling is driven by the same entrypoint via `tiling.masks` config.

## Input CSV schemas

### Tiling

```csv
sample_id,image_path,mask_path
slide-1,/data/slide-1.tif,/data/slide-1-tissue-mask.tif
slide-2,/data/slide-2.tif,
...
```

### Sampling

```csv
sample_id,image_path,mask_path
slide-1,/data/slide-1.tif,/data/slide-1-annotations.tif
slide-2,/data/slide-2.tif,/data/slide-2-annotations.tif
...
```

### Optional spacing override

Works in either mode:

```csv
sample_id,image_path,mask_path,spacing_at_level_0
slide-1,/data/slide-1.tif,,0.25
slide-2,/data/slide-2.tif,/data/slide-2-tissue-mask.tif,
...
```

The override must be finite and greater than zero. When supplied, it is authoritative:
level 0 uses that spacing and every other level uses the override multiplied by the
selected backend's downsample factor. It can therefore rescue missing spacing metadata.
If existing native spacing genuinely differs, the selected backend warns once with the
slide path, native value, supplied value, and backend; floating-point representation
noise and missing native metadata do not produce a conflict warning.

## Quick start

Start from [`hs2p/configs/default.yaml`](../hs2p/configs/default.yaml), then edit:

- `csv`
- `output_dir`
- `tiling.backend`
- `tiling.mask_backend`
- `tiling.params.requested_spacing_um`
- `tiling.params.requested_tile_size_px`

Run:

```bash
hs2p /path/to/config.yaml
```

## Installation and backends

Base install:

```bash
pip install hs2p
```

Optional extras:

```bash
pip install "hs2p[openslide]"
pip install "hs2p[asap]"
pip install "hs2p[vips]"
pip install "hs2p[cucim]"
pip install "hs2p[turbojpeg]"
pip install "hs2p[all]"
```

The base install includes Pillow both as the flat-raster input reader
(`tiling.backend: pil`) and as the portable default JPEG encoder for tile TAR
export (`speed.jpeg_backend: pil`). These settings are independent: the first
chooses how hs2p reads an input, while the second chooses how it encodes saved
tiles. The `turbojpeg` extra installs PyTurboJPEG as an optional performance
encoder and is also included in `all`.

`tiling.backend` (slide reader) and `tiling.mask_backend` (source-mask reader) both support:

- `auto`
- `pil`
- `cucim`
- `vips`
- `openslide`
- `asap`

For `.png`, `.jpg`, and `.jpeg` inputs (case-insensitive), `auto` selects only
PIL. A corrupt, unsupported, or oversized flat raster fails through PIL without
another backend probe or recommendation. Other inputs use the unchanged
`cucim -> vips -> openslide -> asap` openability chain, which never considers
PIL. Null and any other value are rejected up front by configuration validation
(including when constructing `TilingConfig` directly in Python).

### Independent slide and mask backends

The slide backend and the mask backend are resolved **independently, each from its own path**:
`tiling.backend` from the slide path, `tiling.mask_backend` from the source-mask path. They
share the same format-aware selection policy. Flat-raster suffixes route directly
to PIL; other suffixes use the native-backend openability chain. Neither path
inspects decoded label semantics or retries after selection. A slide with no
source mask never resolves or validates mask-backend availability, and its mask
provenance is recorded as null.

Because `auto` is openability-only, a backend that can *open* but not *decode* a mask (e.g.
cuCIM opening a deflate-compressed label TIFF whose pixels it cannot decode) can be selected
and then fail at read time. When that happens, set `tiling.mask_backend` explicitly to a
backend that decodes the mask (e.g. `openslide`). An unknown mask backend fails at
configuration time; an unavailable or incompatible one fails only when the mask is read, with
the mask path and requested backend named in the error.

Both the requested and resolved slide and mask backends are recorded as provenance in the
tiling metadata, `TilingArtifacts`, and `process_list.csv` (`requested_backend` / `backend`
and `requested_mask_backend` / `mask_backend`). Requested values are provenance only. On
`resume`, compatibility compares the **resolved** slide backend and — when the slide has a
source mask — the **resolved** mask backend; artifacts are not rejected merely because a
requested value differs. Pre-#163 metadata and `process_list.csv` files (without the mask
backend fields/columns) are rejected clearly rather than loaded.

## Config areas

- `tiling.read_coordinates_from`
  - Reuse precomputed `{sample_id}.coordinates.*` artifacts instead of recomputing tile
    coordinates
  - Coordinate reuse is not completed-slide processing: outputs requested by the current run
    are still materialized. In particular, `save_tiles: true` writes the tile TAR and manifest,
    and an enabled tiling preview is rendered when at least one coordinate is present.
  - Disabled outputs are not created, and a reused zero-tile artifact does not create or report
    a tiling preview.
- `tiling.params`
  - spacing, tile size, overlap, tolerance, padding, and minimum tissue fraction
- `tiling.seg_params`
  - tissue segmentation settings
  - `method` selects `hsv`, `otsu`, `threshold`, or `sam2`
- `tiling.filter_params`
  - contour and optional white/black filtering settings
- `tiling.preview`
  - preview rendering settings
  - `save` enables both batch mask previews and tiling previews
  - `downsample` controls preview resolution
  - `tissue_contour_color` controls the RGB border color used for `preview/mask/*.jpg`
  - `mask_overlay_alpha` controls opacity for the filled annotation-mask overlay path; contour-only previews ignore it
- `tiling.masks`
  - multi-label annotation sampling (pixel_mapping, color_mapping, min_coverage, output_mode)
  - annotation sampling activates when `pixel_mapping` declares a foreground class other than
    the default `tissue`; otherwise the CLI runs binary tissue tiling. Each slide's annotation
    mask is taken from the `mask_path` column of the input CSV.
  - `pixel_mapping` is your own label vocabulary: it must enumerate **every** label value
    present in the raster (each value distinct, in `[0, 255]`, regardless of the raster's
    integer storage width). The annotation name `merged` is reserved for structural merged
    coordinate output; `tissue` remains a valid conventional annotation. Any pixel value not
    declared here makes the mask read fail (the discreteness guard). If the raster reserves a
    value for unannotated pixels, declare it like any other class and simply give it no
    `min_coverage` threshold.
  - `min_coverage` selects **which** classes are actually sampled: only classes given a
    (non-null) coverage threshold get tiled, and the coverage report's `frac`/`est_tiles`
    are computed relative to those classes. To sample a subset (e.g. only Gleason grades 4
    and 5 from a 6-grade mask), list all grades in `pixel_mapping` so the raster validates,
    but give thresholds only to grades 4 and 5. Because configs are deep-merged over the
    default `{background: 0, tissue: 1}`, set `min_coverage.tissue: null` to drop the default
    tissue class from sampling, and set `pixel_mapping.tissue: null` to remove the default
    label entirely (required to reuse its value, e.g. a `tumor: 1` mask).
  - `output_mode` (annotation sampling only): `per_annotation` (default) writes one coordinate
    artifact per sampled class; `merged` writes one flat merged per-slide artifact (the
    union of tiles passing any class threshold), identified structurally rather than as an
    annotation named `merged`.
  - `tiling.independent_sampling` chooses `independent_sampling` (tile each class separately)
    vs the default joint sampling (one pass over the union mask, then per-class coverage
    filtering).
  - Not yet supported with annotation sampling: `resume`, `read_coordinates_from`, and
    `save_tiles` raise a clear error if enabled; previews are skipped.
- `save_tiles`
  - write `tiles/{sample_id}.tiles.tar`
- `speed.num_workers`
  - slide-level batch parallelism
- `speed.jpeg_backend`
  - `pil` (default) uses the core Pillow dependency and works in a base install
  - `turbojpeg` opts into the faster PyTurboJPEG encoder and requires
    `pip install "hs2p[turbojpeg]"`
  - explicit selections never fall back; an unavailable TurboJPEG dependency is
    reported before slide tile extraction begins

## Progress reporting

When stdout is interactive, the entrypoint uses `rich` live progress:

- shows discovered tile totals during the run
- reports `empty_masks` while resolving precomputed tissue masks and in final summaries
- finishes with a summary panel including output locations and `process_list.csv`

When stdout is non-interactive, `hs2p` falls back to concise plain-text progress and summary logs.

Detailed logs still go to `output_dir/logs/log.txt`.

## Partial batch failures

The CLI attempts every requested slide and persists every outcome to
`process_list.csv`. If any slide failed, it reports a completed run with failed
slides and exits non-zero only after the manifest and tracebacks have been
written. It exits zero when every slide succeeded.

This differs from the Python `tile_slides()` contract: Python callers receive
the successful artifacts and one aggregate `BatchPartialFailureWarning` instead
of a non-zero process exit.

## Resume and precomputed artifacts

- `resume: true` treats a compatible successful `process_list.csv` row as completed slide
  processing and expects the current process-list schema
- reused artifacts are validated against structured metadata, not `config_hash`
- `tiling.read_coordinates_from` reuses only compatible tile coordinates; downstream outputs
  enabled for the current run are still produced

## Performance notes

### Segmentation downsample

`tiling.seg_params.downsample` controls the resolution used for tissue segmentation:

- larger values are faster and coarser
- smaller values improve edge precision but cost more time and memory

`tiling.seg_params.method` controls how the segmentation mask is generated at that level:

- `hsv` uses the existing HSV heuristic
- `otsu` thresholds the saturation channel with Otsu
- `threshold` applies a fixed saturation threshold
- `sam2` runs SAM2 inference on an internal fixed `8.0 um/px` thumbnail
  - hs2p chooses the thumbnail level in physical units first, then resizes to the requested spacing only if the nearest pyramid level is outside tolerance
  - if `sam2_checkpoint_path` is empty, hs2p downloads the default AtlasPatch checkpoint from Hugging Face
  - if `sam2_config_path` is empty, hs2p downloads the default AtlasPatch SAM2 config from Hugging Face
  - `tiling.seg_params.downsample` is ignored by SAM2
  - `sam2_num_workers` caps concurrent SAM2 mask-resolution workers; set it to `1` to serialize GPU inference and avoid CUDA OOMs

### Tile pixel QC

`tiling.filter_params.filter_white`, `filter_black`, `filter_grayspace`, and `filter_blur` are disabled by default.

When enabled, hs2p evaluates candidate tiles at `tiling.filter_params.qc_spacing_um`, which is typically coarser than the final extraction spacing. This is still slower than mask-only tiling, but cheaper than running pixel QC at the requested tile spacing.

### Tile tar export

When `save_tiles: true`, hs2p also writes `tiles/{sample_id}.tiles.tar`.

- non-CuCIM paths coalesce dense tile regions before slicing them back into tiles
- CuCIM paths use batched reads
- `gpu_decode=True` is opt-in in the Python API for CuCIM tar export

## Outputs

See [artifacts.md](artifacts.md) for the exact coordinate artifact schema and process-list columns.
