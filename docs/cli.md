# CLI guide

Run batch tissue tiling or annotation sampling with one command:

```bash
hs2p config.yaml [key=value ...]
```

The configuration selects the workflow. Tissue tiling uses an optional tissue `mask_path`; annotation sampling requires an annotation `mask_path` for every slide.

## First run

Install a reader using the [installation quick start](../README.md#installation) or the [backend options](#backends) below. Create `slides.csv`:

```csv
sample_id,image_path,mask_path
slide-1,/data/slide-1.tif,/data/slide-1-tissue-mask.tif
slide-2,/data/slide-2.tif,
```

Save `config.yaml`:

```yaml
csv: slides.csv
output_dir: output
tiling:
  backend: openslide
  params:
    requested_spacing_um: 0.5
    requested_tile_size_px: 224
```

Omitted settings inherit the [default config](../hs2p/configs/default.yaml). Override individual values on the command line:

```bash
hs2p config.yaml speed.num_workers=4
```

The CLI creates `output/<YYYY-MM-DD_HH_MM>/` with the effective `config.yaml`, artifacts, `process_list.csv`, and `logs/log.txt`. Use `--skip-datetime` to write directly into `output_dir`, or `--output-dir /path/to/output` to override the output root. Relative CSV and image paths are resolved from the working directory, so absolute paths are useful for shared configs.

## Input CSV

`sample_id` and `image_path` are required, and sample IDs must be unique. Omit `mask_path` or leave it blank to segment tissue on the fly. For annotation sampling, fill it with the label-raster path:

```csv
sample_id,image_path,mask_path
slide-1,/data/slide-1.tif,/data/slide-1-annotations.tif
slide-2,/data/slide-2.tif,/data/slide-2-annotations.tif
```

The old `tissue_mask_path` and `annotation_mask_path` columns are rejected; use `mask_path` for either workflow.

An optional `spacing_at_level_0` column overrides the slide's native spacing in microns per pixel:

```csv
sample_id,image_path,mask_path,spacing_at_level_0
slide-1,/data/slide-1.tif,,0.25
slide-2,/data/slide-2.tif,/data/slide-2-tissue-mask.tif,
```

The override must be finite and positive. It supplies missing metadata or replaces existing spacing; other pyramid spacings follow the backend's downsample factors. A disagreement with valid native metadata produces one warning identifying the slide, backend, and both values. Missing metadata and floating-point representation noise do not trigger that warning. Flat PNG/JPEG slides require this override because their density metadata is not interpreted as pathology spacing.

## Annotation sampling

The default label vocabulary is `{background: 0, tissue: 1}`, with only `tissue` selected. Changing the label vocabulary or sampled set activates annotation sampling when at least one class has a non-null coverage threshold. Changing only the tissue threshold keeps binary tissue tiling.

For a mask containing background `0`, tumor `1`, and stroma `2`, add this to the first-run config:

```yaml
tiling:
  masks:
    pixel_mapping:
      background: 0
      tissue: null
      tumor: 1
      stroma: 2
    colors:
      background: null
      tumor: [220, 60, 60]
      stroma: [60, 160, 220]
    min_coverage:
      tissue: null
      tumor: 0.5
      stroma: null
    output_mode: per_annotation
```

This selects tumor tiles with at least 50% coverage. Stroma is declared so the raster validates, but it is not sampled.

- `pixel_mapping` must declare every raster label, with distinct integer values in `[0, 255]`, even if the raster uses a wider integer type. Undeclared pixel values fail validation. Names become directory components: do not use path separators, `.` or `..`; `merged` is reserved for merged output.
- `min_coverage` selects classes through non-null thresholds. Coverage reports (`frac` and `est_tiles`) are relative to the selected classes. Declare unannotated pixels in `pixel_mapping` and leave their threshold null to exclude them.
- Configs are deep-merged with the defaults. `min_coverage.tissue: null` stops sampling tissue; `pixel_mapping.tissue: null` removes that label and its companion settings, allowing another class to use value `1`.
- `colors` must cover the remaining labels when supplied. Use RGB triplets or null to omit an overlay for a class; set `colors: null` to omit the mapping.

`output_mode: per_annotation` writes separate coordinate artifacts for selected classes. `output_mode: merged` writes their union as one per-slide artifact. By default, joint sampling builds candidates over the union mask and filters them by class coverage. Set `tiling.independent_sampling: true` to generate candidates separately for each class.

Annotation sampling supports filled mask previews and a tiling-grid preview for each non-empty coordinate output. `resume`, `tiling.read_coordinates_from`, and `save_tiles` are not supported for this workflow; enabling them raises an error. See the [artifact reference](artifacts.md) for output paths and the common batch manifest.

## Backends

`tiling.backend` selects the slide reader; `tiling.mask_backend` selects the source-mask reader. Both accept `auto`, `pil`, `cucim`, `vips`, `openslide`, or `asap`. Null and unknown values fail configuration validation.

| Reader | Install | Prerequisite |
| --- | --- | --- |
| `pil` | `pip install hs2p` | Supply `spacing_at_level_0` for flat slides. |
| `openslide` | `pip install "hs2p[openslide]"` | The extra includes OpenSlide's Python binding and binary package. |
| `vips` | `pip install "hs2p[vips]"` | libvips must also be available. |
| `asap` | `pip install "hs2p[asap]"` | Native ASAP with its Python bindings must also be installed. |
| `cucim` | `pip install "hs2p[cucim]"` | The extra supplies the CUDA 12 cuCIM stack. |

`pip install "hs2p[all]"` installs all reader extras and the optional TurboJPEG encoder; it does not install SAM2 or replace native system prerequisites.

With `auto`, each path is resolved independently:

- `.png`, `.jpg`, and `.jpeg` suffixes (case-insensitive) select PIL only. Corrupt, unsupported, or oversized flat rasters fail without trying another reader.
- Other inputs probe `cucim → vips → openslide → asap` and stop at the first reader that opens the file. PIL is not part of this chain.

Selection does not inspect mask labels or retry after a later decode failure. If a native reader opens a mask but cannot decode its pixels, explicitly set `tiling.mask_backend` to a reader that can decode it. Explicit reader choices are authoritative. Missing or incompatible mask backends fail with the mask path and backend in the error; a slide without a source mask does not check mask-reader availability.

Requested and resolved readers are saved separately as provenance. Resume compares resolved readers, including the mask reader when a source mask exists. Pin explicit readers to keep decoder choice stable across environments or backend-priority changes. See [artifact validation](artifacts.md#resume-and-validation) for compatibility rules, including older metadata without mask-backend fields.

### JPEG encoder

`speed.jpeg_backend` controls saved-tile encoding independently of the input readers. `pil` is the default and works with the base install. For TurboJPEG:

```bash
pip install "hs2p[turbojpeg]"
hs2p config.yaml save_tiles=true speed.jpeg_backend=turbojpeg
```

PyTurboJPEG also needs its native libjpeg-turbo library. An unavailable encoder is reported before tile extraction; explicit encoder selections do not fall back.

### SAM2 segmentation

SAM2 requires both the optional dependencies and the model package:

```bash
pip install "hs2p[sam2]"
pip install "git+https://github.com/facebookresearch/sam2.git"
```

Set `tiling.seg_params.method: sam2`. SAM2 uses an internal `8.0 µm/px` thumbnail and ignores `seg_params.downsample`. It selects a pyramid level by physical spacing and resizes only when that level falls outside tolerance.

If `sam2_checkpoint_path` or `sam2_config_path` is empty, hs2p downloads the corresponding default AtlasPatch asset from Hugging Face. Set those paths to use local files. `sam2_device` selects the inference device; `sam2_num_workers` caps concurrent mask-resolution workers. Use one worker to serialize GPU inference when memory is limited.

## Tiling settings and previews

The [default config](../hs2p/configs/default.yaml) lists all settings and defaults. The main controls are:

| Config area | Controls |
| --- | --- |
| `tiling.params` | Requested spacing, tile size, spacing tolerance, and overlap as a fraction. |
| `tiling.masks.min_coverage.tissue` | Minimum tissue fraction for binary tissue tiling. |
| `tiling.seg_params` | Tissue segmentation method and thumbnail resolution. |
| `tiling.filter_params` | Contour area filtering and optional tile-pixel QC. |
| `tiling.preview` | Separate `save_mask_preview` and `save_tiling_preview` toggles, plus styling. |
| `speed.num_workers` | Slide-level batch parallelism. |
| `save_tiles` | Export JPEG tiles to TAR with a CSV sidecar. |

For `hsv`, `otsu` (Otsu on saturation), and `threshold` (fixed saturation threshold), a larger `seg_params.downsample` selects a coarser segmentation thumbnail and reduces work. Smaller values give finer tissue boundaries at a higher time and memory cost.

Pixel QC is disabled by default. Enable `filter_white`, `filter_black`, `filter_grayspace`, or `filter_blur` under `tiling.filter_params` when needed. QC reads candidate tiles at `qc_spacing_um`, typically coarser than final extraction, and adds work beyond mask-only tiling.

`tiling.preview.downsample` controls preview resolution. Tissue-mask previews draw contours: `tissue_contour_color` sets the outer RGB border (default `#255E3B`), and holes use `#F26B3A`. `mask_overlay_alpha` affects filled annotation overlays only. Non-empty tissue results can also produce a tiling-grid preview.

Tile TAR export uses batched reads for cuCIM and coalesces dense regions on other readers. GPU decoding is opt-in through `gpu_decode=True` in the Python API. See [benchmarks](benchmark.md) for measured throughput and [artifacts](artifacts.md) for TAR paths and fields.

## Progress and failures

Interactive terminals show live tile totals, empty-mask counts, and a final summary with output locations. Redirected output uses plain-text progress and summaries. Detailed logs are written to the run directory's `logs/log.txt`.

The CLI attempts each slide and records its outcome in `process_list.csv`. Slide failures produce a nonzero exit status after the manifest and tracebacks are written; all-success runs exit zero. Python `tile_slides()` instead returns successful artifacts and emits one aggregate `BatchPartialFailureWarning`.

## Resume and coordinate reuse

To continue a dated run, keep `output_dir` as the parent directory and select its existing run name:

```bash
hs2p config.yaml resume=true resume_dirname=2026-09-08_10_30
```

For a run created with `--skip-datetime`, use `resume=true resume_dirname=.` to target `output_dir` itself.

`resume: true` treats compatible successful manifest rows as completed processing. It does not recreate a recorded TAR or preview if that file has since been deleted.

To reuse coordinates while producing outputs for the current run:

```bash
hs2p config.yaml tiling.read_coordinates_from=/data/previous-run/tiles save_tiles=true
```

Coordinate reuse skips coordinate computation and still writes requested TARs and tiling-grid previews; it does not regenerate mask previews. Disabled outputs are not created, and zero-tile artifacts do not produce tiling previews. Compatibility uses structured metadata, not `config_hash`; consult the [artifact reference](artifacts.md#resume-and-validation) before reusing results after changing inputs or settings.
