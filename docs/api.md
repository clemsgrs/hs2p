# Python API

The Python API is the primary interface when you want to call `hs2p` from your own code instead of driving it through the CLI. It supports:

- explicit slide objects through `SlideSpec`
- extraction at any requested spacing, whether or not that spacing exists natively in the slide pyramid
- direct in-memory results through `TilingResult`
- reusable persisted artifacts through `save_tiling_result()` / `load_tiling_result()`
- batch orchestration with optional previews through `tile_slides()`
- explicit preview helpers for single-slide workflows through `write_tiling_preview()` and `overlay_mask_on_slide()`

The config dataclasses keep the main knobs explicit and fill secondary options from the packaged defaults in `hs2p/configs/default.yaml`.

## Main types

- `SlideSpec`
  - Identifies one sample through `sample_id`, `image_path`, and optional `mask_path`
  - If `mask_path` is omitted, HS2P can segment tissue directly from the slide
- `TilingConfig`
  - Defines the requested backend, spacing, tile size, overlap, and tissue threshold
- `SegmentationConfig`
  - Controls tissue segmentation before coordinates are extracted
- `FilterConfig`
  - Controls contour and white/black filtering after segmentation
- `QCConfig`
  - Controls whether batch preview images are written and at what downsample
- `TilingResult`
  - In-memory tile coordinates plus read-level metadata for one slide
- `TilingArtifacts`
  - Paths to the saved `.tiles.npz` and `.tiles.meta.json` outputs
  - Can also carry preview-image paths when batch QC is enabled in `tile_slides()`

For field-by-field details, see the dataclass docstrings in [hs2p/api.py](../hs2p/api.py).

## Single-slide flow

Use `tile_slide()` when you want an in-memory result for one slide and you will decide yourself whether to persist it.

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
        mask_path=Path("/data/slide-1-mask.tif"),
    ),
    tiling=TilingConfig(
        backend="openslide",
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
    ),
    segmentation=SegmentationConfig(downsample=64),
    filtering=FilterConfig(ref_tile_size=224, a_t=4, a_h=2),
    num_workers=1,
)

artifacts = save_tiling_result(result, output_dir=Path("output"))
```

`result` is a [`TilingResult`](../hs2p/api.py#L144) for one slide. That object is what downstream code should use when it needs coordinates immediately in memory.

Typical downstream uses:

- extracting image patches from the WSI at the returned `x` / `y` coordinates
- joining model outputs back to tiles through `tile_index`
- persisting one tiling pass and reusing it across later training, inference, or QC jobs

`tile_slide()` is intentionally compute-only. Persist the coordinates with `save_tiling_result()`, then call `write_tiling_preview()` or `overlay_mask_on_slide()` if you want preview images in the same single-slide workflow.

## Preview helpers for single-slide workflows

When you are not using `tile_slides()`, the public preview helpers let you render the same kinds of QC assets explicitly:

- `write_tiling_preview(result=..., output_dir=..., downsample=...)`
  - Draws the returned tile grid on the slide overview and writes `visualization/tiling/{sample_id}.jpg`
  - Useful after `tile_slide()` when you want a quick visual check of coverage without switching to the batch API
- `overlay_mask_on_slide(...)`
  - Overlays a tissue or annotation mask on the slide overview
  - Useful when you want to inspect the mask itself, independently of the tiling grid
  - The default tissue preview used by `tile_slides(..., qc=QCConfig(save_mask_preview=True, ...))` hides background and overlays tissue in `[157, 219, 129]`

Example:

```python
from pathlib import Path

from hs2p import overlay_mask_on_slide, write_tiling_preview

tiling_preview_path = write_tiling_preview(
    result=result,
    output_dir=Path("output"),
    downsample=32,
)

mask_overlay = overlay_mask_on_slide(
    wsi_path=result.image_path,
    annotation_mask_path=Path("/data/slide-1-mask.tif"),
    downsample=32,
    backend=result.backend,
)
mask_overlay.save("output/visualization/mask/slide-1.jpg")
```

## Batch flow with previews

Use `tile_slides()` when you want batch processing, named outputs, `process_list.csv`, resume support, or preview images.

```python
from pathlib import Path

from hs2p import (
    FilterConfig,
    QCConfig,
    SegmentationConfig,
    SlideSpec,
    TilingConfig,
    tile_slides,
)

slides = [
    SlideSpec(
        sample_id="slide-1",
        image_path=Path("/data/slide-1.tif"),
        mask_path=Path("/data/slide-1-mask.tif"),
    ),
    SlideSpec(
        sample_id="slide-2",
        image_path=Path("/data/slide-2.tif"),
    ),
]

artifacts = tile_slides(
    slides,
    tiling=TilingConfig(
        backend="openslide",
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
    ),
    segmentation=SegmentationConfig(downsample=64),
    filtering=FilterConfig(ref_tile_size=224, a_t=4, a_h=2),
    qc=QCConfig(
        save_mask_preview=True,
        save_tiling_preview=True,
        downsample=32,
    ),
    output_dir=Path("output"),
    num_workers=4,
)
```

In this mode, `QCConfig` is useful because preview rendering is handled by `tile_slides()`, not by `tile_slide()`. Mask previews are rendered as tissue overlays rather than contour-and-hole line drawings.

## Results versus artifacts

The API intentionally separates compute results from persisted artifacts:

- `TilingResult`
  - In-memory object returned by `tile_slide()`
  - Best when your next step is immediate patch extraction or further Python-side processing
- `TilingArtifacts`
  - Lightweight record of the files written by `save_tiling_result()` or `tile_slides()`
  - Best when you want to pass around filenames, manifests, or cached tiling outputs

You can round-trip a saved result later:

```python
from hs2p import load_tiling_result

loaded = load_tiling_result(
    tiles_npz_path=artifacts.tiles_npz_path,
    tiles_meta_path=artifacts.tiles_meta_path,
)
```

## Choosing the right entry point

- Use `tile_slide()` for single-slide, in-memory use
- Use `save_tiling_result()` when you want to persist that result explicitly
- Use `load_tiling_result()` when a downstream stage should consume saved coordinates instead of recomputing them
- Use `tile_slides()` when you want batch output directories, manifests, preview images, or resume/precomputed-artifact workflows

If you want to create masks ahead of time instead of segmenting inside the tiling run, see [tissue-mask-generation.md](tissue-mask-generation.md). For the exact on-disk artifact format, see [artifacts.md](artifacts.md).
