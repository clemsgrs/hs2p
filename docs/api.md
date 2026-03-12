# Python API

The Python API is the primary public interface for programmatic use.

## Main types

- `WholeSlide`
  - Identifies one sample through `sample_id`, `image_path`, and optional `mask_path`
- `TilingConfig`
  - Defines the requested tile spacing, tile size, overlap, tissue threshold, padding behavior, and backend
- `SegmentationConfig`
  - Controls tissue segmentation before coordinates are extracted
- `FilterConfig`
  - Controls contour-size filtering and optional white/black tile rejection
- `QCConfig`
  - Controls whether preview images are written in batch mode
- `TilingResult`
  - In-memory tile coordinates and metadata for one slide
- `TilingArtifacts`
  - Paths to the saved `.tiles.npz` and `.tiles.meta.json` outputs

## Main functions

- `tile_slide(...)`
  - Computes a `TilingResult` for one slide
  - This is the compute-only path and does not write preview images
- `tile_slides(...)`
  - Batch-processes multiple slides, writes artifacts, and can write previews
- `save_tiling_result(...)`
  - Writes a `TilingResult` to named artifact files
- `load_tiling_result(...)`
  - Reconstructs a `TilingResult` from saved artifacts

## Minimal example

```python
from pathlib import Path

from hs2p import (
    FilterConfig,
    SegmentationConfig,
    TilingConfig,
    WholeSlide,
    save_tiling_result,
    tile_slide,
)

tiling = TilingConfig(
    target_spacing_um=0.5,
    target_tile_size_px=224,
    tolerance=0.07,
    overlap=0.0,
    tissue_threshold=0.1,
    drop_holes=False,
    use_padding=True,
    backend="asap",
)

segmentation = SegmentationConfig(
    downsample=64,
    sthresh=8,
    sthresh_up=255,
    mthresh=7,
    close=4,
    use_otsu=False,
    use_hsv=True,
)

filtering = FilterConfig(
    ref_tile_size=224,
    a_t=4,
    a_h=2,
    max_n_holes=8,
    filter_white=False,
    filter_black=False,
    white_threshold=220,
    black_threshold=25,
    fraction_threshold=0.9,
)

result = tile_slide(
    WholeSlide(
        sample_id="slide-1",
        image_path=Path("/data/slide-1.tif"),
        mask_path=Path("/data/slide-1-mask.tif"),
    ),
    tiling=tiling,
    segmentation=segmentation,
    filtering=filtering,
    num_workers=1,
)

artifacts = save_tiling_result(result, output_dir=Path("output"))
```

## Notes

- `tile_slide()` is the right entry point when you want to stay fully in Python.
- `tile_slides()` is the right entry point when you want batch processing, output manifests, resume support, or preview images.
- Saved artifacts are intentionally explicit and named by `sample_id`; see [artifacts.md](artifacts.md).
