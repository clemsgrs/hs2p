# Python API

The Python API is the primary public interface for programmatic use. It supports extraction at any requested spacing, whether or not that spacing exists natively in the slide pyramid.

For convenience, the config dataclasses keep the main knobs explicit and fill the secondary ones from the packaged defaults in `hs2p/configs/default.yaml`.

## Main types

- `WholeSlide`
  - Identifies one sample through `sample_id`, `image_path`, and optional `mask_path`
  - If `mask_path` is omitted, HS2P can segment tissue directly from the slide
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

result = tile_slide(
    WholeSlide(
        sample_id="slide-1",
        image_path=Path("/data/slide-1.tif"),
        mask_path=Path("/data/slide-1-mask.tif"),
    ),
    tiling=TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
        backend="asap",
    ),
    segmentation=SegmentationConfig(downsample=64),
    filtering=FilterConfig(ref_tile_size=224, a_t=4, a_h=2),
    num_workers=1,
)

artifacts = save_tiling_result(result, output_dir=Path("output"))
```

## Notes

- `tile_slide()` is the right entry point when you want to stay fully in Python.
- `tile_slides()` is the right entry point when you want batch processing, output manifests, resume support, or preview images.
- If you want to create masks ahead of time instead of segmenting inside the tiling run, see [tissue-mask-generation.md](tissue-mask-generation.md).
- Saved artifacts are intentionally explicit and named by `sample_id`; see [artifacts.md](artifacts.md).
