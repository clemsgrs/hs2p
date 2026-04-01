# Python API

The Python API is the best entrypoint when you want to integrate `hs2p` into your own pipeline instead of driving it through the CLI. The current public surface is split into:

- high-level orchestration in `hs2p.api`
- the canonical in-memory result model in `hs2p.preprocessing`

## Main public types

- `SlideSpec`
  - Identifies one slide via `sample_id`, `image_path`, and optional `mask_path`
  - `SlideSpec` stays generic because it is shared across tiling and sampling internals
  - `spacing_at_level_0` can override broken or missing slide metadata
- `TilingConfig`
  - Requested backend, spacing, tile size, overlap, padding, and minimum tissue fraction
- `SegmentationConfig`
  - Tissue-segmentation settings used before coordinate extraction
- `FilterConfig`
  - Contour filtering plus optional coarse tile QC for white-space, black-space, grayspace, and blur
- `PreviewConfig`
  - Batch preview toggles, preview downsample, and tissue-overlay styling for mask previews
- `TilingResult`
  - Canonical in-memory result model from `hs2p.preprocessing`
- `TileGeometry`
  - Canonical geometry container with `coordinates`, `tissue_fractions`, and `tile_index`
- `TilingArtifacts`
  - Lightweight record of saved artifact paths and optional preview/tar outputs

## Canonical result contract

`TilingResult` is the only supported tiling-result model. Downstream code should use:

- `coordinates`
- `tissue_fractions`
- `tile_index`
- `requested_tile_size_px`
- `requested_spacing_um`
- `effective_tile_size_px`
- `effective_spacing_um`
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
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
        use_padding=True,
    ),
    segmentation=SegmentationConfig(downsample=64),
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
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=0.1,
        use_padding=True,
    ),
    segmentation=SegmentationConfig(downsample=64),
    filtering=FilterConfig(ref_tile_size=224, a_t=4, a_h=2),
    preview=PreviewConfig(
        save_mask_preview=True,
        save_tiling_preview=True,
        downsample=32,
        mask_overlay_color=(157, 219, 129),
        mask_overlay_alpha=0.5,
    ),
    output_dir=Path("output"),
    num_workers=4,
)
```

When `save_mask_preview=True`, `tile_slides()` writes `preview/mask/{sample_id}.jpg`
as a slide preview with the binary tissue mask overlaid on top. `mask_overlay_color`
and `mask_overlay_alpha` control that overlay style.

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

The lower-level WSI helpers use explicit public mask names:

- `extract_coordinates(..., tissue_mask_path=...)`
- `sample_coordinates(..., annotation_mask_path=...)`
- `filter_coordinates(..., annotation_mask_path=...)`

Internally, the shared coordinate engine still uses a generic `mask_path`.

## Backend selection

`TilingConfig.backend` supports:

- `auto`
- `cucim`
- `vips`
- `openslide`
- `asap`

`auto` prefers `cucim -> vips -> openslide -> asap`.
