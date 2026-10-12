"""Previews reopen a flat slide with the spacing override preprocessing used (F4)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from hs2p.api import (
    DEFAULT_MASK_SAMPLING,
    FilterConfig,
    MaskSamplingConfig,
    PreviewConfig,
    SlideSpec,
    TilingConfig,
    tile_slides,
)

EXPECTED_ORIGINS = [(0, 0), (0, 64), (64, 0), (64, 64)]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """A 256x256 RGB PNG slide and a same-size label PNG with label 1 in the top-left
    128x128 quadrant; at 0.5 um/px and 64 px tiles that is exactly four full tiles."""
    slide_path = tmp_path / "slide.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(90, 200, size=(256, 256, 3), dtype=np.uint8)).save(slide_path)
    labels = np.zeros((256, 256), dtype=np.uint8)
    labels[:128, :128] = 1
    mask_path = tmp_path / "mask.png"
    Image.fromarray(labels, mode="L").save(mask_path)
    return slide_path, mask_path


def _tiling_config() -> TilingConfig:
    return TilingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=64,
        tolerance=0.05,
        overlap=0.0,
        backend="pil",
        mask_backend="pil",
    )


_MASKS = replace(DEFAULT_MASK_SAMPLING, min_coverage={"tissue": 0.5})


def _assert_success_row(output_dir: Path, sample_id: str) -> dict:
    process_df = pd.read_csv(output_dir / "process_list.csv", converters={"sample_id": str})
    rows = process_df.to_dict(orient="records")
    assert [row["sample_id"] for row in rows] == [sample_id]
    assert rows[0]["tiling_status"] == "success", rows[0].get("error")
    return rows[0]


def _assert_readable_jpeg(path: Path) -> None:
    assert path.is_file(), path
    with Image.open(path) as image:
        image.load()
        assert image.size[0] > 0 and image.size[1] > 0


@pytest.mark.parametrize(
    ("save_mask_preview", "save_tiling_preview"),
    [(True, False), (False, True), (True, True)],
)
def test_tissue_previews_use_the_supplied_spacing_on_a_flat_slide(
    tmp_path: Path, save_mask_preview: bool, save_tiling_preview: bool
):
    slide_path, mask_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"

    artifacts = tile_slides(
        [
            SlideSpec(
                sample_id="flat", image_path=slide_path, mask_path=mask_path, spacing_at_level_0=0.5
            )
        ],
        tiling=_tiling_config(),
        masks=_MASKS,
        filtering=FilterConfig(a_t=0, a_h=0),
        preview=PreviewConfig(
            save_mask_preview=save_mask_preview, save_tiling_preview=save_tiling_preview
        ),
        output_dir=output_dir,
    )

    assert len(artifacts) == 1
    coordinates = np.load(artifacts[0].coordinates_npz_path)
    assert sorted(zip(coordinates["x"].tolist(), coordinates["y"].tolist())) == EXPECTED_ORIGINS
    row = _assert_success_row(output_dir, "flat")
    if save_mask_preview:
        _assert_readable_jpeg(output_dir / "preview" / "mask" / "flat.jpg")
        assert Path(row["mask_preview_path"]) == output_dir / "preview" / "mask" / "flat.jpg"
    if save_tiling_preview:
        _assert_readable_jpeg(output_dir / "preview" / "tiling" / "flat.jpg")
        assert Path(row["tiling_preview_path"]) == output_dir / "preview" / "tiling" / "flat.jpg"


def test_previews_from_reused_coordinates_use_the_recorded_spacing(tmp_path: Path):
    slide_path, mask_path = _write_inputs(tmp_path)
    spec = SlideSpec(
        sample_id="flat", image_path=slide_path, mask_path=mask_path, spacing_at_level_0=0.5
    )
    first_dir = tmp_path / "first"
    tile_slides(
        [spec],
        tiling=_tiling_config(),
        masks=_MASKS,
        filtering=FilterConfig(a_t=0, a_h=0),
        preview=PreviewConfig(save_mask_preview=False, save_tiling_preview=False),
        output_dir=first_dir,
    )

    second_dir = tmp_path / "second"
    tile_slides(
        [spec],
        tiling=_tiling_config(),
        masks=_MASKS,
        filtering=FilterConfig(a_t=0, a_h=0),
        preview=PreviewConfig(save_mask_preview=False, save_tiling_preview=True),
        output_dir=second_dir,
        read_coordinates_from=first_dir / "coordinates",
    )

    _assert_success_row(second_dir, "flat")
    _assert_readable_jpeg(second_dir / "preview" / "tiling" / "flat.jpg")


def test_sampling_previews_use_the_supplied_spacing_on_a_flat_slide(tmp_path: Path):
    slide_path, mask_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"
    masks = MaskSamplingConfig(
        pixel_mapping={"background": 0, "tumor": 1},
        colors={"background": None, "tumor": [255, 0, 0]},
        min_coverage={"tumor": 0.5},
        output_mode="per_annotation",
        strategy="joint",
    )

    artifacts = tile_slides(
        [
            SlideSpec(
                sample_id="flat", image_path=slide_path, mask_path=mask_path, spacing_at_level_0=0.5
            )
        ],
        tiling=_tiling_config(),
        filtering=FilterConfig(a_t=0, a_h=0),
        preview=PreviewConfig(save_mask_preview=True, save_tiling_preview=True),
        output_dir=output_dir,
        masks=masks,
    )

    assert len(artifacts) == 1
    coordinates = np.load(artifacts[0].coordinates_npz_path)
    assert sorted(zip(coordinates["x"].tolist(), coordinates["y"].tolist())) == EXPECTED_ORIGINS
    _assert_success_row(output_dir, "flat")
    _assert_readable_jpeg(output_dir / "preview" / "mask" / "flat.jpg")
    _assert_readable_jpeg(output_dir / "preview" / "tiling" / "tumor" / "flat.jpg")
