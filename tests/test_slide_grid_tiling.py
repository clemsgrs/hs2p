"""Slide-grid tiling: one fixed lattice over the whole slide, anchored at its top-left corner.

A flat 512 px PNG slide at 0.5 um/px is tiled with 64 px tiles at its native spacing on the
CPU ``pil`` reader, against a binary PNG tissue mask holding two separate rectangular blobs:

- blob A covers x in [40, 150), y in [30, 140);
- blob B covers x in [300, 420), y in [330, 450).

Neither blob starts on a multiple of 64, so the contour grid (anchored at each blob's
bounding box) and the slide grid (anchored at the slide's corner) place different tiles.
Every expected coverage below is worked out by hand from those rectangles.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import hs2p.__main__ as cli_mod
from hs2p.api import (
    DEFAULT_MASK_SAMPLING,
    BatchPartialFailureWarning,
    FilterConfig,
    MaskSamplingConfig,
    PreviewConfig,
    SlideSpec,
    TilingConfig,
    load_tiling_result,
    tile_slide,
    tile_slides,
)
from hs2p.configs.resolvers import resolve_tiling_config
from hs2p.preprocessing import detect_contours, generate_tiles

SPACING_UM = 0.5
TILE_PX = 64
SLIDE_PX = 512
BLOB_A = (slice(30, 140), slice(40, 150))
BLOB_B = (slice(330, 450), slice(300, 420))


def _write_two_blob_slide(tmp_path: Path) -> SlideSpec:
    pixels = np.full((SLIDE_PX, SLIDE_PX, 3), 255, dtype=np.uint8)
    labels = np.zeros((SLIDE_PX, SLIDE_PX), dtype=np.uint8)
    for blob in (BLOB_A, BLOB_B):
        pixels[blob] = (200, 60, 120)
        labels[blob] = 1
    image_path = tmp_path / "slide.png"
    Image.fromarray(pixels).save(image_path)
    mask_path = tmp_path / "tissue.png"
    Image.fromarray(labels, mode="L").save(mask_path)
    return SlideSpec(
        sample_id="slide",
        image_path=image_path,
        mask_path=mask_path,
        spacing_at_level_0=SPACING_UM,
    )


def _tiling(**overrides) -> TilingConfig:
    return TilingConfig(
        requested_spacing_um=SPACING_UM,
        requested_tile_size_px=TILE_PX,
        tolerance=0.05,
        overlap=overrides.pop("overlap", 0.0),
        backend="pil",
        mask_backend="pil",
        **overrides,
    )


def _masks(min_coverage: float):
    return replace(DEFAULT_MASK_SAMPLING, min_coverage={"tissue": min_coverage})


def _origins(result) -> list[tuple[int, int]]:
    return sorted(zip(result.x.tolist(), result.y.tolist()))


def test_the_default_contour_grid_keeps_anchoring_each_blob_at_its_bounding_box(tmp_path):
    result = tile_slide(
        _write_two_blob_slide(tmp_path), tiling=_tiling(), masks=_masks(0.5)
    )

    assert _origins(result) == [
        (40, 30), (40, 94), (104, 30), (104, 94),
        (300, 330), (300, 394), (364, 330), (364, 394),
    ]


def test_the_slide_grid_puts_the_tiles_of_both_blobs_on_one_lattice(tmp_path):
    result = tile_slide(
        _write_two_blob_slide(tmp_path), tiling=_tiling(grid="slide"), masks=_masks(0.5)
    )

    origins = _origins(result)
    assert result.step_px_lv0 == TILE_PX
    assert all(x % TILE_PX == 0 and y % TILE_PX == 0 for x, y in origins)
    assert len(set(origins)) == len(origins)
    # Blob A covers 0.53 of the cell at (64, 0) and all of (64, 64); blob B all of
    # (320, 384), 0.84 of (320, 320) and 0.56 of (384, 384). No other cell reaches 0.5.
    assert origins == [(64, 0), (64, 64), (320, 320), (320, 384), (384, 384)]


def test_overlap_shrinks_the_slide_grid_stride_and_origins_follow_the_new_lattice(
    tmp_path,
):
    result = tile_slide(
        _write_two_blob_slide(tmp_path),
        tiling=_tiling(grid="slide", overlap=0.5),
        masks=_masks(0.5),
    )

    origins = _origins(result)
    half_tile = TILE_PX // 2
    assert result.step_px_lv0 == half_tile
    assert all(x % half_tile == 0 and y % half_tile == 0 for x, y in origins)
    assert len(set(origins)) == len(origins)
    # Blob A alone keeps ten tiles at stride 32, among them origins off the 64 lattice.
    assert [(x, y) for x, y in origins if x < 256] == [
        (32, 32), (32, 64), (32, 96),
        (64, 0), (64, 32), (64, 64), (64, 96),
        (96, 32), (96, 64), (96, 96),
    ]


# Blob A covers rows [30, 64) of the 64 px cell at (64, 0): exactly 34 / 64 of the tile.
# Coverage is measured in float32, so "just above" is the next float32 up.
EDGE_CELL = (64, 0)
EDGE_COVERAGE = 34 / 64
JUST_ABOVE = float(np.nextafter(np.float32(EDGE_COVERAGE), np.float32(1.0)))


@pytest.mark.parametrize(
    ("min_coverage", "kept"),
    [
        pytest.param(EDGE_COVERAGE, True, id="at-threshold"),
        pytest.param(JUST_ABOVE, False, id="just-above"),
    ],
)
def test_a_slide_grid_tile_is_kept_exactly_when_its_coverage_reaches_the_threshold(
    tmp_path, min_coverage, kept
):
    result = tile_slide(
        _write_two_blob_slide(tmp_path),
        tiling=_tiling(grid="slide"),
        masks=_masks(min_coverage),
    )

    coverage = {
        (x, y): fraction
        for x, y, fraction in zip(
            result.x.tolist(), result.y.tolist(), result.tissue_fractions.tolist()
        )
    }
    assert (EDGE_CELL in coverage) is kept
    if kept:
        assert coverage[EDGE_CELL] == EDGE_COVERAGE


SLIDE_GRID_ORIGINS = [(64, 0), (64, 64), (320, 320), (320, 384), (384, 384)]


@pytest.mark.parametrize("grid", ["contour", "slide"])
def test_a_tiling_result_and_its_saved_artifact_record_the_grid_used(tmp_path, grid):
    whole_slide = _write_two_blob_slide(tmp_path)

    in_memory = tile_slide(whole_slide, tiling=_tiling(grid=grid), masks=_masks(0.5))
    (artifact,) = tile_slides(
        [whole_slide],
        tiling=_tiling(grid=grid),
        masks=_masks(0.5),
        output_dir=tmp_path / "out",
    )

    assert in_memory.grid == grid
    meta = json.loads(artifact.coordinates_meta_path.read_text())
    assert meta["tiling"]["grid"] == grid
    loaded = load_tiling_result(
        artifact.coordinates_npz_path, artifact.coordinates_meta_path
    )
    assert loaded.grid == grid
    assert _origins(loaded) == _origins(in_memory)


def _rewrite_tiling_meta(meta_path: Path, edit) -> None:
    meta = json.loads(meta_path.read_text())
    edit(meta["tiling"])
    meta_path.write_text(json.dumps(meta, indent=2))


def test_an_artifact_written_before_grids_existed_loads_as_a_contour_grid_artifact(
    tmp_path,
):
    """Earlier versions only had the contour grid and wrote no ``grid`` key (soma keeps a
    real hs2p 4.4.3 cache that must still load)."""
    (artifact,) = tile_slides(
        [_write_two_blob_slide(tmp_path)],
        tiling=_tiling(),
        masks=_masks(0.5),
        output_dir=tmp_path / "out",
    )
    _rewrite_tiling_meta(artifact.coordinates_meta_path, lambda tiling: tiling.pop("grid"))

    loaded = load_tiling_result(
        artifact.coordinates_npz_path, artifact.coordinates_meta_path
    )

    assert loaded.grid == "contour"


def test_an_artifact_naming_an_unknown_grid_is_rejected(tmp_path):
    (artifact,) = tile_slides(
        [_write_two_blob_slide(tmp_path)],
        tiling=_tiling(grid="slide"),
        masks=_masks(0.5),
        output_dir=tmp_path / "out",
    )
    _rewrite_tiling_meta(
        artifact.coordinates_meta_path, lambda tiling: tiling.update(grid="hexagonal")
    )

    with pytest.raises(ValueError, match="grid must be one of"):
        load_tiling_result(artifact.coordinates_npz_path, artifact.coordinates_meta_path)


def _tile_into(whole_slide: SlideSpec, output_dir: Path, *, grid: str, **kwargs):
    return tile_slides(
        [whole_slide],
        tiling=_tiling(grid=grid),
        masks=_masks(0.5),
        output_dir=output_dir,
        **kwargs,
    )


def _process_row(output_dir: Path) -> dict:
    (row,) = pd.read_csv(
        output_dir / "process_list.csv", converters={"sample_id": str}
    ).to_dict(orient="records")
    return row


@pytest.mark.parametrize(
    ("written", "requested"), [("slide", "contour"), ("contour", "slide")]
)
@pytest.mark.parametrize("reuse", ["read_coordinates_from", "resume"])
def test_coordinates_tiled_on_one_grid_are_never_reused_for_the_other(
    tmp_path, written, requested, reuse
):
    whole_slide = _write_two_blob_slide(tmp_path)
    first = tmp_path / "first"
    _tile_into(whole_slide, first, grid=written)
    if reuse == "resume":
        output_dir, kwargs = first, {"resume": True}
    else:
        output_dir, kwargs = tmp_path / "second", {"read_coordinates_from": first / "tiles"}

    with pytest.warns(BatchPartialFailureWarning, match="grid mismatch"):
        _tile_into(whole_slide, output_dir, grid=requested, **kwargs)

    row = _process_row(output_dir)
    assert row["tiling_status"] == "failed"
    assert row["error"] == "precomputed tiles grid mismatch"


def test_slide_grid_coordinates_are_reused_by_a_slide_grid_request(tmp_path):
    whole_slide = _write_two_blob_slide(tmp_path)
    first = tmp_path / "first"
    _tile_into(whole_slide, first, grid="slide")

    (reused,) = _tile_into(
        whole_slide, tmp_path / "second", grid="slide",
        read_coordinates_from=first / "tiles",
    )

    assert _process_row(tmp_path / "second")["tiling_status"] == "success"
    loaded = load_tiling_result(reused.coordinates_npz_path, reused.coordinates_meta_path)
    assert _origins(loaded) == SLIDE_GRID_ORIGINS


def test_a_slide_grid_edge_tile_extends_past_the_slide_and_counts_its_whole_area():
    """A 100 px slide tiled with 32 px tiles: the last lattice origin, 96, lies inside the
    slide and its tile runs 28 px past the edge. A 4 x 32 px tissue strip along the edge
    covers 128 / 1024 of that whole tile, not all of the part inside the slide."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:32, 96:100] = 255
    contours = detect_contours(mask, slide_dimensions=(100, 100), a_t=0)

    result = generate_tiles(
        (100, 100), contours, requested_tile_size_px=32,
        requested_spacing_um=1.0, base_spacing_um=1.0,
        level_downsamples=[1.0], min_tissue_fraction=0.1, grid="slide",
    )

    assert _origins(result) == [(96, 0)]
    assert result.tissue_fractions.tolist() == [0.125]


@pytest.mark.parametrize("num_workers", [1, 2])
def test_slide_grid_coverage_work_scales_with_tissue_not_with_the_canvas(
    monkeypatch, num_workers
):
    """Two 5 px specks on a 1024 px canvas tiled with 8 px tiles: only the lattice cells
    around each speck are measured, never the whole slide."""
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[203:208, 101:106] = 255
    mask[600:605, 800:805] = 255
    contours = detect_contours(mask, slide_dimensions=(1024, 1024), a_t=0)
    integral_sizes = []
    original_integral = cv2.integral

    def track_integral(image, *args, **kwargs):
        integral_sizes.append(image.size)
        return original_integral(image, *args, **kwargs)

    monkeypatch.setattr(cv2, "integral", track_integral)
    result = generate_tiles(
        (1024, 1024), contours, requested_tile_size_px=8,
        requested_spacing_um=1.0, base_spacing_um=1.0,
        level_downsamples=[1.0], min_tissue_fraction=0.1,
        num_workers=num_workers, grid="slide",
    )

    # 15 / 64 and 10 / 64 of the two cells the first speck straddles, 25 / 64 of the cell
    # holding the second.
    assert _origins(result) == [(96, 200), (104, 200), (800, 600)]
    # At most a 2 x 2 block of tiles around each speck.
    assert sum(integral_sizes) <= 2 * (2 * 8) ** 2


# ------------------------------------------------------ filtering and previews

# An 8 px tissue speck alone in the cell at (448, 0): 64 / 4096 of the tile, far below the
# default contour area filter (a_t=4 reference tiles of 16 px) and almost all white.
SPECK = (slice(16, 24), slice(480, 488))
SPECK_CELL = (448, 0)


def _write_speck_slide(tmp_path: Path) -> SlideSpec:
    whole_slide = _write_two_blob_slide(tmp_path)
    for path in (whole_slide.image_path, whole_slide.mask_path):
        with Image.open(path) as image:
            pixels = np.array(image)
        pixels[SPECK] = (200, 60, 120) if pixels.ndim == 3 else 1
        Image.fromarray(pixels).save(path)
    return whole_slide


@pytest.mark.parametrize(
    ("filtering", "kept"),
    [
        pytest.param(FilterConfig(), False, id="contour-area-filter"),
        pytest.param(FilterConfig(a_t=0), True, id="no-area-filter"),
        pytest.param(FilterConfig(a_t=0, filter_white=True), False, id="white-qc"),
    ],
)
def test_the_slide_grid_applies_contour_and_pixel_filtering(tmp_path, filtering, kept):
    result = tile_slide(
        _write_speck_slide(tmp_path),
        tiling=_tiling(grid="slide"),
        filtering=filtering,
        masks=_masks(0.01),
    )

    assert (SPECK_CELL in _origins(result)) is kept
    assert (64, 64) in _origins(result)


def test_tile_slides_writes_mask_and_tiling_previews_for_the_slide_grid(tmp_path):
    (artifact,) = _tile_into(
        _write_two_blob_slide(tmp_path),
        tmp_path / "out",
        grid="slide",
        preview=PreviewConfig(
            save_mask_preview=True, save_tiling_preview=True, downsample=1
        ),
    )

    for preview_path in (artifact.mask_preview_path, artifact.tiling_preview_path):
        assert preview_path is not None
        with Image.open(preview_path) as preview:
            assert preview.size == (SLIDE_PX, SLIDE_PX)


# ---------------------------------------------------------- annotation sampling


def _write_annotation_slide(tmp_path: Path) -> SlideSpec:
    """The two-blob slide with a half-resolution label mask: tumor is blob A, stroma is
    blob B."""
    whole_slide = _write_two_blob_slide(tmp_path)
    labels = np.zeros((SLIDE_PX // 2, SLIDE_PX // 2), dtype=np.uint8)
    for (rows, cols), value in ((BLOB_A, 2), (BLOB_B, 3)):
        halved = (slice(rows.start // 2, rows.stop // 2), slice(cols.start // 2, cols.stop // 2))
        labels[halved] = value
    mask_path = tmp_path / "labels.png"
    Image.fromarray(labels, mode="L").save(mask_path)
    return replace(whole_slide, mask_path=mask_path)


@pytest.mark.parametrize("strategy", ["joint", "independent"])
def test_annotation_sampling_on_the_slide_grid_keeps_each_label_on_the_lattice(
    tmp_path, strategy
):
    results = tile_slide(
        _write_annotation_slide(tmp_path),
        tiling=_tiling(grid="slide"),
        masks=MaskSamplingConfig(
            pixel_mapping={"background": 0, "tumor": 2, "stroma": 3},
            min_coverage={"tumor": 0.5, "stroma": 0.5},
            strategy=strategy,
        ),
    )

    assert {label: result.grid for label, result in results.items()} == {
        "tumor": "slide",
        "stroma": "slide",
    }
    assert _origins(results["tumor"]) == [(64, 0), (64, 64)]
    assert _origins(results["stroma"]) == [(320, 320), (320, 384), (384, 384)]


# ------------------------------------------------------------------ configuration


def _cli_tiling(monkeypatch, tmp_path: Path, config_body: str) -> TilingConfig:
    """Run the real CLI entrypoint on ``config_body`` and return the tiling config it
    hands to ``tile_slides``."""
    received = {}

    def _record(whole_slides, **kwargs):
        received.update(kwargs)
        (kwargs["output_dir"] / "process_list.csv").write_text("")
        return []

    monkeypatch.setattr(cli_mod, "tile_slides", _record)
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text("sample_id,image_path,mask_path\nslide-1,slide-1.svs,\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"csv: {csv_path}\n" + config_body)
    cli_mod.entrypoint(
        [
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-datetime",
            "--skip-logging",
        ]
    )
    return received["tiling"]


@pytest.mark.parametrize(
    ("config_body", "grid"),
    [
        pytest.param("", "contour", id="default"),
        pytest.param("tiling:\n  params:\n    grid: slide\n", "slide", id="slide"),
    ],
)
def test_the_config_file_selects_the_grid(monkeypatch, tmp_path, config_body, grid):
    assert _cli_tiling(monkeypatch, tmp_path, config_body).grid == grid


def test_the_config_file_rejects_an_unknown_grid(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as excinfo:
        _cli_tiling(monkeypatch, tmp_path, "tiling:\n  params:\n    grid: hexagonal\n")

    assert (
        "Config value tiling.params.grid must be one of 'contour', 'slide', "
        "got str 'hexagonal'"
    ) in str(excinfo.value)


def test_a_programmatic_adapter_without_a_grid_tiles_on_the_contour_grid():
    """Shaped like slide2vec's adapter, which predates the grid option."""
    cfg = SimpleNamespace(
        tiling=SimpleNamespace(
            params=SimpleNamespace(
                requested_spacing_um=0.5,
                requested_tile_size_px=224,
                tolerance=0.05,
                overlap=0.0,
            ),
            backend="auto",
            mask_backend="auto",
        )
    )

    assert resolve_tiling_config(cfg).grid == "contour"
