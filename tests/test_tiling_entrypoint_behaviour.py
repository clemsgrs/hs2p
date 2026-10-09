"""Tissue tiling and annotation sampling behave the same from every entrypoint.

The public scalar entrypoints (``preprocess_slide``, ``build_tiling_result_from_mask``,
``preprocess_slide_per_annotation``, ``build_per_annotation_tiling_results``) and the
config-driven workflows (``tile_slide``, ``tile_slides``) are run on one flat 256 px PNG
slide at 0.5 um/px, tiled with 64 px tiles at its native spacing, so every expected origin
below is a hand-derived position on the level-0 pixel grid.

Slide pixels: textured tissue colours everywhere except one pure-white tile at (64, 0).

Tissue mask (0/1):
  * a 128 x 128 block at the origin (tiles (0,0), (64,0), (0,64), (64,64)) holding a
    16 x 16 hole in tile (0,0) -> coverage 0.9375 there, 1.0 elsewhere;
  * a 16 px wide strip right of it (x 128:144, y 0:128) -> tiles (128,0), (128,64) at 0.25;
  * a 4 px tall strip below it (x 0:128, y 128:132) -> tiles (0,128), (64,128) at 0.0625;
  * a separate 24 x 24 island at (200, 200) -> one tile at 0.140625, but its contour
    (area 529 mask px) is under the default ``a_t=4`` area filter (4 x 16 x 16 = 1024).

So the tissue gate picks: 0.5 -> 4 tiles, 0.1 -> 6 tiles, 0.01 -> 8 tiles.

Annotation mask (background 0, tumor 1, stroma 2):
  * tumor: the 128 x 128 block at the origin;
  * stroma: x 96:256, y 128:176, touching the tumor block, so the union mask is one
    contour anchored at (0, 0) while stroma alone is anchored at (96, 128).

At a 0.5 per-label threshold, INDEPENDENT stroma tiles sit on its own grid
((96,128), (160,128) at 0.75) while JOINT stroma tiles sit on the union grid
((128,128), (192,128) at 0.75); tumor is the four origin-block tiles either way.

Each entrypoint keeps its own defaults: the scalar APIs default ``a_h=0`` while
``FilterConfig`` defaults ``a_h=2``; ``preprocess_slide`` gates tissue at 0.1,
``build_tiling_result_from_mask`` at 0.5, and the configured coverage here is 0.01.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hs2p.api import (
    FilterConfig,
    SegmentationConfig,
    SlideSpec,
    TilingConfig,
    save_tiling_result,
    tile_slide,
    tile_slides,
)
from hs2p.mask import AnnotationLabels, Mask
from hs2p.preprocessing import (
    build_per_annotation_tiling_results,
    build_tiling_result_from_mask,
    open_slide,
    preprocess_slide,
    resolve_annotation_masks,
    resolve_tissue_mask,
)
from hs2p.tiling.mask import open_tissue_mask
from hs2p.tiling.single import preprocess_slide_per_annotation
from hs2p.wsi.types import (
    CoordinateOutputMode,
    CoordinateSelectionStrategy,
    SamplingSpec,
)

SPACING_UM = 0.5
TILE_PX = 64
SAMPLE_ID = "slide"

ORIGIN_BLOCK = [(0, 0), (0, 64), (64, 0), (64, 64)]
RIGHT_STRIP = [(128, 0), (128, 64)]
BOTTOM_STRIP = [(0, 128), (64, 128)]
ISLAND = [(200, 200)]
WHITE_TILE = (64, 0)

TISSUE_AT_HALF = ORIGIN_BLOCK
TISSUE_AT_TENTH = sorted(ORIGIN_BLOCK + RIGHT_STRIP)
TISSUE_AT_ONE_PERCENT = sorted(ORIGIN_BLOCK + RIGHT_STRIP + BOTTOM_STRIP)

TUMOR = ORIGIN_BLOCK
INDEPENDENT_STROMA = [(96, 128), (160, 128)]
JOINT_STROMA = [(128, 128), (192, 128)]

SCALAR_FILTERING = {
    "a_t": 4,
    "a_h": 0,
    "filter_white": False,
    "filter_black": False,
    "white_threshold": 220,
    "black_threshold": 25,
    "fraction_threshold": 0.9,
    "filter_grayspace": False,
    "grayspace_saturation_threshold": 0.05,
    "grayspace_fraction_threshold": 0.6,
    "filter_blur": False,
    "blur_threshold": 50.0,
    "qc_spacing_um": 2.0,
}
CONFIG_FILTERING = {**SCALAR_FILTERING, "a_h": 2}
WHITE_QC = {"filter_white": True, "white_threshold": 230, "fraction_threshold": 0.8}

CUSTOM_SEGMENTATION = SegmentationConfig(
    method="hsv", downsample=64, sthresh=20, sthresh_up=250, mthresh=5, close=2
)
# A precomputed tissue mask or an annotation mask is never segmented, so its result
# records no segmentation thresholds, whatever the entrypoint was given.
UNAPPLIED_SEG_THRESHOLDS = {"sthresh": None, "sthresh_up": None, "mthresh": None, "close": None}

AUTO_PROVENANCE = {
    "backend": "pil",
    "requested_backend": "auto",
    "mask_backend": "pil",
    "requested_mask_backend": "auto",
}


# --------------------------------------------------------------------------- inputs


def _write_slide(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    pixels = rng.integers(90, 200, size=(256, 256, 3), dtype=np.uint8)
    pixels[0:64, 64:128] = 255
    path = tmp_path / "slide.png"
    Image.fromarray(pixels).save(path)
    return path


def _write_tissue_mask(tmp_path: Path) -> Path:
    labels = np.zeros((256, 256), dtype=np.uint8)
    labels[0:128, 0:128] = 1
    labels[16:32, 16:32] = 0
    labels[0:128, 128:144] = 1
    labels[128:132, 0:128] = 1
    labels[200:224, 200:224] = 1
    path = tmp_path / "tissue.png"
    Image.fromarray(labels, mode="L").save(path)
    return path


def _write_annotation_mask(tmp_path: Path) -> Path:
    labels = np.zeros((256, 256), dtype=np.uint8)
    labels[0:128, 0:128] = 1
    labels[128:176, 96:256] = 2
    path = tmp_path / "annotations.png"
    Image.fromarray(labels, mode="L").save(path)
    return path


def _sampling() -> SamplingSpec:
    return SamplingSpec(
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2},
        color_mapping=None,
        tissue_percentage={"background": None, "tumor": 0.5, "stroma": 0.5},
        active_annotations=("tumor", "stroma"),
    )


def _tiling(*, tissue: float = 0.01, **kwargs) -> TilingConfig:
    return TilingConfig(
        requested_spacing_um=SPACING_UM,
        requested_tile_size_px=TILE_PX,
        tolerance=0.05,
        overlap=0.0,
        min_coverage={"tissue": tissue, "tumor": 0.5, "stroma": 0.5},
        **kwargs,
    )


def _slide_spec(image_path: Path, mask_path: Path | None) -> SlideSpec:
    return SlideSpec(
        sample_id=SAMPLE_ID,
        image_path=image_path,
        mask_path=mask_path,
        spacing_at_level_0=SPACING_UM,
    )


# -------------------------------------------------------------------------- outputs


def _origins(result) -> list[tuple[int, int]]:
    return sorted(zip(result.x.tolist(), result.y.tolist()))


def _persisted(result, output_dir: Path) -> tuple[list[tuple[int, int]], dict]:
    artifact = save_tiling_result(
        result, output_dir=output_dir, annotation=result.annotation
    )
    return _read_artifact(artifact)


def _read_artifact(artifact) -> tuple[list[tuple[int, int]], dict]:
    coordinates = np.load(artifact.coordinates_npz_path)
    origins = sorted(zip(coordinates["x"].tolist(), coordinates["y"].tolist()))
    return origins, json.loads(artifact.coordinates_meta_path.read_text())


def _provenance(meta: dict) -> dict:
    return {
        key: meta["provenance"][key]
        for key in ("backend", "requested_backend", "mask_backend", "requested_mask_backend")
    }


def _seg_thresholds(meta: dict) -> dict:
    return {key: meta["segmentation"][key] for key in UNAPPLIED_SEG_THRESHOLDS}


# ---------------------------------------------------- tissue tiling, scalar entrypoints


def _preprocess_tissue(image_path: Path, mask_path: Path, **kwargs):
    return preprocess_slide(
        image_path=image_path,
        sample_id=SAMPLE_ID,
        tissue_mask_path=mask_path,
        spacing_override=SPACING_UM,
        requested_tile_size_px=TILE_PX,
        requested_spacing_um=SPACING_UM,
        **kwargs,
    )


def _from_resolved_mask(image_path: Path, mask_path: Path, **kwargs):
    with open_slide(image_path, backend="pil", spacing_override=SPACING_UM) as slide:
        with open_tissue_mask(mask_path, backend="pil") as mask:
            resolved = resolve_tissue_mask(slide=slide, sample_id=SAMPLE_ID, mask=mask)
        return build_tiling_result_from_mask(
            slide=slide,
            resolved_mask=resolved,
            image_path=image_path,
            backend=slide.backend_name,
            requested_backend="pil",
            spacing_at_level_0=SPACING_UM,
            sample_id=SAMPLE_ID,
            requested_tile_size_px=TILE_PX,
            requested_spacing_um=SPACING_UM,
            **kwargs,
        )


def test_preprocess_slide_defaults_gate_tissue_at_a_tenth_and_record_a_h_zero(tmp_path):
    result = _preprocess_tissue(_write_slide(tmp_path), _write_tissue_mask(tmp_path))

    origins, meta = _persisted(result, tmp_path / "out")

    assert origins == TISSUE_AT_TENTH
    assert dict(zip(_origins(result), result.tissue_fractions.tolist())) == {
        (0, 0): 0.9375,
        (0, 64): 1.0,
        (64, 0): 1.0,
        (64, 64): 1.0,
        (128, 0): 0.25,
        (128, 64): 0.25,
    }
    assert meta["tiling"]["min_tissue_fraction"] == 0.1
    assert meta["filtering"] == SCALAR_FILTERING
    assert meta["segmentation"]["tissue_method"] == "precomputed_mask"
    assert meta["segmentation"]["ref_tile_size_px"] == 16
    assert _seg_thresholds(meta) == UNAPPLIED_SEG_THRESHOLDS
    assert _provenance(meta) == AUTO_PROVENANCE
    assert meta["artifact"] == {
        **meta["artifact"],
        "annotation": None,
        "selection_strategy": None,
        "output_mode": None,
    }


def test_build_tiling_result_from_mask_defaults_gate_tissue_at_half_and_record_a_h_zero(
    tmp_path,
):
    result = _from_resolved_mask(_write_slide(tmp_path), _write_tissue_mask(tmp_path))

    origins, meta = _persisted(result, tmp_path / "out")

    assert origins == TISSUE_AT_HALF
    assert meta["tiling"]["min_tissue_fraction"] == 0.5
    assert meta["filtering"] == SCALAR_FILTERING
    assert _seg_thresholds(meta) == UNAPPLIED_SEG_THRESHOLDS
    assert _provenance(meta) == {
        "backend": "pil",
        "requested_backend": "pil",
        "mask_backend": "pil",
        "requested_mask_backend": "pil",
    }


SCALAR_TISSUE_ENTRYPOINTS = {
    # Each entrypoint names the segmentation thresholds its own way.
    "preprocess_slide": (
        _preprocess_tissue,
        {"sthresh": 20, "sthresh_up": 250, "mthresh": 5, "close": 2},
    ),
    "build_tiling_result_from_mask": (
        _from_resolved_mask,
        {"seg_sthresh": 20, "seg_sthresh_up": 250, "seg_mthresh": 5, "seg_close": 2},
    ),
}


@pytest.mark.parametrize("entrypoint", list(SCALAR_TISSUE_ENTRYPOINTS))
def test_scalar_tissue_entrypoints_apply_and_record_explicit_settings(
    tmp_path, entrypoint
):
    run, segmentation_kwargs = SCALAR_TISSUE_ENTRYPOINTS[entrypoint]
    result = run(
        _write_slide(tmp_path),
        _write_tissue_mask(tmp_path),
        a_t=0,
        a_h=3,
        min_tissue_fraction=0.1,
        **segmentation_kwargs,
        **WHITE_QC,
    )

    origins, meta = _persisted(result, tmp_path / "out")

    # a_t=0 keeps the island contour, the 0.1 gate keeps its 0.14 tile, and pixel QC drops
    # the white tile.
    assert origins == sorted(set(TISSUE_AT_TENTH + ISLAND) - {WHITE_TILE})
    assert meta["tiling"]["min_tissue_fraction"] == 0.1
    assert meta["filtering"] == {**SCALAR_FILTERING, "a_t": 0, "a_h": 3, **WHITE_QC}
    # The explicit thresholds never reached a segmentation: the mask was precomputed.
    assert _seg_thresholds(meta) == UNAPPLIED_SEG_THRESHOLDS


# ------------------------------------------------- tissue tiling, config-driven workflows


def test_tile_slide_uses_configured_coverage_and_filter_config_defaults(tmp_path):
    result = tile_slide(
        _slide_spec(_write_slide(tmp_path), _write_tissue_mask(tmp_path)),
        tiling=_tiling(),
    )

    origins, meta = _persisted(result, tmp_path / "out")

    assert origins == TISSUE_AT_ONE_PERCENT
    assert meta["tiling"]["min_tissue_fraction"] == 0.01
    assert meta["filtering"] == CONFIG_FILTERING
    assert meta["segmentation"]["tissue_method"] == "precomputed_mask"
    assert _seg_thresholds(meta) == UNAPPLIED_SEG_THRESHOLDS
    assert _provenance(meta) == AUTO_PROVENANCE
    assert meta["artifact"]["selection_strategy"] == (
        CoordinateSelectionStrategy.MERGED_DEFAULT_TILING
    )
    assert meta["artifact"]["output_mode"] == CoordinateOutputMode.MERGED


def test_tile_slide_applies_and_records_configured_segmentation_and_filtering(tmp_path):
    result = tile_slide(
        _slide_spec(_write_slide(tmp_path), _write_tissue_mask(tmp_path)),
        tiling=_tiling(backend="pil", mask_backend="pil"),
        segmentation=CUSTOM_SEGMENTATION,
        filtering=FilterConfig(a_t=0, a_h=5, **WHITE_QC),
    )

    origins, meta = _persisted(result, tmp_path / "out")

    assert origins == sorted(set(TISSUE_AT_ONE_PERCENT + ISLAND) - {WHITE_TILE})
    assert meta["filtering"] == {**CONFIG_FILTERING, "a_t": 0, "a_h": 5, **WHITE_QC}
    assert _seg_thresholds(meta) == UNAPPLIED_SEG_THRESHOLDS
    assert _provenance(meta) == {
        "backend": "pil",
        "requested_backend": "pil",
        "mask_backend": "pil",
        "requested_mask_backend": "pil",
    }


@pytest.mark.parametrize("save_tiles", [False, True], ids=["coordinates", "tile-export"])
def test_tile_slides_persists_the_configured_tissue_tiling(tmp_path, save_tiles):
    output_dir = tmp_path / "out"
    artifacts = tile_slides(
        [_slide_spec(_write_slide(tmp_path), _write_tissue_mask(tmp_path))],
        tiling=_tiling(),
        segmentation=CUSTOM_SEGMENTATION,
        filtering=FilterConfig(**WHITE_QC),
        output_dir=output_dir,
        save_tiles=save_tiles,
        jpeg_backend="pil",
    )

    assert len(artifacts) == 1
    origins, meta = _read_artifact(artifacts[0])
    assert origins == sorted(set(TISSUE_AT_ONE_PERCENT) - {WHITE_TILE})
    assert meta["tiling"]["min_tissue_fraction"] == 0.01
    assert meta["filtering"] == {**CONFIG_FILTERING, **WHITE_QC}
    assert _seg_thresholds(meta) == UNAPPLIED_SEG_THRESHOLDS
    assert _provenance(meta) == AUTO_PROVENANCE


# --------------------------------------------------------------- tissue segmentation


def _write_segmentation_slide(tmp_path: Path) -> Path:
    """White background, a strongly saturated block and a pale (saturation ~12) block."""
    pixels = np.full((256, 256, 3), 255, dtype=np.uint8)
    pixels[0:128, 0:128] = (200, 60, 120)
    pixels[0:128, 128:256] = (200, 190, 195)
    path = tmp_path / "segment-me.png"
    Image.fromarray(pixels).save(path)
    return path


@pytest.mark.parametrize(
    ("sthresh", "expected"),
    [
        (8, [(x, y) for x in (0, 64, 128, 192) for y in (0, 64)]),
        (30, ORIGIN_BLOCK),
    ],
    ids=["pale-block-kept", "pale-block-dropped"],
)
@pytest.mark.parametrize("entrypoint", ["preprocess_slide", "tile_slide"])
def test_segmentation_thresholds_reach_tissue_detection(
    tmp_path, entrypoint, sthresh, expected
):
    image_path = _write_segmentation_slide(tmp_path)
    if entrypoint == "preprocess_slide":
        result = preprocess_slide(
            image_path=image_path,
            sample_id=SAMPLE_ID,
            spacing_override=SPACING_UM,
            requested_tile_size_px=TILE_PX,
            tissue_method="threshold",
            sthresh=sthresh,
        )
    else:
        result = tile_slide(
            _slide_spec(image_path, None),
            tiling=_tiling(tissue=0.1),
            segmentation=SegmentationConfig(method="threshold", sthresh=sthresh),
        )

    origins, meta = _persisted(result, tmp_path / "out")

    assert origins == sorted(expected)
    assert meta["segmentation"]["tissue_method"] == "threshold"
    assert meta["segmentation"]["sthresh"] == sthresh


# ---------------------------------------------------------------- annotation sampling


STRATEGY_STROMA = {
    CoordinateSelectionStrategy.INDEPENDENT_SAMPLING: INDEPENDENT_STROMA,
    CoordinateSelectionStrategy.JOINT_SAMPLING: JOINT_STROMA,
}
STRATEGIES = list(STRATEGY_STROMA)


def _per_annotation_scalar(image_path, mask_path, *, strategy, **kwargs):
    sampling = _sampling()
    return preprocess_slide_per_annotation(
        image_path=image_path,
        mask_path=mask_path,
        pixel_mapping=sampling.pixel_mapping,
        sampling_spec=sampling,
        selection_strategy=strategy,
        sample_id=SAMPLE_ID,
        spacing_override=SPACING_UM,
        requested_tile_size_px=TILE_PX,
        requested_spacing_um=SPACING_UM,
        **kwargs,
    )


def _per_annotation_from_resolved_masks(image_path, mask_path, *, strategy, **kwargs):
    sampling = _sampling()
    with open_slide(image_path, backend="pil", spacing_override=SPACING_UM) as slide:
        with Mask(
            path=mask_path,
            labels=AnnotationLabels(pixel_mapping=sampling.pixel_mapping),
            backend="pil",
        ) as mask:
            resolved = resolve_annotation_masks(slide=slide, mask=mask)
        return build_per_annotation_tiling_results(
            slide=slide,
            resolved_masks=resolved,
            sampling_spec=sampling,
            selection_strategy=strategy,
            image_path=image_path,
            backend=slide.backend_name,
            requested_backend="pil",
            spacing_at_level_0=SPACING_UM,
            sample_id=SAMPLE_ID,
            requested_tile_size_px=TILE_PX,
            requested_spacing_um=SPACING_UM,
            **kwargs,
        )


def _per_annotation_tile_slide(image_path, mask_path, *, strategy, **kwargs):
    filtering = kwargs.pop("filtering", FilterConfig())
    return tile_slide(
        _slide_spec(image_path, mask_path),
        tiling=_tiling(),
        segmentation=CUSTOM_SEGMENTATION,
        filtering=filtering,
        sampling=_sampling(),
        selection_strategy=strategy,
        **kwargs,
    )


ANNOTATION_ENTRYPOINTS = {
    "preprocess_slide_per_annotation": _per_annotation_scalar,
    "build_per_annotation_tiling_results": _per_annotation_from_resolved_masks,
    "tile_slide": _per_annotation_tile_slide,
}


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("entrypoint", list(ANNOTATION_ENTRYPOINTS))
def test_annotation_sampling_places_each_label_on_its_strategy_grid(
    tmp_path, entrypoint, strategy
):
    results = ANNOTATION_ENTRYPOINTS[entrypoint](
        _write_slide(tmp_path), _write_annotation_mask(tmp_path), strategy=strategy
    )

    assert list(results) == ["tumor", "stroma"]
    expected_filtering = CONFIG_FILTERING if entrypoint == "tile_slide" else SCALAR_FILTERING
    for annotation, expected in (("tumor", TUMOR), ("stroma", STRATEGY_STROMA[strategy])):
        result = results[annotation]
        origins, meta = _persisted(result, tmp_path / "out")
        assert origins == expected, annotation
        expected_fraction = 1.0 if annotation == "tumor" else 0.75
        assert result.tissue_fractions.tolist() == [expected_fraction] * len(expected)
        assert meta["tiling"]["min_tissue_fraction"] == 0.5
        assert meta["filtering"] == expected_filtering
        # Annotation sampling never segments, so it records no segmentation thresholds
        # whatever segmentation config the workflow carries.
        assert _seg_thresholds(meta) == UNAPPLIED_SEG_THRESHOLDS
        assert meta["segmentation"]["tissue_method"] == "precomputed_mask"
        assert meta["artifact"] == {
            **meta["artifact"],
            "annotation": annotation,
            "selection_strategy": strategy,
            "output_mode": CoordinateOutputMode.PER_ANNOTATION,
        }
        if entrypoint == "build_per_annotation_tiling_results":
            assert _provenance(meta) == {
                "backend": "pil",
                "requested_backend": "pil",
                "mask_backend": "pil",
                "requested_mask_backend": "pil",
            }
        else:
            assert _provenance(meta) == AUTO_PROVENANCE


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("entrypoint", list(ANNOTATION_ENTRYPOINTS))
def test_annotation_sampling_applies_pixel_qc_and_records_its_thresholds(
    tmp_path, entrypoint, strategy
):
    if entrypoint == "tile_slide":
        settings = {"filtering": FilterConfig(a_t=2, a_h=1, **WHITE_QC)}
    else:
        settings = {"a_t": 2, "a_h": 1, **WHITE_QC}

    results = ANNOTATION_ENTRYPOINTS[entrypoint](
        _write_slide(tmp_path),
        _write_annotation_mask(tmp_path),
        strategy=strategy,
        **settings,
    )

    tumor_origins, tumor_meta = _persisted(results["tumor"], tmp_path / "out")
    stroma_origins, stroma_meta = _persisted(results["stroma"], tmp_path / "out")
    assert tumor_origins == sorted(set(TUMOR) - {WHITE_TILE})
    assert stroma_origins == STRATEGY_STROMA[strategy]
    base = CONFIG_FILTERING if entrypoint == "tile_slide" else SCALAR_FILTERING
    expected_filtering = {**base, "a_t": 2, "a_h": 1, **WHITE_QC}
    assert tumor_meta["filtering"] == expected_filtering
    assert stroma_meta["filtering"] == expected_filtering


@pytest.mark.parametrize(
    ("strategy", "stroma"),
    [
        (CoordinateSelectionStrategy.INDEPENDENT_SAMPLING, INDEPENDENT_STROMA),
        (CoordinateSelectionStrategy.JOINT_SAMPLING, JOINT_STROMA),
    ],
)
@pytest.mark.parametrize("entrypoint", list(ANNOTATION_ENTRYPOINTS))
def test_merged_annotation_output_is_the_union_of_label_tiles(
    tmp_path, entrypoint, strategy, stroma
):
    results = ANNOTATION_ENTRYPOINTS[entrypoint](
        _write_slide(tmp_path),
        _write_annotation_mask(tmp_path),
        strategy=strategy,
        output_mode=CoordinateOutputMode.MERGED,
    )

    assert list(results) == [None]
    origins, meta = _persisted(results[None], tmp_path / "out")
    assert origins == sorted(TUMOR + stroma)
    assert meta["tiling"]["min_tissue_fraction"] == 0.0
    assert meta["artifact"] == {
        **meta["artifact"],
        "annotation": None,
        "selection_strategy": strategy,
        "output_mode": CoordinateOutputMode.MERGED,
    }


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_tile_slides_persists_one_artifact_per_sampled_label(tmp_path, strategy):
    output_dir = tmp_path / "out"
    artifacts = tile_slides(
        [_slide_spec(_write_slide(tmp_path), _write_annotation_mask(tmp_path))],
        tiling=_tiling(),
        segmentation=CUSTOM_SEGMENTATION,
        filtering=FilterConfig(**WHITE_QC),
        output_dir=output_dir,
        sampling=_sampling(),
        selection_strategy=strategy,
    )

    by_label = {artifact.annotation: _read_artifact(artifact) for artifact in artifacts}
    assert set(by_label) == {"tumor", "stroma"}
    assert by_label["tumor"][0] == sorted(set(TUMOR) - {WHITE_TILE})
    assert by_label["stroma"][0] == STRATEGY_STROMA[strategy]
    for annotation, (_, meta) in by_label.items():
        assert meta["filtering"] == {**CONFIG_FILTERING, **WHITE_QC}
        assert _seg_thresholds(meta) == UNAPPLIED_SEG_THRESHOLDS
        assert meta["tiling"]["min_tissue_fraction"] == 0.5
        assert _provenance(meta) == AUTO_PROVENANCE
        assert meta["artifact"]["annotation"] == annotation
        assert meta["artifact"]["selection_strategy"] == strategy
