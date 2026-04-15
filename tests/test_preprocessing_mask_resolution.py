from pathlib import Path
from types import SimpleNamespace

import numpy as np

import hs2p.preprocessing as preprocessing_mod


def _make_slide():
    return SimpleNamespace(
        level_downsamples=[1.0, 4.0],
        spacing=0.25,
        level_dimensions=[(100, 100), (25, 25)],
        dimensions=(100, 100),
        backend_name="asap",
        read_region=lambda *args, **kwargs: np.zeros((25, 25, 3), dtype=np.uint8),
    )


def test_resolve_tissue_mask_uses_precomputed_mask_without_segmentation(
    monkeypatch,
):
    slide = _make_slide()
    called = {"segment": False}

    def _fake_load_precomputed_tissue_mask(**kwargs):
        assert kwargs["mask_path"] == Path("mask.png")
        assert kwargs["seg_level"] == 1
        return np.full((25, 25), 255, dtype=np.uint8), 0, 1.0

    def _fake_segment_tissue_image(*args, **kwargs):
        called["segment"] = True
        raise AssertionError("segment_tissue_image should not be called")

    monkeypatch.setattr(
        preprocessing_mod,
        "load_precomputed_tissue_mask",
        _fake_load_precomputed_tissue_mask,
    )
    monkeypatch.setattr(
        preprocessing_mod,
        "segment_tissue_image",
        _fake_segment_tissue_image,
    )

    resolved = preprocessing_mod.resolve_tissue_mask(
        slide=slide,
        tissue_mask_path=Path("mask.png"),
        tissue_method="sam2",
        seg_downsample=64,
    )

    assert resolved.tissue_method == "precomputed_mask"
    assert resolved.seg_downsample == 64
    assert resolved.mask_path == Path("mask.png")
    assert resolved.tissue_mask_tissue_value == 1
    assert not called["segment"]


def test_resolve_tissue_mask_requires_an_explicit_method():
    slide = _make_slide()

    try:
        preprocessing_mod.resolve_tissue_mask(
            slide=slide,
            tissue_mask_path=None,
        )
    except ValueError as exc:
        assert "tissue_method is required" in str(exc)
    else:
        raise AssertionError("resolve_tissue_mask should require tissue_method")


def test_resolve_tissue_mask_allows_precomputed_masks_without_a_method(
    monkeypatch,
):
    slide = _make_slide()

    monkeypatch.setattr(
        preprocessing_mod,
        "load_precomputed_tissue_mask",
        lambda **kwargs: (np.full((25, 25), 255, dtype=np.uint8), 0, 1.0),
    )
    monkeypatch.setattr(
        preprocessing_mod,
        "segment_tissue_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("segment_tissue_image should not be called")
        ),
    )

    resolved = preprocessing_mod.resolve_tissue_mask(
        slide=slide,
        tissue_mask_path=Path("mask.png"),
    )

    assert resolved.tissue_method == "precomputed_mask"
    assert resolved.mask_path == Path("mask.png")


def test_build_tiling_result_from_mask_preserves_resolved_mask_metadata():
    slide = _make_slide()
    resolved = preprocessing_mod.ResolvedTissueMask(
        tissue_mask=np.zeros((25, 25), dtype=np.uint8),
        tissue_method="sam2",
        seg_downsample=64,
        seg_level=1,
        seg_spacing_um=1.0,
        mask_path=Path("mask.png"),
        tissue_mask_tissue_value=1,
        mask_level=0,
        mask_spacing_um=1.0,
    )

    result = preprocessing_mod.build_tiling_result_from_mask(
        slide=slide,
        resolved_mask=resolved,
        image_path=Path("slide.svs"),
        backend="asap",
        requested_backend="auto",
        sample_id="slide-1",
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        min_tissue_fraction=0.1,
        overlap=0.0,
        tolerance=0.05,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=16,
        a_t=4,
        a_h=0,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
        filter_grayspace=False,
        grayspace_saturation_threshold=0.05,
        grayspace_fraction_threshold=0.6,
        filter_blur=False,
        blur_threshold=50.0,
        qc_spacing_um=2.0,
        num_workers=1,
    )

    assert result.tissue_method == "sam2"
    assert result.seg_downsample == 64
    assert result.seg_level == 1
    assert result.seg_spacing_um == 1.0
    assert result.mask_path == Path("mask.png")
    assert result.tissue_mask_tissue_value == 1
    assert result.mask_level == 0
    assert result.mask_spacing_um == 1.0
    assert result.image_path == Path("slide.svs")
    assert result.backend == "asap"
    assert result.requested_backend == "auto"
