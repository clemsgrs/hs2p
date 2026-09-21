from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.mask as source_mask_mod
import hs2p.preprocessing as preprocessing_mod
import hs2p.tiling.mask as tiling_mask_mod
from hs2p.mask import AnnotationLabels, Mask, TissueLabels


def _make_slide():
    return SimpleNamespace(
        level_downsamples=[1.0, 4.0],
        spacing=0.25,
        level_dimensions=[(100, 100), (25, 25)],
        dimensions=(100, 100),
        backend_name="asap",
        read_region=lambda *args, **kwargs: np.zeros((25, 25, 3), dtype=np.uint8),
    )


def _make_sam2_slide(level_downsamples, level_dimensions, spacing=0.25):
    calls = []

    def _read_region(location, level, size):
        del location
        calls.append((level, size))
        return np.full((size[1], size[0], 3), fill_value=level, dtype=np.uint8)

    return (
        SimpleNamespace(
            level_downsamples=level_downsamples,
            spacing=spacing,
            level_dimensions=level_dimensions,
            dimensions=level_dimensions[0],
            backend_name="asap",
            read_region=_read_region,
        ),
        calls,
    )


def _open_mask(monkeypatch, labels) -> Mask:
    """Open an all-tissue 25 px ``Mask`` at 1.0 um/px: the slide's level-1 grid."""
    reader = SimpleNamespace(
        native_spacing=1.0,
        level_dimensions=[(25, 25)],
        level_downsamples=[(1.0, 1.0)],
        read_region=lambda location, level, size: np.ones((25, 25), dtype=np.uint8),
        close=lambda: None,
    )
    monkeypatch.setattr(source_mask_mod, "open_slide", lambda *args, **kwargs: reader)
    return Mask(path=Path("mask.png"), labels=labels, backend="asap")


def _forbid_segmentation(monkeypatch) -> None:
    monkeypatch.setattr(
        tiling_mask_mod,
        "segment_tissue_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("segment_tissue_image should not be called")
        ),
    )


def test_resolve_tissue_mask_uses_precomputed_mask_without_segmentation(
    monkeypatch,
):
    mask = _open_mask(monkeypatch, TissueLabels(background=0, tissue=1))
    _forbid_segmentation(monkeypatch)

    resolved = preprocessing_mod.resolve_tissue_mask(
        slide=_make_slide(),
        mask=mask,
        tissue_method="sam2",
        seg_downsample=64,
    )

    assert resolved.tissue_method == "precomputed_mask"
    assert resolved.requested_seg_downsample == 64
    # Slide level_downsamples=[1.0, 4.0] → closest pyramid level for requested
    # downsample=64 is level 1 (downsample 4.0 → seg_spacing=1.0 µm at base
    # spacing 0.25), so the resolved value is 4.
    assert resolved.seg_downsample == 4
    assert resolved.mask_path == Path("mask.png")
    assert resolved.tissue_mask_tissue_value == 1
    assert (resolved.mask_level, resolved.mask_spacing_um) == (0, 1.0)
    np.testing.assert_array_equal(
        resolved.tissue_mask, np.full((25, 25), 255, dtype=np.uint8)
    )


def test_resolve_tissue_mask_rejects_a_mask_without_tissue_labels(monkeypatch):
    mask = _open_mask(
        monkeypatch, AnnotationLabels(pixel_mapping={"background": 0, "tumor": 1})
    )

    with pytest.raises(ValueError, match="must declare TissueLabels"):
        preprocessing_mod.resolve_tissue_mask(slide=_make_slide(), mask=mask)


@pytest.mark.parametrize(
    ("pixel_mapping", "background", "tissue"),
    [
        (None, 0, 1),
        ({"tumor": 3}, 0, 1),
        ({"background": 2, "tissue": 255, "tumor": 3}, 2, 255),
    ],
)
def test_tissue_labels_come_from_the_pixel_mapping_with_defaults(
    pixel_mapping, background, tissue
):
    labels = tiling_mask_mod.tissue_labels_from_pixel_mapping(pixel_mapping)

    assert labels == TissueLabels(background=background, tissue=tissue)


def test_tissue_labels_reject_a_tissue_entry_merging_several_ids():
    with pytest.raises(ValueError, match="TissueLabels tissue"):
        tiling_mask_mod.tissue_labels_from_pixel_mapping({"tissue": [1, 2]})


def test_prepare_sam2_thumbnail_uses_existing_level_when_within_tolerance():
    slide, calls = _make_sam2_slide(
        level_downsamples=[1.0, 32.0],
        level_dimensions=[(256, 256), (8, 8)],
        spacing=0.25,
    )

    thumbnail = preprocessing_mod.prepare_sam2_thumbnail(
        slide=slide,
        target_spacing_um=8.0,
    )

    assert calls == [(1, (8, 8))]
    assert thumbnail.seg_level == 1
    assert thumbnail.seg_spacing_um == 8.0
    assert thumbnail.resized is False
    assert thumbnail.image.shape == (8, 8, 3)


def test_prepare_sam2_thumbnail_resizes_when_out_of_tolerance():
    slide, calls = _make_sam2_slide(
        level_downsamples=[1.0, 16.0],
        level_dimensions=[(256, 256), (16, 16)],
        spacing=0.25,
    )

    thumbnail = preprocessing_mod.prepare_sam2_thumbnail(
        slide=slide,
        target_spacing_um=8.0,
    )

    assert calls == [(1, (16, 16))]
    assert thumbnail.seg_level == 1
    assert thumbnail.seg_spacing_um == 8.0
    assert thumbnail.resized is True
    assert thumbnail.image.shape == (8, 8, 3)


def test_resolve_tissue_mask_requires_an_explicit_method():
    slide = _make_slide()

    try:
        preprocessing_mod.resolve_tissue_mask(slide=slide)
    except ValueError as exc:
        assert "tissue_method is required" in str(exc)
    else:
        raise AssertionError("resolve_tissue_mask should require tissue_method")


def test_resolve_tissue_mask_allows_precomputed_masks_without_a_method(
    monkeypatch,
):
    mask = _open_mask(monkeypatch, TissueLabels(background=0, tissue=1))
    _forbid_segmentation(monkeypatch)

    resolved = preprocessing_mod.resolve_tissue_mask(slide=_make_slide(), mask=mask)

    assert resolved.tissue_method == "precomputed_mask"
    assert resolved.mask_path == Path("mask.png")


def test_resolve_tissue_mask_uses_sam2_thumbnail_spacing(monkeypatch):
    slide, calls = _make_sam2_slide(
        level_downsamples=[1.0, 16.0],
        level_dimensions=[(256, 256), (16, 16)],
        spacing=0.25,
    )
    captured = {}

    def _fake_segment_tissue_image(image, *, config):
        captured["shape"] = image.shape
        captured["method"] = config.method
        captured["downsample"] = config.downsample
        return np.ones(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(
        tiling_mask_mod,
        "segment_tissue_image",
        _fake_segment_tissue_image,
    )

    resolved = preprocessing_mod.resolve_tissue_mask(
        slide=slide,
        tissue_method="sam2",
        seg_downsample=64,
    )

    assert calls == [(1, (16, 16))]
    assert captured["shape"] == (8, 8, 3)
    assert captured["method"] == "sam2"
    assert captured["downsample"] == 32
    assert resolved.tissue_method == "sam2"
    assert resolved.requested_seg_downsample == 64
    assert resolved.seg_level == 1
    assert resolved.seg_spacing_um == 8.0
    assert resolved.seg_downsample == 32
    assert resolved.tissue_mask.shape == (8, 8)


def test_build_tiling_result_from_mask_preserves_resolved_mask_metadata():
    slide = _make_slide()
    resolved = preprocessing_mod.ResolvedTissueMask(
        tissue_mask=np.zeros((25, 25), dtype=np.uint8),
        tissue_method="sam2",
        requested_seg_downsample=64,
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
        spacing_at_level_0=0.25,
        sam2_checkpoint_path=Path("sam2.pt"),
        sam2_config_path=Path("sam2.yaml"),
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
    assert result.spacing_at_level_0 == 0.25
    assert result.sam2_checkpoint_path == Path("sam2.pt")
    assert result.sam2_config_path == Path("sam2.yaml")
    assert result.requested_backend == "auto"


def test_build_tiling_result_from_mask_omits_sam2_identity_for_hsv():
    result = preprocessing_mod.build_tiling_result_from_mask(
        slide=_make_slide(),
        resolved_mask=preprocessing_mod.ResolvedTissueMask(
            tissue_mask=np.zeros((25, 25), dtype=np.uint8),
            tissue_method="hsv",
            requested_seg_downsample=64,
            seg_downsample=64,
            seg_level=1,
            seg_spacing_um=1.0,
        ),
        image_path=Path("slide.svs"),
        backend="asap",
        requested_backend="auto",
        sam2_checkpoint_path=Path("unused-sam2.pt"),
        sam2_config_path=Path("unused-sam2.yaml"),
    )

    assert result.sam2_checkpoint_path is None
    assert result.sam2_config_path is None
