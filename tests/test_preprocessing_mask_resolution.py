from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hs2p.preprocessing as preprocessing_mod
from hs2p.configs import SegmentationConfig
import hs2p.tiling.mask as tiling_mask_mod


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


def test_resolve_tissue_mask_uses_sam2_thumbnail_spacing(monkeypatch):
    slide, calls = _make_sam2_slide(
        level_downsamples=[1.0, 16.0],
        level_dimensions=[(256, 256), (16, 16)],
        spacing=0.25,
    )
    captured = {}

    def _fake_segment_tissue_image(image, *, config):
        captured["shape"] = image.shape
        captured["config"] = config
        return np.ones(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(
        tiling_mask_mod,
        "segment_tissue_image",
        _fake_segment_tissue_image,
    )

    resolved = preprocessing_mod.resolve_tissue_mask(
        slide=slide,
        segmentation=SegmentationConfig(
            method="sam2",
            downsample=64,
            sthresh=15,
            sam2_checkpoint_path="sam2.pt",
            sam2_config_path="sam2.yaml",
            sam2_device="cuda",
        ),
    )

    assert calls == [(1, (16, 16))]
    assert captured["shape"] == (8, 8, 3)
    # The caller's whole config reaches segmentation; only the downsample is the one the
    # SAM2 thumbnail actually used.
    assert captured["config"] == SegmentationConfig(
        method="sam2",
        downsample=32,
        sthresh=15,
        sam2_checkpoint_path="sam2.pt",
        sam2_config_path="sam2.yaml",
        sam2_device="cuda",
    )
    assert resolved.tissue_method == "sam2"
    assert resolved.requested_seg_downsample == 64
    assert resolved.seg_level == 1
    assert resolved.seg_spacing_um == 8.0
    assert resolved.seg_downsample == 32
    assert resolved.tissue_mask.shape == (8, 8)


def test_resolve_tissue_mask_without_a_mask_requires_segmentation():
    with pytest.raises(ValueError, match="segmentation"):
        preprocessing_mod.resolve_tissue_mask(slide=_make_slide())


def test_resolve_tissue_mask_no_longer_takes_loose_segmentation_keywords():
    with pytest.raises(TypeError):
        preprocessing_mod.resolve_tissue_mask(
            slide=_make_slide(), tissue_method="hsv", sthresh=8
        )


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
