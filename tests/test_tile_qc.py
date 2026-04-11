from types import SimpleNamespace

import cv2
import numpy as np

from hs2p.tile_qc import (
    apply_tile_qc,
    compute_blur_score,
    filter_grayspace,
    resolve_qc_read_geometry,
)


def _filter_params(**overrides):
    base = {
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
    base.update(overrides)
    return SimpleNamespace(**base)


def test_filter_grayspace_rejects_gray_tile():
    gray = np.full((64, 64, 3), 150, dtype=np.uint8)
    assert not filter_grayspace(gray, saturation_threshold=0.05, max_fraction=0.6)


def test_filter_grayspace_accepts_stained_tile():
    tile = np.zeros((64, 64, 3), dtype=np.uint8)
    tile[:, :, 0] = 200
    tile[:, :, 1] = 130
    tile[:, :, 2] = 170
    assert filter_grayspace(tile, saturation_threshold=0.05, max_fraction=0.6)


def test_filter_grayspace_ignores_masked_padding():
    tile = np.zeros((64, 64, 3), dtype=np.uint8)
    tile[:, :, 0] = 200
    tile[:, :, 1] = 130
    tile[:, :, 2] = 170
    tile[:32, :, :] = 150
    valid_mask = np.zeros((64, 64), dtype=bool)
    valid_mask[32:, :] = True
    assert filter_grayspace(
        tile,
        saturation_threshold=0.05,
        max_fraction=0.1,
        valid_mask=valid_mask,
    )


def test_compute_blur_score_decreases_after_gaussian_blur():
    rng = np.random.RandomState(42)
    base = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(base, (15, 15), 0)
    assert compute_blur_score(base) > compute_blur_score(blurred)


def test_apply_tile_qc_rejects_blurry_tile():
    tile = np.full((64, 64, 3), 150, dtype=np.uint8)
    rng = np.random.RandomState(0)
    tile = tile + rng.randint(-2, 3, tile.shape, dtype=np.int16)
    tile = np.clip(tile, 0, 255).astype(np.uint8)
    assert not apply_tile_qc(
        tile,
        valid_mask=np.ones((64, 64), dtype=bool),
        filter_params=_filter_params(filter_blur=True, blur_threshold=50.0),
    )


def test_apply_tile_qc_keeps_empty_valid_mask():
    tile = np.full((32, 32, 3), 255, dtype=np.uint8)
    assert apply_tile_qc(
        tile,
        valid_mask=np.zeros((32, 32), dtype=bool),
        filter_params=_filter_params(filter_blur=True, blur_threshold=50.0),
    )


def test_resolve_qc_read_geometry_prefers_coarser_qc_spacing_level():
    geometry = resolve_qc_read_geometry(
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        qc_spacing_um=2.0,
        base_spacing_um=0.5,
        level_downsamples=[(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)],
        tolerance=0.05,
    )
    assert geometry.level == 2
    assert geometry.read_spacing_um == 2.0
    assert geometry.qc_tile_size_px == 64
