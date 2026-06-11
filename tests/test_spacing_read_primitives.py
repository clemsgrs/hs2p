"""Tests for the spacing-aware read primitives.

Covers the shared planning kernel (:func:`plan_spacing_read`), the resize helper
(:func:`resize_array`), and the three public read methods built on them:
``WSI.read_region_at_spacing``, ``WSI.read_full_at_spacing``, and
``read_label_at_spacing``. The WSI methods are exercised against a lightweight
stub (they only touch ``get_level_spacing``, ``level_downsamples``,
``read_region``, and ``get_slide``) so no slide backend is required.
"""

import numpy as np
import pytest

from hs2p.wsi.geometry import plan_spacing_read
from hs2p.wsi.masks import read_label_at_spacing
from hs2p.wsi.wsi import WSI, resize_array

# level0 spacing 0.5 µm/px with x1/x2/x4 downsamples -> spacings [0.5, 1.0, 2.0]
DOWNSAMPLES = [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]


# --------------------------------------------------------------------------- #
# plan_spacing_read                                                           #
# --------------------------------------------------------------------------- #
def test_plan_spacing_read_exact_match_reads_target_size_directly():
    plan = plan_spacing_read(
        requested_spacing_um=0.5,
        level0_spacing_um=0.5,
        level_downsamples=DOWNSAMPLES,
        target_size_px=(100, 100),
        tolerance=0.05,
    )

    assert plan.level == 0
    assert plan.is_within_tolerance is True
    # within tolerance -> read the target size, no scaling (never upsampled)
    assert plan.read_size_px == (100, 100)


def test_plan_spacing_read_out_of_tolerance_scales_read_size_up():
    # 1.5 µm/px sits between level1 (1.0) and level2 (2.0); finest level whose
    # spacing is <= request is level1, so read larger and downscale by 1.5/1.0.
    plan = plan_spacing_read(
        requested_spacing_um=1.5,
        level0_spacing_um=0.5,
        level_downsamples=DOWNSAMPLES,
        target_size_px=(100, 100),
        tolerance=0.05,
    )

    assert plan.level == 1
    assert plan.read_spacing_um == pytest.approx(1.0)
    assert plan.is_within_tolerance is False
    assert plan.read_size_px == (150, 150)  # round(100 * 1.5 / 1.0)


def test_plan_spacing_read_never_selects_a_coarser_level_than_requested():
    # 0.9 is closest to level1 (1.0) but 1.0 > 0.9, so the kernel steps down to
    # level0 (0.5) rather than upsample from a coarser level.
    plan = plan_spacing_read(
        requested_spacing_um=0.9,
        level0_spacing_um=0.5,
        level_downsamples=DOWNSAMPLES,
        target_size_px=(100, 100),
        tolerance=0.05,
    )

    assert plan.level == 0
    assert plan.read_spacing_um == pytest.approx(0.5)
    assert plan.is_within_tolerance is False
    assert plan.read_size_px == (180, 180)  # round(100 * 0.9 / 0.5)


# --------------------------------------------------------------------------- #
# resize_array                                                                #
# --------------------------------------------------------------------------- #
def test_resize_array_is_a_noop_when_target_matches_shape():
    arr = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

    out = resize_array(arr, (4, 4), interpolation="area")

    # identical shape -> same object returned (lossless exact match)
    assert out is arr


def test_resize_array_downscales_with_nearest_preserving_class_ids():
    labels = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
        dtype=np.int32,
    )

    out = resize_array(labels, (2, 2), interpolation="nearest")

    assert out.shape == (2, 2)
    assert set(np.unique(out)).issubset({1, 2, 3, 4})


def test_resize_array_rejects_unknown_interpolation():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="unknown interpolation"):
        resize_array(arr, (2, 2), interpolation="bogus")


# --------------------------------------------------------------------------- #
# WSI.read_region_at_spacing / read_full_at_spacing                           #
# --------------------------------------------------------------------------- #
class _StubWSI:
    """Minimal duck-typed stand-in for the attributes the read methods touch."""

    def __init__(self, *, level0_spacing_um: float, levels: list[np.ndarray]):
        self._level0_spacing_um = level0_spacing_um
        self._levels = levels
        width_0, height_0 = levels[0].shape[1], levels[0].shape[0]
        self.level_downsamples = [
            (width_0 / lvl.shape[1], height_0 / lvl.shape[0]) for lvl in levels
        ]

    def get_level_spacing(self, level: int) -> float:
        return self._level0_spacing_um * self.level_downsamples[level][0]

    def get_slide(self, level: int) -> np.ndarray:
        return self._levels[level]

    def read_region(self, location, level, size):
        x, y = location
        ds = self.level_downsamples[level][0]
        arr = self._levels[level]
        x_l, y_l = int(round(x / ds)), int(round(y / ds))
        w, h = int(size[0]), int(size[1])
        return arr[y_l : y_l + h, x_l : x_l + w, :]


def test_read_region_at_spacing_exact_match_returns_native_region_unresized():
    level0 = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    stub = _StubWSI(level0_spacing_um=0.5, levels=[level0])

    region = WSI.read_region_at_spacing(
        stub,
        location=(2, 2),
        requested_spacing_um=0.5,
        size=(4, 4),
        tolerance=0.05,
        interpolation="area",
    )

    # exact spacing match -> read 4x4 natively, no resize
    np.testing.assert_array_equal(region, level0[2:6, 2:6, :])


def test_read_full_at_spacing_exact_match_returns_level_unchanged():
    level0 = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    stub = _StubWSI(level0_spacing_um=0.5, levels=[level0])

    out = WSI.read_full_at_spacing(
        stub, requested_spacing_um=0.5, tolerance=0.05, interpolation="area"
    )

    assert out is level0


def test_read_full_at_spacing_downscales_when_no_level_matches():
    level0 = np.ones((4, 4, 3), dtype=np.uint8) * 200
    stub = _StubWSI(level0_spacing_um=0.5, levels=[level0])

    # request 1.0 with only a 0.5 level present -> downscale by 0.5/1.0
    out = WSI.read_full_at_spacing(
        stub, requested_spacing_um=1.0, tolerance=0.05, interpolation="area"
    )

    assert out.shape == (2, 2, 3)


# --------------------------------------------------------------------------- #
# read_label_at_spacing                                                       #
# --------------------------------------------------------------------------- #
class _StubLabelWSI:
    def __init__(self, arr: np.ndarray):
        self._arr = arr
        self.calls: list[dict] = []

    def read_full_at_spacing(self, requested_spacing_um, *, tolerance, interpolation):
        self.calls.append(
            {
                "requested_spacing_um": requested_spacing_um,
                "tolerance": tolerance,
                "interpolation": interpolation,
            }
        )
        return self._arr


def test_read_label_at_spacing_collapses_channel_replicated_rgb():
    labels = np.array([[0, 1], [2, 3]], dtype=np.int32)
    rgb = np.stack([labels, labels, labels], axis=-1)
    stub = _StubLabelWSI(rgb)

    out = read_label_at_spacing(stub, requested_spacing_um=2.0, tolerance=0.05)

    np.testing.assert_array_equal(out, labels)
    # labels must be resampled with nearest-neighbor to avoid inventing ids
    assert stub.calls == [
        {"requested_spacing_um": 2.0, "tolerance": 0.05, "interpolation": "nearest"}
    ]


def test_read_label_at_spacing_rejects_genuine_colour_image():
    colour = np.zeros((2, 2, 3), dtype=np.uint8)
    colour[..., 0] = 10  # red differs from green/blue -> not a replicated label
    stub = _StubLabelWSI(colour)

    with pytest.raises(ValueError, match="non-identical RGB channels"):
        read_label_at_spacing(stub, requested_spacing_um=2.0, tolerance=0.05)


def test_read_label_at_spacing_rejects_non_integer_dtype():
    labels = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    stub = _StubLabelWSI(labels)

    with pytest.raises(ValueError, match="integer dtype"):
        read_label_at_spacing(stub, requested_spacing_um=2.0, tolerance=0.05)
