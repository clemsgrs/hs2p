"""Tests for the spacing-aware read primitives.

Covers the shared planning kernel (:func:`plan_spacing_read`) and the two public
read methods built on it: ``WSI.read_region_at_spacing`` and
``WSI.read_full_at_spacing``. The WSI methods are
exercised against a lightweight stub (they only touch ``get_level_spacing``,
``level_downsamples``, ``read_region``, and ``get_slide``) so no slide backend is
required.
"""

import numpy as np
import pytest

from hs2p.wsi.geometry import plan_spacing_read
from hs2p.wsi.wsi import WSI

# level0 spacing 0.5 µm/px with x1/x2/x4 downsamples -> spacings [0.5, 1.0, 2.0]
DOWNSAMPLES = [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]


# --------------------------------------------------------------------------- #
# plan_spacing_read                                                           #
# --------------------------------------------------------------------------- #


def test_plan_spacing_read_never_selects_a_coarser_level_than_requested():
    # 0.9 is closest to level1 (1.0) but 1.0 > 0.9, so the kernel steps down to
    # level0 (0.5) rather than upsample from a coarser level.
    plan = plan_spacing_read(
        requested_spacing_um=0.9,
        level0_spacing_um=0.5,
        level_downsamples=DOWNSAMPLES,
        target_size_px=(100, 100),
        tolerance=0.05,
        content_kind="image",
    )

    assert plan.level == 0
    assert plan.read_spacing_um == pytest.approx(0.5)
    assert plan.is_within_tolerance is False
    assert plan.read_size_px == (180, 180)  # round(100 * 0.9 / 0.5)


def test_plan_spacing_read_rejects_finer_image_request_outside_tolerance():
    with pytest.raises(
        ValueError,
        match=(
            r"requested spacing 0\.25.*finest available spacing 0\.5.*"
            r"image upsampling is forbidden"
        ),
    ):
        plan_spacing_read(
            requested_spacing_um=0.25,
            level0_spacing_um=0.5,
            level_downsamples=DOWNSAMPLES,
            target_size_px=(100, 100),
            tolerance=0.05,
            content_kind="image",
        )


@pytest.mark.parametrize("content_kind", ["image", "label"])
def test_plan_spacing_read_accepts_slightly_finer_spacing_within_tolerance_without_resize(
    content_kind,
):
    plan = plan_spacing_read(
        requested_spacing_um=0.49,
        level0_spacing_um=0.5,
        level_downsamples=DOWNSAMPLES,
        target_size_px=(100, 100),
        tolerance=0.05,
        content_kind=content_kind,
    )

    assert plan.level == 0
    assert plan.is_within_tolerance is True
    assert plan.read_size_px == (100, 100)


# --------------------------------------------------------------------------- #
# WSI.read_region_at_spacing / read_full_at_spacing                           #
# --------------------------------------------------------------------------- #
class _StubWSI:
    """Minimal duck-typed stand-in for the attributes the read methods touch."""

    read_full_at_spacing = WSI.read_full_at_spacing
    read_region_at_spacing = WSI.read_region_at_spacing

    def __init__(self, *, level0_spacing_um: float, levels: list[np.ndarray]):
        self._level0_spacing_um = level0_spacing_um
        self._levels = levels
        self.read_region_calls = 0
        self.get_slide_calls = 0
        width_0, height_0 = levels[0].shape[1], levels[0].shape[0]
        self.level_downsamples = [
            (width_0 / lvl.shape[1], height_0 / lvl.shape[0]) for lvl in levels
        ]

    def get_level_spacing(self, level: int) -> float:
        return self._level0_spacing_um * self.level_downsamples[level][0]

    def get_slide(self, level: int) -> np.ndarray:
        self.get_slide_calls += 1
        return self._levels[level]

    def read_region(self, location, level, size):
        self.read_region_calls += 1
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


def test_read_region_at_spacing_defaults_to_image_and_rejects_before_backend_read():
    level0 = np.zeros((8, 8, 3), dtype=np.uint8)
    stub = _StubWSI(level0_spacing_um=0.5, levels=[level0])

    with pytest.raises(ValueError, match="image upsampling is forbidden"):
        WSI.read_region_at_spacing(
            stub,
            location=(2, 2),
            requested_spacing_um=0.25,
            size=(4, 4),
            tolerance=0.05,
            interpolation="nearest",
        )

    assert stub.read_region_calls == 0


def test_read_region_at_spacing_rejects_averaging_when_resizing_labels():
    level0 = np.zeros((8, 8, 3), dtype=np.uint8)
    stub = _StubWSI(level0_spacing_um=0.5, levels=[level0])

    with pytest.raises(
        ValueError,
        match="label content requires nearest-neighbour interpolation when resizing",
    ):
        WSI.read_region_at_spacing(
            stub,
            location=(0, 0),
            requested_spacing_um=0.25,
            size=(4, 4),
            tolerance=0.05,
            interpolation="area",
            content_kind="label",
        )

    assert stub.read_region_calls == 0


def test_read_full_at_spacing_exact_match_returns_level_unchanged():
    level0 = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    stub = _StubWSI(level0_spacing_um=0.5, levels=[level0])

    out = WSI.read_full_at_spacing(
        stub, requested_spacing_um=0.5, tolerance=0.05, interpolation="area"
    )

    assert out is level0


def test_read_full_at_spacing_defaults_to_image_and_rejects_before_backend_load():
    level0 = np.zeros((4, 4, 3), dtype=np.uint8)
    stub = _StubWSI(level0_spacing_um=0.5, levels=[level0])

    with pytest.raises(ValueError, match="image upsampling is forbidden"):
        WSI.read_full_at_spacing(
            stub,
            requested_spacing_um=0.25,
            tolerance=0.05,
            interpolation="nearest",
        )

    assert stub.get_slide_calls == 0


def test_read_full_at_spacing_rejects_averaging_when_resizing_labels():
    level0 = np.zeros((4, 4, 3), dtype=np.uint8)
    stub = _StubWSI(level0_spacing_um=0.5, levels=[level0])

    with pytest.raises(
        ValueError,
        match="label content requires nearest-neighbour interpolation when resizing",
    ):
        WSI.read_full_at_spacing(
            stub,
            requested_spacing_um=0.25,
            tolerance=0.05,
            interpolation="area",
            content_kind="label",
        )

    assert stub.get_slide_calls == 0


def test_read_full_at_spacing_accepts_slightly_finer_image_without_resize():
    level0 = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    stub = _StubWSI(level0_spacing_um=0.5, levels=[level0])

    out = WSI.read_full_at_spacing(
        stub,
        requested_spacing_um=0.49,
        tolerance=0.05,
        interpolation="area",
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
