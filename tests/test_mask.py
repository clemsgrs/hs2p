"""Public behavior of the first-class source-mask domain (``hs2p.mask``, #190, #193).

``_FakeReader`` stands in for pyramidal, spacing-tagged, wide-dtype and multi-channel
sources so every case runs without a native backend.
"""

import numpy as np
import pytest

import hs2p.mask as mask_mod
from hs2p.mask import AnnotationLabels, Mask, MaskRead, TissueLabels

TISSUE = TissueLabels(background=0, tissue=1)


class _FakeReader:
    """An in-memory mask pyramid; ``levels`` are the arrays each level decodes to."""

    def __init__(self, levels, *, native_spacing=None, level_dimensions=None):
        self._levels = levels
        self.native_spacing = native_spacing
        self.level_dimensions = level_dimensions or [
            (level.shape[1], level.shape[0]) for level in levels
        ]
        width_0 = self.level_dimensions[0][0]
        self.level_downsamples = [
            (width_0 / width, width_0 / width) for width, _ in self.level_dimensions
        ]
        self.read_levels: list[int] = []
        self.close_count = 0

    def read_region(self, location, level, size):
        assert location == (0, 0)
        assert size == self.level_dimensions[level]
        self.read_levels.append(level)
        return self._levels[level]

    def close(self):
        self.close_count += 1


def _open_fake_mask(monkeypatch, reader, *, labels=TISSUE, path="fake-mask.tif"):
    """Open a ``Mask`` whose concrete ``fake`` backend yields ``reader``."""
    monkeypatch.setattr(mask_mod, "open_slide", lambda *args, **kwargs: reader)
    return Mask(path=path, labels=labels, backend="fake")


@pytest.mark.parametrize(
    ("background", "tissue"),
    [(1, 1), (-1, 1), (0, 256), (0, 1.0), (False, True)],
)
def test_tissue_labels_reject_invalid_ids(background, tissue):
    with pytest.raises(ValueError, match="TissueLabels"):
        TissueLabels(background=background, tissue=tissue)


def test_closed_mask_rejects_alignment_and_aligned_reads(monkeypatch):
    reader = _FakeReader([np.zeros((2, 2), dtype=np.uint8)])
    mask = _open_fake_mask(monkeypatch, reader, path="closed-mask.tif")
    aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(2, 2))
    mask.close()

    with pytest.raises(ValueError, match=r"Mask is closed: path=closed-mask\.tif"):
        mask.align_to(reference_spacing_um=0.5, reference_dimensions=(2, 2))
    with pytest.raises(ValueError, match=r"Mask is closed: path=closed-mask\.tif"):
        aligned.read_full(target_spacing_um=0.5, target_dimensions=(2, 2))
    assert reader.read_levels == []


def _blank_levels(*level_dimensions):
    return [np.zeros((height, width), dtype=np.uint8) for width, height in level_dimensions]


@pytest.mark.parametrize(
    ("reference_dimensions", "mask_dimensions"),
    [
        # 1000 / 16 = 62.5 rounded up on one axis and down on the other
        ((1000, 1000), (63, 62)),
        # elongated 30:1 reference: 30008 / 16 = 1875.5 -> 1876, 1000 / 16 = 62.5 -> 63
        ((30008, 1000), (1876, 63)),
    ],
)
def test_align_to_allows_one_mask_pixel_of_rounding_per_axis(
    monkeypatch, reference_dimensions, mask_dimensions
):
    mask = _open_fake_mask(monkeypatch, _FakeReader(_blank_levels(mask_dimensions)))

    aligned = mask.align_to(
        reference_spacing_um=0.25, reference_dimensions=reference_dimensions
    )

    assert aligned.reference_dimensions == reference_dimensions


def test_align_to_warns_once_for_a_nominal_spacing_tag_and_still_reads(
    monkeypatch, caplog
):
    # 0.2431 um reference at 1/16 -> effective 3.8896 um; the mask is tagged 4.0 um (2.8%)
    labels = np.ones((50, 100), dtype=np.uint8)
    reader = _FakeReader([labels], native_spacing=4.0)
    mask = _open_fake_mask(monkeypatch, reader, path="nominal-mask.tif")

    with caplog.at_level("WARNING"):
        aligned = mask.align_to(
            reference_spacing_um=0.2431, reference_dimensions=(1600, 800)
        )
    read = aligned.read_full(target_spacing_um=3.8896, target_dimensions=(100, 50))

    assert len(caplog.records) == 1
    warning = caplog.records[0].getMessage()
    assert "path=nominal-mask.tif" in warning
    assert "file spacing 4.0000 um/px" in warning
    assert "effective spacing 3.8896 um/px" in warning
    assert "2.8%" in warning
    assert read.read_spacing_um == pytest.approx(3.8896, rel=1e-12)
    np.testing.assert_array_equal(read.labels, labels)


def test_align_to_rejects_file_spacing_disagreeing_by_more_than_five_percent(
    monkeypatch,
):
    # effective 4.0 um vs tagged 4.3 um: 7.5%
    reader = _FakeReader(_blank_levels((100, 50)), native_spacing=4.3)
    mask = _open_fake_mask(monkeypatch, reader, path="mistagged-mask.tif")

    with pytest.raises(ValueError) as excinfo:
        mask.align_to(reference_spacing_um=0.25, reference_dimensions=(1600, 800))

    message = str(excinfo.value)
    assert "path=mistagged-mask.tif" in message
    assert "mask dimensions 100x50" in message
    assert "reference dimensions 1600x800" in message
    assert "effective spacing 4.0000 um/px" in message
    assert "file spacing 4.3000 um/px" in message
    assert "7.5%" in message


@pytest.mark.parametrize(
    ("reference_spacing_um", "reference_dimensions"),
    [(0.0, (1600, 800)), (float("nan"), (1600, 800)), (0.25, (0, 800))],
)
def test_align_to_rejects_an_invalid_reference(
    monkeypatch, reference_spacing_um, reference_dimensions
):
    mask = _open_fake_mask(monkeypatch, _FakeReader(_blank_levels((100, 50))))

    with pytest.raises(ValueError, match="reference"):
        mask.align_to(
            reference_spacing_um=reference_spacing_um,
            reference_dimensions=reference_dimensions,
        )


@pytest.mark.parametrize("backend", ["openslide", "vips"])
def test_untagged_tiff_mask_without_spacing_opens_aligns_and_reads(
    tmp_path, caplog, backend
):
    pytest.importorskip({"openslide": "openslide", "vips": "pyvips"}[backend])
    tifffile = pytest.importorskip("tifffile")
    labels = np.zeros((32, 64), dtype=np.uint8)
    labels[:16, 32:] = 1
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, labels, tile=(32, 32), photometric="minisblack")

    with caplog.at_level("WARNING"), Mask(path=path, labels=TISSUE, backend=backend) as mask:
        aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(256, 128))
        read = aligned.read_full(target_spacing_um=2.0, target_dimensions=(64, 32))

    assert read.read_level == 0
    # 0.5 um x 256 / 64
    assert read.read_spacing_um == 2.0
    np.testing.assert_array_equal(read.labels, labels)
    assert caplog.records == []


def _three_level_annotation_mask(monkeypatch):
    """Effective level spacings 1.0 / 2.0 / 4.0 um; level ``n`` decodes to all ``n``."""
    reader = _FakeReader(
        [
            np.full((80, 160), 0, dtype=np.uint8),
            np.full((40, 80), 1, dtype=np.uint8),
            np.full((20, 40), 2, dtype=np.uint8),
        ]
    )
    labels = AnnotationLabels(pixel_mapping={"background": 0, "tumor": [1, 2]})
    mask = _open_fake_mask(monkeypatch, reader, labels=labels)
    aligned = mask.align_to(reference_spacing_um=0.25, reference_dimensions=(640, 320))
    return reader, aligned


def test_full_read_keeps_a_level_when_the_request_carries_float_noise(monkeypatch):
    reader, aligned = _three_level_annotation_mask(monkeypatch)

    read = aligned.read_full(target_spacing_um=3.9999996, target_dimensions=(40, 20))

    assert read.read_level == 2
    assert read.read_spacing_um == 4.0
    assert reader.read_levels == [2]
    np.testing.assert_array_equal(read.labels, np.full((20, 40), 2, dtype=np.uint8))


def test_full_read_returns_exact_dimensions_when_the_level_is_within_tolerance(
    monkeypatch,
):
    reader = _FakeReader([np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)])
    mask = _open_fake_mask(monkeypatch, reader)
    aligned = mask.align_to(reference_spacing_um=0.25, reference_dimensions=(48, 32))

    # level 0 is 4.0 um: the 4.02 um request is within the 1% level tolerance
    read = aligned.read_full(target_spacing_um=4.02, target_dimensions=(4, 2))

    assert read.read_level == 0
    np.testing.assert_array_equal(
        read.labels,
        np.array([[0, 0, 1, 0], [1, 1, 0, 1]], dtype=np.uint8),
    )


def test_full_read_validates_the_native_decode_before_resampling(monkeypatch):
    # The undeclared 7 sits where a 4x4 -> 2x2 nearest-neighbor downsample never samples.
    native = np.zeros((4, 4), dtype=np.uint8)
    native[1, 1] = 7
    mask = _open_fake_mask(monkeypatch, _FakeReader([native]), path="stray-label.tif")
    aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(16, 16))

    with pytest.raises(ValueError) as excinfo:
        aligned.read_full(target_spacing_um=4.0, target_dimensions=(2, 2))

    message = str(excinfo.value)
    assert "path=stray-label.tif" in message
    assert "undeclared label IDs [7]" in message
    assert "declared [0, 1]" in message


def test_full_read_narrows_wide_integer_storage_with_ids_within_range(monkeypatch):
    native = np.array([[0, 255], [255, 0]], dtype=np.uint16)
    labels = TissueLabels(background=0, tissue=255)
    mask = _open_fake_mask(monkeypatch, _FakeReader([native]), labels=labels)
    aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(8, 8))

    read = aligned.read_full(target_spacing_um=2.0, target_dimensions=(2, 2))

    assert read.labels.dtype == np.uint8
    np.testing.assert_array_equal(
        read.labels, np.array([[0, 255], [255, 0]], dtype=np.uint8)
    )


@pytest.mark.parametrize(
    ("native", "reason"),
    [
        (np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32), "integer dtype"),
        (np.array([[0, 1], [256, 0]], dtype=np.uint16), r"outside \[0, 255\]"),
        (np.array([[0, 1], [-1, 0]], dtype=np.int16), r"outside \[0, 255\]"),
        (
            np.stack(
                [
                    np.array([[0, 1], [1, 0]], dtype=np.uint8),
                    np.array([[0, 1], [1, 0]], dtype=np.uint8),
                    np.array([[0, 1], [1, 1]], dtype=np.uint8),
                ],
                axis=-1,
            ),
            "channels differ",
        ),
    ],
    ids=["float", "above-range", "negative", "color"],
)
def test_full_read_rejects_invalid_native_decodes(monkeypatch, native, reason):
    mask = _open_fake_mask(monkeypatch, _FakeReader([native]))
    aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(8, 8))

    with pytest.raises(ValueError, match=reason):
        aligned.read_full(target_spacing_um=2.0, target_dimensions=(2, 2))


def test_mask_read_is_immutable(monkeypatch):
    mask = _open_fake_mask(monkeypatch, _FakeReader(_blank_levels((2, 2))))
    aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(8, 8))

    read = aligned.read_full(target_spacing_um=2.0, target_dimensions=(2, 2))

    with pytest.raises(AttributeError):
        read.read_level = 1
    with pytest.raises(ValueError, match="read-only"):
        read.labels[0, 0] = 1


class _WindowFakeReader(_FakeReader):
    """A ``_FakeReader`` that crops windows the way every hs2p backend addresses them.

    ``location`` is in the mask file's own level-0 pixels and lands on the level through
    ``floor(location / level_downsamples[level][0])`` on both axes; ``size`` is in level
    pixels. ``windows`` records every call.
    """

    def __init__(self, levels, **kwargs):
        super().__init__(levels, **kwargs)
        self.windows: list[tuple[tuple[int, int], int, tuple[int, int]]] = []

    def read_region(self, location, level, size):
        self.windows.append((location, level, size))
        downsample = self.level_downsamples[level][0]
        x, y = (int(value // downsample) for value in location)
        width, height = size
        level_width, level_height = self.level_dimensions[level]
        assert 0 <= x and x + width <= level_width
        assert 0 <= y and y + height <= level_height
        return self._levels[level][y : y + height, x : x + width]


def _numbered_labels(count):
    return AnnotationLabels(
        pixel_mapping={"background": 0, "tumor": list(range(1, count))}
    )


def _coarse_numbered_mask(monkeypatch, *, path="fake-mask.tif"):
    """A 0.25 um, 128x64 reference under an 8x4 mask: level 0 is 4.0 um, 16x coarser.

    Mask pixel ``(row, col)`` holds ``row * 8 + col`` and covers the reference pixels
    ``[16 * col, 16 * col + 16) x [16 * row, 16 * row + 16)``.
    """
    reader = _WindowFakeReader([np.arange(32, dtype=np.uint8).reshape(4, 8)])
    mask = _open_fake_mask(
        monkeypatch, reader, labels=_numbered_labels(32), path=path
    )
    aligned = mask.align_to(reference_spacing_um=0.25, reference_dimensions=(128, 64))
    return reader, aligned


@pytest.mark.parametrize(
    ("location", "target_dimensions"),
    [
        ((-16, 0), (2, 2)),
        ((0, -16), (2, 2)),
        # 4.0 um target pixels are 16 reference px: x spans [112, 144) on a 128 px canvas
        ((112, 0), (2, 2)),
        # y spans [48, 80) on a 64 px canvas
        ((0, 48), (2, 2)),
    ],
    ids=["left", "top", "right", "bottom"],
)
def test_region_read_rejects_a_request_beyond_the_reference_canvas(
    monkeypatch, location, target_dimensions
):
    reader, aligned = _coarse_numbered_mask(monkeypatch, path="edge-mask.tif")

    with pytest.raises(ValueError) as excinfo:
        aligned.read_region(
            location=location,
            target_spacing_um=4.0,
            target_dimensions=target_dimensions,
        )

    message = str(excinfo.value)
    assert "path=edge-mask.tif" in message
    assert f"location {location}" in message
    assert "32x32 reference px" in message
    assert "reference dimensions 128x64" in message
    assert reader.windows == []


def test_region_read_lands_on_the_intended_pixel_under_a_non_integer_downsample(
    monkeypatch,
):
    # A 9 px level 0 over a 4 px level 1 is a 2.25x downsample. Level-1 pixel 1 starts at
    # mask-file x = 2.25: a reader flooring 2 / 2.25 would decode pixel 0, 3 / 2.25 is 1.
    reader = _WindowFakeReader(
        [
            np.zeros((9, 9), dtype=np.uint8),
            np.arange(16, dtype=np.uint8).reshape(4, 4),
        ]
    )
    mask = _open_fake_mask(monkeypatch, reader, labels=_numbered_labels(16))
    # effective level spacings: 1.0 um x 36 / 9 = 4.0 um, then 4.0 um x 2.25 = 9.0 um
    aligned = mask.align_to(reference_spacing_um=1.0, reference_dimensions=(36, 36))

    # level 1 holds 36 / 4 = 9 reference px per pixel, so (9, 18) is (col 1, row 2)
    read = aligned.read_region(
        location=(9, 18), target_spacing_um=9.0, target_dimensions=(1, 1)
    )

    assert read.read_level == 1
    np.testing.assert_array_equal(read.labels, np.array([[9]], dtype=np.uint8))
    assert reader.windows == [((3, 5), 1, (1, 1))]


def test_region_read_registers_with_the_reference_raster_under_a_coarser_mask(
    monkeypatch,
):
    # A 0.25 um, 128x64 reference raster whose tissue fills x in [32, 64), y in [16, 48),
    # and its 4.0 um mask: the same rectangle at 1/16, columns 2-3 of rows 1-2.
    reference = np.zeros((64, 128), dtype=np.uint8)
    reference[16:48, 32:64] = 1
    reader = _WindowFakeReader(
        [
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
        ]
    )
    mask = _open_fake_mask(monkeypatch, reader)
    aligned = mask.align_to(reference_spacing_um=0.25, reference_dimensions=(128, 64))

    read = aligned.read_region(
        location=(24, 8), target_spacing_um=0.25, target_dimensions=(48, 48)
    )

    # the mask region is the reference raster cropped at the very same coordinates
    np.testing.assert_array_equal(read.labels, reference[8:56, 24:72])
    assert read.read_level == 0
    assert read.read_spacing_um == 4.0
    assert reader.windows == [((1, 0), 0, (4, 4))]


def test_region_read_returns_exact_dimensions_when_the_level_is_within_tolerance(
    monkeypatch,
):
    reader, aligned = _coarse_numbered_mask(monkeypatch)

    # level 0 is 4.0 um: the 4.02 um request is within the 1% level tolerance. Its 4x2
    # pixels span [8, 72.32) x [0, 32.16) reference px, a 5x3 native window.
    read = aligned.read_region(
        location=(8, 0), target_spacing_um=4.02, target_dimensions=(4, 2)
    )

    assert read.read_level == 0
    assert read.read_spacing_um == 4.0
    np.testing.assert_array_equal(
        read.labels, np.array([[0, 1, 2, 3], [8, 9, 10, 11]], dtype=np.uint8)
    )
    assert reader.windows == [((0, 0), 0, (5, 3))]


def _flat_tissue_mask_with_a_stray_label(monkeypatch, *, stray, path):
    """An 8x8, 1.0 um tissue mask under a 0.5 um reference, with one undeclared 7."""
    native = np.zeros((8, 8), dtype=np.uint8)
    native[stray] = 7
    reader = _WindowFakeReader([native])
    mask = _open_fake_mask(monkeypatch, reader, path=path)
    aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(16, 16))
    return reader, aligned


def test_region_read_returns_an_immutable_mask_read(monkeypatch):
    _, aligned = _coarse_numbered_mask(monkeypatch)

    read = aligned.read_region(
        location=(0, 0), target_spacing_um=4.0, target_dimensions=(2, 2)
    )

    assert isinstance(read, MaskRead)
    with pytest.raises(AttributeError):
        read.read_level = 1
    with pytest.raises(ValueError, match="read-only"):
        read.labels[0, 0] = 1


def test_region_read_validates_the_native_window_before_resampling(monkeypatch):
    # The 2.0 um read samples native columns and rows 0 and 2 of a 4x4 window: never (1, 1).
    _, aligned = _flat_tissue_mask_with_a_stray_label(
        monkeypatch, stray=(1, 1), path="stray-label.tif"
    )

    with pytest.raises(ValueError) as excinfo:
        aligned.read_region(
            location=(0, 0), target_spacing_um=2.0, target_dimensions=(2, 2)
        )

    message = str(excinfo.value)
    assert "path=stray-label.tif" in message
    assert "level 0, window 4x4 at (0, 0)" in message
    assert "undeclared label IDs [7]" in message
    assert "declared [0, 1]" in message


def _flat_giant_mask(monkeypatch, reader_type):
    # 20000 x 20000 = 400 Mpx, above the fixed 256 Mpx cap
    reader = reader_type([None], level_dimensions=[(20000, 20000)])
    mask = _open_fake_mask(monkeypatch, reader, path="flat-giant-mask.tif")
    return mask.align_to(reference_spacing_um=0.5, reference_dimensions=(20000, 20000))


def test_region_read_rejects_an_oversized_native_window_before_decoding(monkeypatch):
    class _ExplodingReader(_FakeReader):
        def read_region(self, location, level, size):  # pragma: no cover
            raise AssertionError("the reader was invoked despite the read-size cap")

    aligned = _flat_giant_mask(monkeypatch, _ExplodingReader)

    with pytest.raises(ValueError) as excinfo:
        aligned.read_region(
            location=(0, 0), target_spacing_um=8.0, target_dimensions=(1250, 1250)
        )

    message = str(excinfo.value)
    assert "path=flat-giant-mask.tif" in message
    assert "level 0, where the requested window is 20000x20000 (400 Mpx)" in message
    assert "256 Mpx" in message


def test_region_read_caps_the_window_not_the_level(monkeypatch):
    class _BlankReader(_FakeReader):
        def read_region(self, location, level, size):
            return np.zeros((size[1], size[0]), dtype=np.uint8)

    aligned = _flat_giant_mask(monkeypatch, _BlankReader)

    read = aligned.read_region(
        location=(19996, 19996), target_spacing_um=0.5, target_dimensions=(4, 4)
    )

    np.testing.assert_array_equal(read.labels, np.zeros((4, 4), dtype=np.uint8))


def test_region_read_maps_each_axis_through_its_own_dimension_ratio(monkeypatch):
    # A 7x6 mask of a 100x100 reference, within one pixel of rounding per axis: a mask
    # pixel spans 100 / 7 = 14.29 reference px along x but 100 / 6 = 16.67 along y.
    reader = _WindowFakeReader([np.arange(42, dtype=np.uint8).reshape(6, 7)])
    mask = _open_fake_mask(monkeypatch, reader, labels=_numbered_labels(42))
    aligned = mask.align_to(reference_spacing_um=0.25, reference_dimensions=(100, 100))

    # x = 72 is column 72 / 14.29 = 5.04 -> 5, y = 72 is row 72 / 16.67 = 4.32 -> 4
    read = aligned.read_region(
        location=(72, 72), target_spacing_um=0.25, target_dimensions=(2, 2)
    )

    np.testing.assert_array_equal(read.labels, np.array([[33, 33], [33, 33]], dtype=np.uint8))
    assert reader.windows == [((5, 4), 0, (1, 1))]


# 0.4862 um/px through float32, as a TIFF resolution tag stores it
_FLOAT32_SPACING_UM = 0.4862000048160553


@pytest.mark.parametrize(
    ("reference_spacing_um", "target_spacing_um"),
    [(_FLOAT32_SPACING_UM, 0.4862), (0.4862, _FLOAT32_SPACING_UM)],
    ids=["ratio-a-hair-below-one", "ratio-a-hair-above-one"],
)
def test_region_read_at_a_float_noisy_reference_spacing_is_the_native_crop(
    monkeypatch, reference_spacing_um, target_spacing_um
):
    # The same spacing read by two backends differs by float32 noise: the read must
    # still take every native pixel, not the one before it.
    native = np.arange(64, dtype=np.uint8).reshape(1, 64)
    reader = _WindowFakeReader([native])
    mask = _open_fake_mask(monkeypatch, reader, labels=_numbered_labels(64))
    aligned = mask.align_to(
        reference_spacing_um=reference_spacing_um, reference_dimensions=(64, 1)
    )

    read = aligned.read_region(
        location=(8, 0),
        target_spacing_um=target_spacing_um,
        target_dimensions=(32, 1),
    )

    np.testing.assert_array_equal(read.labels, native[:, 8:40])
    assert reader.windows == [((8, 0), 0, (32, 1))]


def test_region_read_accepts_a_long_request_ending_on_the_canvas_edge_at_a_noisy_spacing(
    monkeypatch,
):
    # 4096 px a hair wider than the reference pixels overshoot the canvas by 4e-5 px,
    # far more than a fixed epsilon, and far less than a pixel
    native = (np.arange(4096) % 2).astype(np.uint8).reshape(1, 4096)
    reader = _WindowFakeReader([native])
    mask = _open_fake_mask(monkeypatch, reader)
    aligned = mask.align_to(reference_spacing_um=0.4862, reference_dimensions=(4096, 1))

    read = aligned.read_region(
        location=(0, 0),
        target_spacing_um=_FLOAT32_SPACING_UM,
        target_dimensions=(4096, 1),
    )

    np.testing.assert_array_equal(read.labels, native)
    assert reader.windows == [((0, 0), 0, (4096, 1))]


def test_region_read_maps_a_location_on_a_level_pixel_boundary_exactly(monkeypatch):
    # An 18 px reference over a 14 px mask: reference x = 9 is exactly mask x = 7, but
    # 9 / (18 / 14) evaluates to 6.999999999999999
    native = np.tile(np.arange(14, dtype=np.uint8), (14, 1))
    reader = _WindowFakeReader([native])
    mask = _open_fake_mask(monkeypatch, reader, labels=_numbered_labels(14))
    aligned = mask.align_to(reference_spacing_um=1.0, reference_dimensions=(18, 18))

    read = aligned.read_region(
        location=(9, 0), target_spacing_um=18 / 14, target_dimensions=(1, 1)
    )

    np.testing.assert_array_equal(read.labels, np.array([[7]], dtype=np.uint8))
    assert reader.windows == [((7, 0), 0, (1, 1))]


@pytest.mark.parametrize(
    ("location", "target_dimensions", "expected"),
    [
        # 4.0 um target pixels are 16 reference px on a 128x64 canvas
        ((0, 0), (8, 4), (8, 4)),
        ((96, 0), (4, 2), (2, 2)),
        ((100, 40), (4, 4), (1, 1)),
        ((128, 0), (2, 2), (0, 2)),
        ((-16, 0), (2, 2), (0, 2)),
    ],
    ids=["inside", "past-right", "past-both-mid-pixel", "at-right-edge", "left"],
)
def test_dimensions_within_canvas_count_the_target_pixels_on_the_canvas(
    monkeypatch, location, target_dimensions, expected
):
    reader, aligned = _coarse_numbered_mask(monkeypatch)

    within = aligned.dimensions_within_canvas(
        location=location, target_spacing_um=4.0, target_dimensions=target_dimensions
    )

    assert within == expected
    assert reader.windows == []


