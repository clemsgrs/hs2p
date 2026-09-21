"""Public behavior of the first-class source-mask domain (``hs2p.mask``, #190).

Real flat PNGs cover the PIL success path; ``_FakeReader`` stands in for pyramidal,
spacing-tagged, wide-dtype and multi-channel sources so every case runs without a native
backend.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import hs2p.mask as mask_mod
import hs2p.wsi.reader as reader_mod
from hs2p.mask import AlignedMask, AnnotationLabels, Mask, MaskRead, TissueLabels

TISSUE = TissueLabels(background=0, tissue=1)


def _write_png(path: Path, labels: np.ndarray) -> Path:
    Image.fromarray(labels).save(path)
    return path


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


def test_mask_types_are_public_from_the_mask_module_and_the_package_root():
    import hs2p

    names = ["AlignedMask", "AnnotationLabels", "Mask", "MaskRead", "TissueLabels"]

    assert mask_mod.__all__ == names
    for name in names:
        assert getattr(hs2p, name) is getattr(mask_mod, name)


def test_tissue_labels_declare_distinct_background_and_tissue_ids():
    labels = TissueLabels(background=0, tissue=255)

    assert labels.background == 0
    assert labels.tissue == 255
    assert labels.ids == frozenset({0, 255})


def test_tissue_labels_are_keyword_only():
    with pytest.raises(TypeError):
        TissueLabels(0, 1)


@pytest.mark.parametrize(
    ("background", "tissue"),
    [(1, 1), (-1, 1), (0, 256), (0, 1.0), (False, True)],
)
def test_tissue_labels_reject_invalid_ids(background, tissue):
    with pytest.raises(ValueError, match="TissueLabels"):
        TissueLabels(background=background, tissue=tissue)


def test_annotation_labels_normalize_the_pixel_mapping_and_expose_all_ids():
    labels = AnnotationLabels(
        pixel_mapping={"background": 0, "tumor": [1, 3], "stroma": 2}
    )

    assert labels.pixel_mapping == {
        "background": (0,),
        "tumor": (1, 3),
        "stroma": (2,),
    }
    assert labels.ids == frozenset({0, 1, 2, 3})


def test_annotation_labels_reject_an_id_claimed_by_two_labels():
    with pytest.raises(
        ValueError, match="'stroma' and 'tumor' both map to 1"
    ):
        AnnotationLabels(pixel_mapping={"tumor": [1, 3], "stroma": 1})


def test_mask_constructor_is_keyword_only(tmp_path):
    path = _write_png(tmp_path / "mask.png", np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(TypeError):
        Mask(path, TISSUE)


def test_mask_rejects_labels_without_declared_semantics(tmp_path):
    path = _write_png(tmp_path / "mask.png", np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="TissueLabels or AnnotationLabels"):
        Mask(path=path, labels={"tissue": 1})


def test_auto_mask_resolves_its_backend_from_its_own_path(tmp_path):
    path = _write_png(tmp_path / "mask.png", np.zeros((2, 2), dtype=np.uint8))

    with Mask(path=path, labels=TISSUE) as mask:
        assert mask.backend == "pil"
        assert mask.path == path
        assert mask.labels == TISSUE


def test_concrete_mask_backend_is_authoritative_and_opens_without_spacing(
    monkeypatch,
):
    opened: list[tuple[str, str, dict]] = []

    def _unexpected_probe(**kwargs):
        raise AssertionError(f"a concrete mask backend was probed: {kwargs}")

    def _fake_open_slide(path, backend, **kwargs):
        opened.append((str(path), backend, kwargs))
        return _FakeReader([np.zeros((2, 2), dtype=np.uint8)])

    monkeypatch.setattr(reader_mod, "_backend_can_open_source", _unexpected_probe)
    monkeypatch.setattr(mask_mod, "open_slide", _fake_open_slide)

    mask = Mask(path="mask.png", labels=TISSUE, backend="openslide")

    assert mask.backend == "openslide"
    assert opened == [("mask.png", "openslide", {"require_spacing": False})]


def test_mask_open_failure_names_path_and_backend(monkeypatch):
    def _failing_open_slide(path, backend, **kwargs):
        raise OSError("not a TIFF")

    monkeypatch.setattr(mask_mod, "open_slide", _failing_open_slide)

    with pytest.raises(
        RuntimeError, match=r"path=broken\.tif.*backend=vips.*not a TIFF"
    ):
        Mask(path="broken.tif", labels=TISSUE, backend="vips")


def test_mask_context_manager_closes_its_reader_once(monkeypatch):
    reader = _FakeReader([np.zeros((2, 2), dtype=np.uint8)])

    with _open_fake_mask(monkeypatch, reader) as mask:
        assert reader.close_count == 0
    mask.close()

    assert reader.close_count == 1


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


def test_align_to_derives_effective_level_spacings_from_the_dimension_ratio(
    monkeypatch,
):
    reader = _FakeReader(_blank_levels((100, 50), (50, 25)))
    mask = _open_fake_mask(monkeypatch, reader)

    aligned = mask.align_to(reference_spacing_um=0.25, reference_dimensions=(1600, 800))

    assert isinstance(aligned, AlignedMask)
    assert aligned.mask is mask
    assert aligned.reference_spacing_um == 0.25
    assert aligned.reference_dimensions == (1600, 800)
    # level 0: 0.25 um x 1600 / 100; level 1 is a 2x downsample of level 0
    assert aligned.level_spacings_um == (4.0, 8.0)


def test_align_to_is_keyword_only():
    with pytest.raises(TypeError):
        Mask.align_to(object(), 0.25, (1600, 800))


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


def test_align_to_rejects_a_mask_whose_shape_does_not_cover_the_reference(
    monkeypatch,
):
    mask = _open_fake_mask(
        monkeypatch, _FakeReader(_blank_levels((100, 50))), path="cropped-mask.tif"
    )

    with pytest.raises(ValueError) as excinfo:
        mask.align_to(reference_spacing_um=0.25, reference_dimensions=(1000, 1000))

    message = str(excinfo.value)
    assert "path=cropped-mask.tif" in message
    assert "mask dimensions 100x50" in message
    assert "reference dimensions 1000x1000" in message
    # 0.25 um x 1000 / 100
    assert "effective spacing 2.5000 um/px" in message
    assert "file spacing" not in message


def test_align_to_is_silent_when_file_spacing_agrees_within_one_percent(
    monkeypatch, caplog
):
    # effective 4.0 um vs tagged 4.02 um: 0.5%
    reader = _FakeReader(_blank_levels((100, 50)), native_spacing=4.02)
    mask = _open_fake_mask(monkeypatch, reader)

    with caplog.at_level("WARNING"):
        aligned = mask.align_to(
            reference_spacing_um=0.25, reference_dimensions=(1600, 800)
        )

    assert aligned.level_spacings_um == (4.0,)
    assert caplog.records == []


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


def test_flat_png_mask_without_spacing_opens_aligns_and_reads(tmp_path, caplog):
    labels = np.array([[0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1]], dtype=np.uint8)
    path = _write_png(tmp_path / "mask.png", labels)

    with caplog.at_level("WARNING"), Mask(path=path, labels=TISSUE) as mask:
        aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(24, 8))
        read = aligned.read_full(target_spacing_um=2.0, target_dimensions=(6, 2))

    assert isinstance(read, MaskRead)
    assert read.read_level == 0
    # 0.5 um x 24 / 6
    assert read.read_spacing_um == 2.0
    assert read.labels.dtype == np.uint8
    np.testing.assert_array_equal(read.labels, labels)
    assert caplog.records == []


def test_untagged_tiff_mask_without_spacing_opens_aligns_and_reads(tmp_path, caplog):
    pytest.importorskip("openslide")
    tifffile = pytest.importorskip("tifffile")
    labels = np.zeros((32, 64), dtype=np.uint8)
    labels[:16, 32:] = 1
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, labels, tile=(32, 32), photometric="minisblack")

    with caplog.at_level("WARNING"), Mask(
        path=path, labels=TISSUE, backend="openslide"
    ) as mask:
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


def test_full_read_selects_its_level_with_the_shared_label_selector(monkeypatch):
    reader, aligned = _three_level_annotation_mask(monkeypatch)
    calls: list[dict] = []
    shared_selector = mask_mod.select_level_for_spacing_read

    def _spy(**kwargs):
        calls.append(kwargs)
        return shared_selector(**kwargs)

    monkeypatch.setattr(mask_mod, "select_level_for_spacing_read", _spy)

    read = aligned.read_full(target_spacing_um=2.0, target_dimensions=(80, 40))

    assert calls == [
        {
            "requested_spacing_um": 2.0,
            "level0_spacing_um": 1.0,
            "level_downsamples": [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)],
            "tolerance": 0.01,
            "content_kind": "label",
        }
    ]
    assert read.read_level == 1
    assert read.read_spacing_um == 2.0
    assert reader.read_levels == [1]


def test_full_read_keeps_a_level_when_the_request_carries_float_noise(monkeypatch):
    reader, aligned = _three_level_annotation_mask(monkeypatch)

    read = aligned.read_full(target_spacing_um=3.9999996, target_dimensions=(40, 20))

    assert read.read_level == 2
    assert read.read_spacing_um == 4.0
    assert reader.read_levels == [2]
    np.testing.assert_array_equal(read.labels, np.full((20, 40), 2, dtype=np.uint8))


def test_full_read_upsamples_level_zero_when_no_level_is_fine_enough(monkeypatch):
    reader = _FakeReader([np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)])
    mask = _open_fake_mask(monkeypatch, reader)
    aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(12, 8))

    read = aligned.read_full(target_spacing_um=1.0, target_dimensions=(6, 4))

    assert read.read_level == 0
    # 0.5 um x 12 / 3
    assert read.read_spacing_um == 2.0
    np.testing.assert_array_equal(
        read.labels,
        np.array(
            [
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )


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


def test_full_read_collapses_identical_replicated_channels(monkeypatch):
    channel = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    native = np.stack([channel, channel, channel], axis=-1)
    mask = _open_fake_mask(monkeypatch, _FakeReader([native]))
    aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(8, 8))

    read = aligned.read_full(target_spacing_um=2.0, target_dimensions=(2, 2))

    assert read.labels.shape == (2, 2)
    np.testing.assert_array_equal(read.labels, channel)


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


def test_full_read_rejects_an_oversized_native_level_before_decoding(monkeypatch):
    class _ExplodingReader(_FakeReader):
        def read_region(self, location, level, size):  # pragma: no cover
            raise AssertionError("the reader was invoked despite the read-size cap")

    # 20000 x 20000 = 400 Mpx, above the fixed 256 Mpx cap
    reader = _ExplodingReader([None], level_dimensions=[(20000, 20000)])
    mask = _open_fake_mask(monkeypatch, reader, path="flat-giant-mask.tif")
    aligned = mask.align_to(
        reference_spacing_um=0.5, reference_dimensions=(20000, 20000)
    )

    with pytest.raises(ValueError) as excinfo:
        aligned.read_full(target_spacing_um=8.0, target_dimensions=(1250, 1250))

    message = str(excinfo.value)
    assert "path=flat-giant-mask.tif" in message
    assert "level 0 at 20000x20000 (400 Mpx)" in message
    assert "256 Mpx" in message
    assert "pyramid" in message
