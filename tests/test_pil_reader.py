from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import hs2p.wsi.reader as reader_mod
from hs2p.wsi.backends import pil as pil_mod


@pytest.mark.parametrize("suffix", [".png", ".JPG", ".JpEg"])
def test_auto_routes_flat_raster_suffixes_only_to_pil(monkeypatch, suffix):
    def _unexpected_probe(**kwargs):
        raise AssertionError(f"auto probed a backend for flat input: {kwargs}")

    monkeypatch.setattr(reader_mod, "_backend_can_open_source", _unexpected_probe)

    selection = reader_mod.resolve_backend(
        "auto",
        wsi_path=Path(f"benchmark-image{suffix}"),
    )

    assert selection.backend == "pil"
    assert selection.tried == ("pil",)
    assert "PIL" in (selection.reason or "")


def test_pil_reader_reports_one_level_geometry_and_explicit_spacing(tmp_path):
    path = tmp_path / "labels.png"
    Image.fromarray(np.zeros((4, 6), dtype=np.uint8), mode="L").save(path)

    with reader_mod.open_slide(
        path,
        backend="pil",
        spacing_override=0.375,
    ) as slide:
        assert slide.native_spacing is None
        assert slide.spacing == 0.375
        assert slide.spacings == [0.375]
        assert slide.level_count == 1
        assert slide.level_dimensions == [(6, 4)]
        assert slide.level_downsamples == [(1.0, 1.0)]


def test_pil_reader_requires_explicit_level_zero_spacing(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8), mode="RGB").save(path)

    with pytest.raises(
        ValueError,
        match=r"Unable to infer slide spacing.*backend=pil.*spacing_at_level_0",
    ):
        reader_mod.open_slide(path, backend="pil")


def test_auto_rejects_one_pixel_above_the_pil_ceiling_before_decode(
    monkeypatch, tmp_path
):
    path = tmp_path / "too-large.PNG"
    Image.fromarray(np.zeros((2, 3), dtype=np.uint8), mode="L").save(path)
    monkeypatch.setattr(pil_mod, "PIL_MAX_IMAGE_PIXELS", 5)

    def _unexpected_decode(self, *args, **kwargs):
        raise AssertionError("oversized image pixel data was decoded")

    def _unexpected_probe(**kwargs):
        raise AssertionError(f"auto probed an alternative backend: {kwargs}")

    monkeypatch.setattr(Image.Image, "load", _unexpected_decode)
    monkeypatch.setattr(reader_mod, "_backend_can_open_source", _unexpected_probe)

    with pytest.raises(ValueError) as caught:
        reader_mod.open_slide(
            path,
            backend="auto",
            spacing_override=0.5,
        )

    message = str(caught.value)
    assert str(path) in message
    assert "dimensions=3x2" in message
    assert "pixel_count=6" in message
    assert "ceiling=5" in message
    assert "another backend" not in message.lower()


@pytest.mark.parametrize(
    ("mode", "source", "expected"),
    [
        (
            "RGB",
            np.array(
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[7, 8, 9], [10, 11, 12]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[7, 8, 9], [10, 11, 12]],
                ],
                dtype=np.uint8,
            ),
        ),
        (
            "RGBA",
            np.array(
                [
                    [[1, 2, 3, 40], [4, 5, 6, 50]],
                    [[7, 8, 9, 60], [10, 11, 12, 70]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[7, 8, 9], [10, 11, 12]],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
)
def test_rgb_like_level_reads_are_exact_rgb_uint8(
    tmp_path, mode, source, expected
):
    path = tmp_path / f"{mode.lower()}.png"
    Image.fromarray(source, mode=mode).save(path)

    with reader_mod.open_slide(
        path,
        backend="pil",
        spacing_override=0.5,
    ) as slide:
        actual = slide.read_level(0)

    assert actual.dtype == np.uint8
    assert actual.shape == (2, 2, 3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ["L", "P"])
def test_grayscale_and_palette_level_reads_preserve_label_values(tmp_path, mode):
    labels = np.array([[0, 3, 17], [255, 8, 1]], dtype=np.uint8)
    image = Image.fromarray(labels, mode=mode)
    if mode == "P":
        palette = []
        for index in range(256):
            palette.extend(((index * 13) % 256, (index * 29) % 256, 255 - index))
        image.putpalette(palette)
    path = tmp_path / f"labels-{mode}.png"
    image.save(path)

    with reader_mod.open_slide(
        path,
        backend="pil",
        spacing_override=0.5,
    ) as slide:
        actual = slide.read_level(0)

    assert actual.dtype == np.uint8
    assert actual.shape == (2, 3)
    np.testing.assert_array_equal(actual, labels)


def test_uint16_grayscale_reads_preserve_label_values_and_dtype(tmp_path):
    labels = np.array([[0, 300, 65_535], [42, 1_024, 7]], dtype=np.uint16)
    path = tmp_path / "labels-uint16.png"
    Image.fromarray(labels).save(path)
    expected_region = np.full((3, 5), 255, dtype=np.uint16)
    expected_region[1:3, 1:4] = labels

    with reader_mod.open_slide(
        path,
        backend="pil",
        spacing_override=0.5,
    ) as slide:
        level = slide.read_level(0)
        region = slide.read_region((-1, -1), 0, (5, 3))

    assert level.dtype == np.uint16
    assert level.shape == (2, 3)
    np.testing.assert_array_equal(level, labels)
    np.testing.assert_array_equal(region, expected_region)


def test_one_bit_grayscale_reads_return_integer_label_values(tmp_path):
    labels = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    path = tmp_path / "labels-one-bit.png"
    Image.fromarray(labels.astype(bool)).save(path)

    with reader_mod.open_slide(
        path,
        backend="pil",
        spacing_override=0.5,
    ) as slide:
        actual = slide.read_level(0)

    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, labels)


def test_pil_region_read_uses_white_out_of_bounds_padding(tmp_path):
    source = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "rgb.png"
    Image.fromarray(source, mode="RGB").save(path)
    expected = np.full((4, 5, 3), 255, dtype=np.uint8)
    expected[1:3, 1:4] = source

    with reader_mod.open_slide(
        path,
        backend="pil",
        spacing_override=0.5,
    ) as slide:
        actual = slide.read_region((-1, -1), 0, (5, 4))

    np.testing.assert_array_equal(actual, expected)
