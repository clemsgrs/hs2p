from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import hs2p.wsi.reader as reader_mod
import hs2p.tiling.mask as mask_mod
from hs2p.configs import TilingConfig
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


@pytest.mark.parametrize(
    ("filename", "image_format"),
    [("image.png", "PNG"), ("image.jpg", "JPEG"), ("image.JPEG", "JPEG")],
)
def test_auto_opens_small_flat_rasters_without_native_backend_probes(
    monkeypatch, tmp_path, filename, image_format
):
    path = tmp_path / filename
    Image.fromarray(np.zeros((3, 5, 3), dtype=np.uint8), mode="RGB").save(
        path,
        format=image_format,
    )

    def _unexpected_probe(**kwargs):
        raise AssertionError(f"auto probed a native backend: {kwargs}")

    monkeypatch.setattr(reader_mod, "_backend_can_open_source", _unexpected_probe)

    with reader_mod.open_slide(
        path,
        backend="auto",
        spacing_override=0.25,
    ) as slide:
        assert slide.backend_name == "pil"
        assert slide.dimensions == (5, 3)


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


def test_pil_reader_uses_the_project_owned_pixel_ceiling():
    assert pil_mod.PIL_MAX_IMAGE_PIXELS == 89_478_485


def test_pil_reader_accepts_an_image_at_the_project_pixel_ceiling(
    monkeypatch, tmp_path
):
    path = tmp_path / "at-limit.png"
    Image.fromarray(np.zeros((2, 3), dtype=np.uint8), mode="L").save(path)
    monkeypatch.setattr(pil_mod, "PIL_MAX_IMAGE_PIXELS", 6)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    with reader_mod.open_slide(
        path,
        backend="pil",
        spacing_override=0.5,
    ) as slide:
        assert slide.dimensions == (3, 2)

    assert Image.MAX_IMAGE_PIXELS == 1


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


def test_auto_mask_oversize_error_does_not_recommend_another_backend(
    monkeypatch, tmp_path
):
    path = tmp_path / "too-large-mask.png"
    Image.fromarray(np.zeros((2, 3), dtype=np.uint8), mode="L").save(path)
    monkeypatch.setattr(pil_mod, "PIL_MAX_IMAGE_PIXELS", 5)

    with pytest.raises(ValueError) as caught:
        reader_mod.open_mask_reader(path, mask_backend="auto")

    message = str(caught.value)
    assert "backend=pil" in message
    assert "ceiling=5" in message
    assert "another" not in message.lower()
    assert "select" not in message.lower()


def test_tiling_mask_oversize_error_does_not_recommend_another_backend(
    monkeypatch, tmp_path
):
    path = tmp_path / "too-large-tiling-mask.png"
    Image.fromarray(np.zeros((2, 3), dtype=np.uint8), mode="L").save(path)
    monkeypatch.setattr(pil_mod, "PIL_MAX_IMAGE_PIXELS", 5)

    with pytest.raises(ValueError) as caught:
        mask_mod.load_precomputed_tissue_mask(
            mask_path=path,
            slide=object(),
            seg_level=0,
            tissue_value=1,
            mask_backend="auto",
        )

    message = str(caught.value)
    assert "backend=pil" in message
    assert "ceiling=5" in message
    assert "another" not in message.lower()
    assert "select" not in message.lower()


def test_corrupt_flat_raster_fails_with_pil_context_and_no_fallback(
    monkeypatch, tmp_path
):
    path = tmp_path / "corrupt.jpeg"
    path.write_bytes(b"not an image")

    def _unexpected_probe(**kwargs):
        raise AssertionError(f"auto probed an alternative backend: {kwargs}")

    monkeypatch.setattr(reader_mod, "_backend_can_open_source", _unexpected_probe)

    with pytest.raises(RuntimeError) as caught:
        reader_mod.open_slide(
            path,
            backend="auto",
            spacing_override=0.5,
        )

    message = str(caught.value)
    assert "PIL backend failed to open" in message
    assert str(path) in message
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


def test_pil_thumbnail_fits_inside_requested_size_as_rgb_uint8(tmp_path):
    source = np.full((2, 4, 3), [12, 34, 56], dtype=np.uint8)
    path = tmp_path / "thumbnail.png"
    Image.fromarray(source, mode="RGB").save(path)

    with reader_mod.open_slide(
        path,
        backend="pil",
        spacing_override=0.5,
    ) as slide:
        thumbnail = slide.get_thumbnail((2, 2))

    assert thumbnail.dtype == np.uint8
    assert thumbnail.shape == (1, 2, 3)
    np.testing.assert_array_equal(
        thumbnail,
        np.full((1, 2, 3), [12, 34, 56], dtype=np.uint8),
    )


def test_pil_reader_conforms_to_protocol_and_context_manager_closes_it(tmp_path):
    path = tmp_path / "context.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(path)

    with reader_mod.open_slide(
        path,
        backend="pil",
        spacing_override=0.5,
    ) as slide:
        assert isinstance(slide, reader_mod.SlideReader)
        reader = slide

    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.read_level(0)


def test_palette_region_read_preserves_indices_and_single_channel_shape(tmp_path):
    labels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    image = Image.fromarray(labels, mode="P")
    image.putpalette([channel for index in range(256) for channel in (index, 0, 0)])
    path = tmp_path / "palette.png"
    image.save(path)
    expected = np.full((3, 3), 255, dtype=np.uint8)
    expected[:2, :2] = labels[:, 1:]

    with reader_mod.open_slide(
        path,
        backend="pil",
        spacing_override=0.5,
    ) as slide:
        actual = slide.read_region((1, 0), 0, (3, 3))

    np.testing.assert_array_equal(actual, expected)


def test_pil_is_a_supported_tiling_backend():
    config = TilingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        min_coverage={"tissue": 0.1},
        backend="pil",
        mask_backend="pil",
    )

    assert config.backend == "pil"
    assert config.mask_backend == "pil"


def test_flat_raster_resolution_records_requested_and_resolved_provenance():
    resolved = reader_mod.resolve_backends(
        requested_slide_backend="auto",
        requested_mask_backend=None,
        wsi_path=Path("benchmark.png"),
    )

    assert resolved.requested_slide_backend == "auto"
    assert resolved.slide.backend == "pil"
    assert resolved.slide.tried == ("pil",)
