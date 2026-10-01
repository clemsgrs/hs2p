import numpy as np
import pytest

from hs2p.wsi.backends.common import resolve_padded_read_bounds


def test_noninteger_downsample_maps_the_clipped_origin_onto_the_same_level_pixel():
    # level-0 (4, 4) is level index floor(4 / 3.2) = 1; the location handed to the
    # reader must floor back to 1, and round(3.2) = 3 does not (3 / 3.2 -> 0)
    bounds = resolve_padded_read_bounds(
        location=(4, 4), size=(3, 3), level_dimensions=(20, 20), downsample=3.2
    )

    assert bounds.read_location == (4, 4)
    assert bounds.read_size == (3, 3)
    assert bounds.paste_offset == (0, 0)
    assert int(np.floor(bounds.read_location[0] / 3.2)) == 1


@pytest.mark.parametrize("downsample", [1.0, 2.0, 4.0])
def test_integer_downsamples_are_unchanged(downsample):
    bounds = resolve_padded_read_bounds(
        location=(40, 24), size=(8, 8), level_dimensions=(100, 100), downsample=downsample
    )

    level_x, level_y = int(40 // downsample), int(24 // downsample)
    assert bounds.read_location == (int(level_x * downsample), int(level_y * downsample))
    assert bounds.paste_offset == (0, 0)


def test_negative_origins_are_clipped_and_pasted_with_an_offset():
    bounds = resolve_padded_read_bounds(
        location=(-8, -4), size=(6, 6), level_dimensions=(20, 20), downsample=3.2
    )

    # floor(-8 / 3.2) = -3 and floor(-4 / 3.2) = -2 level pixels lie off the canvas
    assert bounds.read_location == (0, 0)
    assert bounds.paste_offset == (3, 2)
    assert bounds.read_size == (3, 4)
    assert bounds.canvas.shape == (6, 6, 3)


def _write_noninteger_pyramid(path):
    """64x64 and 20x20 RGB levels (downsample 3.2); level-1 pixel (r, c) stores r + c."""
    tifffile = pytest.importorskip("tifffile")

    def coded(size):
        labels = (np.add.outer(np.arange(size), np.arange(size)) % 256).astype(np.uint8)
        return np.repeat(labels[..., None], 3, axis=-1)

    with tifffile.TiffWriter(path) as writer:
        writer.write(coded(64), tile=(16, 16), photometric="rgb", compression="deflate")
        writer.write(
            coded(20), tile=(16, 16), photometric="rgb", subfiletype=1, compression="deflate"
        )
    return path


EXPECTED_3X3_AT_LEVEL1_INDEX_1 = np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6]], dtype=np.uint8)


@pytest.mark.parametrize("backend", ["cucim", "openslide"])
def test_native_region_reads_on_a_3_2x_level_start_at_the_intended_pixel(tmp_path, backend):
    pytest.importorskip({"cucim": "cucim", "openslide": "openslide"}[backend])
    from hs2p.wsi.reader import open_slide

    path = _write_noninteger_pyramid(tmp_path / "pyramid.tif")

    with open_slide(path, backend, require_spacing=False) as reader:
        assert reader.level_count == 2
        assert reader.level_downsamples[1][0] == pytest.approx(3.2)
        region = reader.read_region((4, 4), 1, (3, 3))

    np.testing.assert_array_equal(region[..., 0], EXPECTED_3X3_AT_LEVEL1_INDEX_1)


@pytest.mark.parametrize("backend", ["cucim", "openslide"])
def test_aligned_mask_window_on_a_3_2x_level_starts_at_the_intended_pixel(tmp_path, backend):
    pytest.importorskip({"cucim": "cucim", "openslide": "openslide"}[backend])
    from hs2p.mask import AnnotationLabels, Mask

    path = _write_noninteger_pyramid(tmp_path / "mask.tif")
    labels = AnnotationLabels(pixel_mapping={"coded": list(range(0, 39))})

    with Mask(path=path, labels=labels, backend=backend) as mask:
        aligned = mask.align_to(reference_spacing_um=1.0, reference_dimensions=(64, 64))
        read = aligned.read_region(
            location=(4, 4), target_spacing_um=3.2, target_dimensions=(3, 3)
        )

    assert read.read_level == 1
    np.testing.assert_array_equal(read.labels, EXPECTED_3X3_AT_LEVEL1_INDEX_1)
