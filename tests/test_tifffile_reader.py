from pathlib import Path

import numpy as np
import pytest

from hs2p.configs.models import VALID_BACKENDS
from hs2p.mask import Mask, TissueLabels
from hs2p.wsi.backends.tifffile import PADDING_VALUE, TifffileReader, supports_tifffile_path
from hs2p.wsi.reader import AUTO_BACKEND_ORDER, SlideReader, open_slide

tifffile = pytest.importorskip("tifffile")

FIXTURE_MASK = Path(__file__).resolve().parent / "fixtures" / "input" / "test-mask.tif"


def _write_pyramid(path, level0, *, levels, tile=(16, 16), **kwargs):
    """Write ``level0`` plus ``levels - 1`` nearest-neighbour SubIFD levels."""
    with tifffile.TiffWriter(path) as writer:
        writer.write(level0, tile=tile, subifds=levels - 1, **kwargs)
        for level in range(1, levels):
            step = 2**level
            writer.write(level0[::step, ::step], tile=tile, subfiletype=1, **kwargs)
    return path


def _uint16_labels():
    labels = np.zeros((64, 96), dtype=np.uint16)
    labels[:32, :48] = 1
    labels[32:, 48:] = 300
    labels[10:20, 60:70] = 2
    return labels


def test_tifffile_is_an_explicit_backend_only():
    assert "tifffile" in VALID_BACKENDS
    assert "tifffile" not in AUTO_BACKEND_ORDER
    assert supports_tifffile_path("mask.TIF")
    assert supports_tifffile_path("mask.tiff")
    assert not supports_tifffile_path("mask.png")
    assert not supports_tifffile_path("slide.svs")


def test_reader_satisfies_the_slide_reader_protocol(tmp_path):
    path = tifffile.imwrite(
        tmp_path / "mask.tif", np.zeros((8, 8), dtype=np.uint8), photometric="minisblack"
    )
    with open_slide(tmp_path / "mask.tif", "tifffile", require_spacing=False) as reader:
        assert isinstance(reader, SlideReader)
        assert reader.backend_name == "tifffile"
    del path


def test_real_fixture_pyramid_geometry_and_spacing():
    if not FIXTURE_MASK.is_file():
        pytest.skip("Real fixture TIFF files are not present")
    with TifffileReader(FIXTURE_MASK) as reader:
        assert reader.level_count == 2
        assert reader.level_dimensions == [(864, 800), (432, 400)]
        assert reader.level_downsamples == [(1.0, 1.0), (2.0, 2.0)]
        # 2480.168 pixels per centimeter in the resolution tag
        assert reader.spacing == pytest.approx(10000.0 / 2480.16845703125)
        assert reader.spacings == pytest.approx([reader.spacing, 2 * reader.spacing])


def test_real_fixture_region_reads_match_full_level_crops():
    if not FIXTURE_MASK.is_file():
        pytest.skip("Real fixture TIFF files are not present")
    level1 = tifffile.imread(FIXTURE_MASK, level=1)
    with TifffileReader(FIXTURE_MASK) as reader:
        np.testing.assert_array_equal(reader.read_level(1), level1)
        # level-0 location (500, 300) is (250, 150) on the 2x level; the window crosses
        # the 256 px tile boundary on both axes
        region = reader.read_region((500, 300), 1, (100, 200))
        np.testing.assert_array_equal(region, level1[150:350, 250:350])
        assert region.dtype == np.uint8
        assert region.ndim == 2


def test_uint16_pyramid_reads_keep_stored_values(tmp_path):
    labels = _uint16_labels()
    path = _write_pyramid(tmp_path / "labels.tif", labels, levels=3, photometric="minisblack")

    with TifffileReader(path, require_spacing=False) as reader:
        assert reader.spacing is None
        assert reader.level_dimensions == [(96, 64), (48, 32), (24, 16)]
        assert reader.level_downsamples == [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]

        full = reader.read_level(0)
        assert full.dtype == np.uint16
        np.testing.assert_array_equal(full, labels)
        np.testing.assert_array_equal(reader.read_level(2), labels[::4, ::4])

        region = reader.read_region((40, 20), 0, (40, 30))
        assert region.dtype == np.uint16
        np.testing.assert_array_equal(region, labels[20:50, 40:80])

        region = reader.read_region((40, 20), 1, (20, 15))
        np.testing.assert_array_equal(region, labels[::2, ::2][10:25, 20:40])
        assert set(np.unique(reader.read_level(0)).tolist()) == {0, 1, 2, 300}


def test_region_reads_pad_outside_the_canvas(tmp_path):
    labels = _uint16_labels()
    path = _write_pyramid(tmp_path / "labels.tif", labels, levels=1, photometric="minisblack")

    with TifffileReader(path, require_spacing=False) as reader:
        region = reader.read_region((-8, -4), 0, (24, 20))
        assert region.shape == (20, 24)
        assert (region[:4, :] == PADDING_VALUE).all()
        assert (region[:, :8] == PADDING_VALUE).all()
        np.testing.assert_array_equal(region[4:, 8:], labels[:16, :16])

        region = reader.read_region((88, 56), 0, (16, 16))
        np.testing.assert_array_equal(region[:8, :8], labels[56:, 88:])
        assert (region[8:, :] == PADDING_VALUE).all()
        assert (region[:, 8:] == PADDING_VALUE).all()

        fully_outside = reader.read_region((200, 200), 0, (4, 4))
        assert (fully_outside == PADDING_VALUE).all()


def test_strip_based_pages_read_by_region(tmp_path):
    labels = (np.arange(40 * 30, dtype=np.uint8) % 7).reshape(40, 30)
    path = tifffile.imwrite(
        tmp_path / "strips.tif", labels, rowsperstrip=8, photometric="minisblack"
    )
    del path

    with TifffileReader(tmp_path / "strips.tif", require_spacing=False) as reader:
        np.testing.assert_array_equal(reader.read_region((5, 6), 0, (20, 30)), labels[6:36, 5:25])
        np.testing.assert_array_equal(reader.read_level(0), labels)


def test_rgb_pages_keep_their_samples(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 255, size=(32, 48, 3), dtype=np.uint8)
    tifffile.imwrite(tmp_path / "rgb.tif", rgb, tile=(16, 16), photometric="rgb")

    with TifffileReader(tmp_path / "rgb.tif", require_spacing=False) as reader:
        region = reader.read_region((10, 5), 0, (20, 20))
        assert region.shape == (20, 20, 3)
        np.testing.assert_array_equal(region, rgb[5:25, 10:30])
        assert reader.get_thumbnail((16, 16)).shape[-1] == 3


def test_spacing_comes_from_resolution_tags(tmp_path):
    labels = np.zeros((16, 16), dtype=np.uint8)
    per_cm = tifffile.imwrite(
        tmp_path / "cm.tif", labels, resolution=(20000, 20000), resolutionunit="CENTIMETER"
    )
    per_inch = tifffile.imwrite(
        tmp_path / "inch.tif", labels, resolution=(25400, 25400), resolutionunit="INCH"
    )
    untagged = tifffile.imwrite(tmp_path / "untagged.tif", labels)
    del per_cm, per_inch, untagged

    with TifffileReader(tmp_path / "cm.tif") as reader:
        assert reader.spacing == pytest.approx(0.5)
    with TifffileReader(tmp_path / "inch.tif") as reader:
        assert reader.spacing == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Unable to infer slide spacing"):
        TifffileReader(tmp_path / "untagged.tif")
    with TifffileReader(tmp_path / "untagged.tif", spacing_override=0.25) as reader:
        assert reader.spacing == 0.25


def test_planar_separate_samples_are_rejected(tmp_path):
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    tifffile.imwrite(
        tmp_path / "planar.tif", rgb, photometric="rgb", planarconfig="separate"
    )

    with pytest.raises(ValueError, match="planar-separate"):
        TifffileReader(tmp_path / "planar.tif", require_spacing=False)


def test_mask_reads_uint16_labels_losslessly_through_tifffile(tmp_path):
    labels = np.zeros((32, 64), dtype=np.uint16)
    labels[:16, 32:] = 1
    path = _write_pyramid(tmp_path / "mask.tif", labels, levels=2, photometric="minisblack")

    with Mask(path=path, labels=TissueLabels(background=0, tissue=1), backend="tifffile") as mask:
        assert mask.backend == "tifffile"
        aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(256, 128))
        read = aligned.read_full(target_spacing_um=2.0, target_dimensions=(64, 32))
        # the mask's right half: 16x16 px of the 4.0 um level (mask level 1)
        region = aligned.read_region(
            location=(128, 0), target_spacing_um=4.0, target_dimensions=(16, 16)
        )

    np.testing.assert_array_equal(read.labels, labels.astype(np.uint8))
    assert read.read_level == 0
    np.testing.assert_array_equal(region.labels, labels[::2, ::2][:16, 16:].astype(np.uint8))
    assert region.read_level == 1


def test_mask_rejects_out_of_range_uint16_labels_through_tifffile(tmp_path):
    labels = np.full((32, 64), 257, dtype=np.uint16)
    path = _write_pyramid(tmp_path / "mask.tif", labels, levels=1, photometric="minisblack")

    with Mask(path=path, labels=TissueLabels(background=0, tissue=1), backend="tifffile") as mask:
        aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(256, 128))
        with pytest.raises(ValueError, match=r"outside \[0, 255\]"):
            aligned.read_full(target_spacing_um=2.0, target_dimensions=(64, 32))


def _write_flat_slide(path, size=64):
    from PIL import Image

    Image.fromarray(np.full((size, size, 3), 120, dtype=np.uint8)).save(path)
    return path


def _sixteen_tile_config(mask_backend):
    from hs2p import TilingConfig

    return TilingConfig(
        requested_spacing_um=1.0,
        requested_tile_size_px=16,
        tolerance=0.01,
        overlap=0,
        min_coverage={"tissue": 0.5},
        backend="pil",
        mask_backend=mask_backend,
    )


EXPECTED_SIXTEEN = sorted([(x, y) for x in (0, 16, 32, 48) for y in (0, 16, 32, 48)])


def test_tile_slide_auto_reads_a_uint16_mask_through_tifffile(tmp_path, monkeypatch):
    from hs2p import FilterConfig, SlideSpec, tile_slide
    import hs2p.tiling.orchestration as orchestration_mod

    slide_path = _write_flat_slide(tmp_path / "slide.png")
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(
        mask_path, np.ones((64, 64), dtype=np.uint16), tile=(16, 16), photometric="minisblack"
    )
    events = []
    monkeypatch.setattr(
        orchestration_mod,
        "emit_progress",
        lambda kind, **payload: events.append((kind, payload)),
    )

    result = tile_slide(
        SlideSpec(sample_id="s", image_path=slide_path, mask_path=mask_path, spacing_at_level_0=1.0),
        tiling=_sixteen_tile_config("auto"),
        filtering=FilterConfig(a_t=0, a_h=0),
    )

    assert sorted(zip(result.x.tolist(), result.y.tolist())) == EXPECTED_SIXTEEN
    assert np.unique(result.tissue_mask).tolist() == [255]
    assert result.mask_backend == "tifffile"
    assert result.requested_mask_backend == "auto"
    selected = [payload for kind, payload in events if kind == "mask_backend.selected"]
    assert len(selected) == 1
    assert selected[0]["backend"] == "tifffile"
    assert "16-bit unsigned integer samples" in selected[0]["reason"]


def test_tile_slide_auto_opens_an_untagged_8bit_mask_with_a_native_reader(tmp_path):
    if not any(
        __import__("importlib").util.find_spec(module) is not None
        for module in ("cucim", "openslide")
    ):
        pytest.skip("no native reader installed for the auto chain")
    from hs2p import FilterConfig, SlideSpec, tile_slide

    slide_path = _write_flat_slide(tmp_path / "slide.png")
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(
        mask_path, np.ones((64, 64), dtype=np.uint8), tile=(16, 16), photometric="minisblack"
    )

    result = tile_slide(
        SlideSpec(sample_id="s", image_path=slide_path, mask_path=mask_path, spacing_at_level_0=1.0),
        tiling=_sixteen_tile_config("auto"),
        filtering=FilterConfig(a_t=0, a_h=0),
    )

    assert sorted(zip(result.x.tolist(), result.y.tolist())) == EXPECTED_SIXTEEN
    assert result.mask_backend in {"cucim", "openslide"}


def _write_min_is_white_pair(tmp_path, *, tissue=255):
    """The same stored 0/``tissue`` raster under both grayscale photometrics."""
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[:32, :] = tissue
    paths = {}
    for photometric in ("miniswhite", "minisblack"):
        paths[photometric] = tmp_path / f"{photometric}.tif"
        tifffile.imwrite(
            paths[photometric], labels, tile=(16, 16), photometric=photometric
        )
    return labels, paths


def _read_full_labels(path, backend):
    with Mask(path=path, labels=TissueLabels(background=0, tissue=255), backend=backend) as mask:
        aligned = mask.align_to(reference_spacing_um=0.5, reference_dimensions=(64, 64))
        return aligned.read_full(target_spacing_um=0.5, target_dimensions=(64, 64)).labels


def test_mask_reads_min_is_white_labels_unchanged_through_tifffile(tmp_path):
    labels, paths = _write_min_is_white_pair(tmp_path)

    np.testing.assert_array_equal(_read_full_labels(paths["miniswhite"], "tifffile"), labels)
    np.testing.assert_array_equal(_read_full_labels(paths["minisblack"], "tifffile"), labels)


def test_native_control_matches_tifffile_for_min_is_black_only(tmp_path):
    """cuCIM inverts min-is-white on decode (stored 255 -> 0), so the guard refuses it,
    while the min-is-black control decodes to the stored values through both readers."""
    pytest.importorskip("cucim")
    labels, paths = _write_min_is_white_pair(tmp_path)

    np.testing.assert_array_equal(_read_full_labels(paths["minisblack"], "cucim"), labels)
    with pytest.raises(ValueError, match="which the cucim backend inverts"):
        _read_full_labels(paths["miniswhite"], "cucim")


def test_mask_reads_a_mixed_width_pyramid_losslessly_through_tifffile(tmp_path):
    """A uint8 root with a uint16 reduced level: tifffile serves the root and resamples
    it, so a reduced-level request still returns the stored class."""
    path = tmp_path / "mixed.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(
            np.ones((64, 64), dtype=np.uint8), tile=(16, 16), photometric="minisblack"
        )
        writer.write(
            np.ones((32, 32), dtype=np.uint16),
            tile=(16, 16),
            photometric="minisblack",
            subfiletype=1,
        )

    with Mask(path=path, labels=TissueLabels(background=0, tissue=1), backend="tifffile") as mask:
        aligned = mask.align_to(reference_spacing_um=1.0, reference_dimensions=(64, 64))
        read = aligned.read_full(target_spacing_um=2.0, target_dimensions=(32, 32))

    assert read.read_level == 0
    np.testing.assert_array_equal(read.labels, np.ones((32, 32), dtype=np.uint8))


def _native_reader_installed():
    import importlib.util

    return any(
        importlib.util.find_spec(module) is not None for module in ("cucim", "openslide")
    )


def test_tile_slide_auto_routes_a_min_is_white_mask_to_tifffile(tmp_path, monkeypatch):
    """Stored 0/1 under min-is-white: auto reads it through tifffile and finds the same
    tiles as the min-is-black control through a native reader (a native decode of the
    min-is-white file inverts it to 255/254, which no longer names any tissue)."""
    from hs2p import FilterConfig, SlideSpec, tile_slide
    import hs2p.tiling.orchestration as orchestration_mod

    slide_path = _write_flat_slide(tmp_path / "slide.png")
    labels, paths = _write_min_is_white_pair(tmp_path, tissue=1)
    events = []
    monkeypatch.setattr(
        orchestration_mod,
        "emit_progress",
        lambda kind, **payload: events.append((kind, payload)),
    )

    def _tile(mask_path):
        result = tile_slide(
            SlideSpec(
                sample_id="s", image_path=slide_path, mask_path=mask_path, spacing_at_level_0=1.0
            ),
            tiling=_sixteen_tile_config("auto"),
            filtering=FilterConfig(a_t=0, a_h=0),
        )
        return result, sorted(zip(result.x.tolist(), result.y.tolist()))

    # the top half is tissue: eight 16 px tiles
    expected = sorted([(x, y) for x in (0, 16, 32, 48) for y in (0, 16)])
    inverted, inverted_tiles = _tile(paths["miniswhite"])
    assert inverted.mask_backend == "tifffile"
    assert inverted_tiles == expected
    selected = [payload for kind, payload in events if kind == "mask_backend.selected"]
    assert [payload["backend"] for payload in selected] == ["tifffile"]
    assert "min-is-white, which a display reader inverts" in selected[0]["reason"]

    if _native_reader_installed():
        control, control_tiles = _tile(paths["minisblack"])
        assert control.mask_backend in {"cucim", "openslide"}
        assert control_tiles == expected
        np.testing.assert_array_equal(control.tissue_mask, inverted.tissue_mask)


def test_tile_slide_auto_routes_a_mixed_width_pyramid_to_tifffile(tmp_path):
    """A uint8 root with a uint16 reduced level (class 1 everywhere) yields the full
    sixteen tiles through tifffile; a native reduced-level decode returned zeros."""
    from hs2p import FilterConfig, SlideSpec, tile_slide

    slide_path = _write_flat_slide(tmp_path / "slide.png")
    mask_path = tmp_path / "mixed.tif"
    with tifffile.TiffWriter(mask_path) as writer:
        writer.write(np.ones((64, 64), dtype=np.uint8), tile=(16, 16), photometric="minisblack")
        writer.write(
            np.ones((32, 32), dtype=np.uint16),
            tile=(16, 16),
            photometric="minisblack",
            subfiletype=1,
        )

    result = tile_slide(
        SlideSpec(sample_id="s", image_path=slide_path, mask_path=mask_path, spacing_at_level_0=1.0),
        tiling=_sixteen_tile_config("auto"),
        filtering=FilterConfig(a_t=0, a_h=0),
    )

    assert result.mask_backend == "tifffile"
    assert sorted(zip(result.x.tolist(), result.y.tolist())) == EXPECTED_SIXTEEN
