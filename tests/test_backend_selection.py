import warnings
from pathlib import Path

import numpy as np
import pytest

import hs2p.api as api_mod
import hs2p.preprocessing as preprocessing_mod
import hs2p.tiling.orchestration as orchestration_mod
import hs2p.wsi.backend as backend_mod
import hs2p.wsi.reader as reader_mod


def _opens_if(condition: bool) -> "reader_mod._ProbeOutcome":
    return (
        reader_mod._ProbeOutcome.USABLE
        if condition
        else reader_mod._ProbeOutcome.CANNOT_OPEN
    )


class _FakeReader:
    level_count = 2
    level_dimensions = [(512, 512), (256, 256)]

    def __init__(self, *, decodes: bool, undecodable_levels: tuple[int, ...] = ()):
        self._decodes = decodes
        self._undecodable_levels = undecodable_levels
        self.reads: list[tuple] = []
        self.closed = False

    def read_region(self, location, level, size):
        self.reads.append((location, level, size))
        if not self._decodes or level in self._undecodable_levels:
            raise RuntimeError("requested compression method is not configured")
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def _make_tiling_result(sample_id: str = "slide-1") -> preprocessing_mod.TilingResult:
    return preprocessing_mod.TilingResult(
        tiles=preprocessing_mod.TileGeometry(
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tissue_fractions=np.array([0.0], dtype=np.float32),
            tile_index=np.array([0], dtype=np.int32),
            requested_tile_size_px=256,
            requested_spacing_um=0.5,
            read_level=0,
            read_tile_size_px=256,
            read_spacing_um=0.5,
            tile_size_lv0=256,
            is_within_tolerance=True,
            base_spacing_um=0.5,
            slide_dimensions=[1000, 1000],
            level_downsamples=[1.0],
            overlap=0.0,
            min_tissue_fraction=0.1,
        ),
        sample_id=sample_id,
        image_path=Path("slide.svs"),
        backend="cucim",
        requested_backend="cucim",
        tolerance=0.05,
        step_px_lv0=256,
        tissue_method="hsv",
        requested_seg_downsample=64,
        seg_downsample=64,
        seg_level=0,
        seg_spacing_um=0.5,
        seg_sthresh=8,
        seg_sthresh_up=255,
        seg_mthresh=7,
        seg_close=4,
        ref_tile_size_px=16,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def _cucim_auto_backend_selection(
    requested_backend: str, *, wsi_path: Path, mask_path=None
) -> backend_mod.BackendSelection:
    del requested_backend, wsi_path, mask_path
    return backend_mod.BackendSelection(
        backend="cucim",
        reason="selected cuCIM for auto backend",
        tried=("cucim",),
    )


def _cucim_auto_resolve_backends(
    *,
    requested_slide_backend: str,
    requested_mask_backend,
    wsi_path: Path,
    mask_path=None,
    slide_spacing_override=None,
) -> backend_mod.ResolvedBackends:
    del wsi_path, slide_spacing_override
    slide = _cucim_auto_backend_selection("auto", wsi_path=Path("slide.svs"))
    mask = (
        None
        if mask_path is None
        else _cucim_auto_backend_selection("auto", wsi_path=Path(mask_path))
    )
    return backend_mod.ResolvedBackends(
        slide=slide,
        mask=mask,
        requested_slide_backend=requested_slide_backend,
        requested_mask_backend=None if mask_path is None else requested_mask_backend,
    )


def test_auto_mask_fallback_uses_shared_priority_independently(monkeypatch):
    calls: list[tuple[str, str, float | None]] = []

    def _fake_can_open_source(
        *,
        source_path: str,
        companion_path: str | None,
        backend: str,
        spacing_override: float | None = None,
        require_spacing: bool = True,
    ):
        del companion_path
        calls.append((source_path, backend, spacing_override))
        if source_path.endswith("slide.svs"):
            return _opens_if(backend == "cucim")
        return _opens_if(backend == "asap")

    monkeypatch.setattr(reader_mod, "_probe_backend", _fake_can_open_source)

    resolved = backend_mod.resolve_backends(
        requested_slide_backend="auto",
        requested_mask_backend="auto",
        wsi_path=Path("slide.svs"),
        mask_path=Path("mask.tif"),
        slide_spacing_override=0.5,
    )

    assert resolved.slide_backend == "cucim"
    assert resolved.mask_backend == "asap"
    assert resolved.requested_slide_backend == "auto"
    assert resolved.requested_mask_backend == "auto"
    assert calls == [
        ("slide.svs", "cucim", 0.5),
        ("mask.tif", "cucim", None),
        ("mask.tif", "vips", None),
        ("mask.tif", "openslide", None),
        ("mask.tif", "asap", None),
    ]


def test_auto_open_uses_spacing_override_for_probe_and_warns_only_on_selected_open(
    monkeypatch,
):
    opened_with: list[float | None] = []

    def _fake_opener(
        path, *, spacing_override=None, gpu_decode=False, require_spacing=True
    ):
        assert gpu_decode is False
        opened_with.append(spacing_override)
        if spacing_override is None:
            raise ValueError("missing native spacing")
        warnings.warn(
            "Slide spacing override conflict: "
            f"path={path}, native=0.5, supplied={spacing_override}, backend=cucim; "
            "using the supplied level-0 spacing.",
            UserWarning,
        )
        return _FakeReader(decodes=True)

    monkeypatch.setattr(
        reader_mod,
        "_BACKENDS",
        {
            "cucim": reader_mod._BackendSpec(
                opener=_fake_opener,
                supports_path=lambda path: True,
            ),
        },
    )
    reader_mod._probe_backend.cache_clear()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader = reader_mod.open_slide(
            Path("missing-spacing.svs"),
            backend="auto",
            spacing_override=0.25,
        )
        reader.close()

    assert opened_with == [0.25, 0.25]
    assert len(caught) == 1
    assert "path=missing-spacing.svs" in str(caught[0].message)
    assert "backend=cucim" in str(caught[0].message)


def test_auto_openability_probe_honors_require_spacing(monkeypatch):
    def _fake_opener(path, *, spacing_override=None, require_spacing=True):
        del path, spacing_override
        if require_spacing:
            raise ValueError("missing native spacing")
        return _FakeReader(decodes=True)

    monkeypatch.setattr(
        reader_mod,
        "_BACKENDS",
        {
            "openslide": reader_mod._BackendSpec(
                opener=_fake_opener,
                supports_path=lambda path: True,
            ),
        },
    )
    monkeypatch.setattr(reader_mod, "AUTO_BACKEND_ORDER", ("openslide",))
    reader_mod._probe_backend.cache_clear()

    with pytest.raises(RuntimeError, match="Unable to open untagged-mask.tif"):
        reader_mod.resolve_backend("auto", wsi_path=Path("untagged-mask.tif"))

    selection = reader_mod.resolve_backend(
        "auto",
        wsi_path=Path("untagged-mask.tif"),
        require_spacing=False,
    )

    assert selection.backend == "openslide"


def test_open_slide_forwards_gpu_decode_to_cucim_only(monkeypatch):
    """Only the cuCIM reader takes ``gpu_decode``; every other reader opens without it."""
    opened: dict[str, dict] = {}

    class _Reader:
        def close(self):
            return None

    def _cucim(path, *, spacing_override=None, gpu_decode=False, require_spacing=True):
        opened["cucim"] = dict(
            spacing_override=spacing_override,
            gpu_decode=gpu_decode,
            require_spacing=require_spacing,
        )
        return _Reader()

    def _openslide(path, *, spacing_override=None, require_spacing=True):
        opened["openslide"] = dict(
            spacing_override=spacing_override, require_spacing=require_spacing
        )
        return _Reader()

    monkeypatch.setitem(
        reader_mod._BACKENDS, "cucim", reader_mod._BackendSpec(_cucim, lambda p: True)
    )
    monkeypatch.setitem(
        reader_mod._BACKENDS,
        "openslide",
        reader_mod._BackendSpec(_openslide, lambda p: True),
    )

    for backend in ("cucim", "openslide"):
        reader_mod.open_slide(
            Path("slide.svs"),
            backend=backend,
            spacing_override=0.25,
            gpu_decode=True,
            require_spacing=False,
        ).close()

    assert opened == {
        "cucim": {"spacing_override": 0.25, "gpu_decode": True, "require_spacing": False},
        "openslide": {"spacing_override": 0.25, "require_spacing": False},
    }


def test_tile_slide_uses_resolved_backend_for_hash_and_result(monkeypatch):
    captured: dict[str, str] = {}

    def _fake_preprocess_slide(**kwargs):
        captured["backend"] = kwargs["backend"]
        return _make_tiling_result()

    monkeypatch.setattr(orchestration_mod, "resolve_backends", _cucim_auto_resolve_backends)
    monkeypatch.setattr(orchestration_mod, "_preprocess_slide", _fake_preprocess_slide)

    result = api_mod.tile_slide(
        api_mod.SlideSpec(sample_id="slide-1", image_path=Path("slide.svs")),
        tiling=api_mod.TilingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=256,
            tolerance=0.05,
            overlap=0.0,
            min_coverage={"tissue": 0.1},
            backend="auto",
        ),
        segmentation=api_mod.SegmentationConfig(method="hsv", downsample=64, sthresh=8, sthresh_up=255, mthresh=7, close=4),
        filtering=api_mod.FilterConfig(
            ref_tile_size=16,
            a_t=4,
            a_h=2,
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
        num_workers=1,
    )

    assert result.backend == "cucim"
    assert captured["backend"] == "cucim"


def test_effective_backend_resolution_forwards_level0_spacing_override(monkeypatch):
    captured: dict[str, float | None] = {}

    def _fake_resolve_backends(
        *,
        requested_slide_backend,
        requested_mask_backend,
        wsi_path,
        mask_path=None,
        slide_spacing_override=None,
    ):
        del requested_slide_backend, requested_mask_backend, wsi_path, mask_path
        captured["slide_spacing_override"] = slide_spacing_override
        return backend_mod.ResolvedBackends(
            slide=backend_mod.BackendSelection(backend="cucim", tried=("cucim",)),
            mask=None,
            requested_slide_backend="auto",
            requested_mask_backend=None,
        )

    monkeypatch.setattr(orchestration_mod, "resolve_backends", _fake_resolve_backends)

    orchestration_mod._resolve_effective_backends(
        api_mod.SlideSpec(
            sample_id="missing-spacing",
            image_path=Path("missing-spacing.svs"),
            spacing_at_level_0=0.25,
        ),
        api_mod.TilingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=256,
            tolerance=0.05,
            overlap=0.0,
            min_coverage={"tissue": 0.1},
            backend="auto",
        ),
        emit=False,
    )

    assert captured == {"slide_spacing_override": 0.25}


def test_mask_role_probe_never_requires_spacing_while_slide_role_does(monkeypatch):
    calls: list[tuple[str, str, bool]] = []

    def _fake_can_open_source(
        *,
        source_path: str,
        companion_path: str | None,
        backend: str,
        spacing_override: float | None = None,
        require_spacing: bool = True,
    ):
        del companion_path, spacing_override
        calls.append((source_path, backend, require_spacing))
        return _opens_if(backend == "cucim")

    monkeypatch.setattr(reader_mod, "_probe_backend", _fake_can_open_source)

    resolved = reader_mod.resolve_backends(
        requested_slide_backend="auto",
        requested_mask_backend="auto",
        wsi_path=Path("slide.svs"),
        mask_path=Path("untagged-mask.tif"),
    )

    assert resolved.slide_backend == "cucim"
    assert resolved.mask_backend == "cucim"
    # the mask's spacing comes from its dimensions, so the probe must accept an
    # untagged TIFF; the slide still needs spacing
    assert calls == [
        ("slide.svs", "cucim", True),
        ("untagged-mask.tif", "cucim", False),
    ]


def test_mask_role_auto_selects_tifffile_from_a_lossy_header(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(mask_path, np.ones((8, 8), dtype=np.uint16), photometric="minisblack")
    monkeypatch.setattr(
        reader_mod,
        "_probe_backend",
        lambda **kwargs: pytest.fail("the openability chain must not run"),
    )

    selection = reader_mod.resolve_mask_backend("auto", mask_path=mask_path)

    assert selection.backend == "tifffile"
    assert selection.tried == ("tifffile",)
    assert "16-bit unsigned integer samples" in selection.reason
    assert "which a display reader rescales to 8 bits" in selection.reason


def test_mask_role_auto_selects_tifffile_for_a_min_is_white_header(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(mask_path, np.full((8, 8), 255, dtype=np.uint8), photometric="miniswhite")

    selection = reader_mod.resolve_mask_backend("auto", mask_path=mask_path)

    assert selection.backend == "tifffile"
    assert "photometric min-is-white, which a display reader inverts" in selection.reason


def test_mask_role_auto_selects_tifffile_for_a_lossy_reduced_level(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    mask_path = tmp_path / "mask.tif"
    with tifffile.TiffWriter(mask_path) as writer:
        writer.write(np.ones((64, 64), dtype=np.uint8), tile=(16, 16), photometric="minisblack")
        writer.write(
            np.ones((32, 32), dtype=np.uint16),
            tile=(16, 16),
            photometric="minisblack",
            subfiletype=1,
        )

    selection = reader_mod.resolve_mask_backend("auto", mask_path=mask_path)

    assert selection.backend == "tifffile"
    assert "directory 1 stores 16-bit unsigned integer samples" in selection.reason


def test_mask_role_auto_keeps_the_chain_for_an_8bit_header(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(mask_path, np.ones((8, 8), dtype=np.uint8), photometric="minisblack")
    probed: list[str] = []

    def _fake_can_open_source(*, backend: str, **kwargs):
        probed.append(backend)
        return _opens_if(backend == "openslide")

    monkeypatch.setattr(reader_mod, "_probe_backend", _fake_can_open_source)

    selection = reader_mod.resolve_mask_backend("auto", mask_path=mask_path)

    assert selection.backend == "openslide"
    assert probed == ["cucim", "vips", "openslide"]


def test_mask_role_explicit_backend_is_authoritative_over_the_header(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(mask_path, np.ones((8, 8), dtype=np.uint16), photometric="minisblack")

    selection = reader_mod.resolve_mask_backend("cucim", mask_path=mask_path)

    assert selection == reader_mod.BackendSelection(backend="cucim", reason=None, tried=("cucim",))


def _some_auto_backend_decodes(path) -> bool:
    """Whether any installed reader in the ``auto`` chain decodes ``path`` when asked
    for explicitly (explicit backends are not probed). Which readers decode a given
    compression depends on the install (cuCIM builds, ASAP versions), so the test
    checks the precondition directly instead of guessing from installed modules."""
    for backend in reader_mod.AUTO_BACKEND_ORDER:
        try:
            with reader_mod.open_slide(path, backend, require_spacing=False) as reader:
                reader.read_region((0, 0), 0, (16, 16))
        except Exception:
            continue
        return True
    return False


@pytest.mark.parametrize("compression", ["deflate", "adobe_deflate"])
def test_auto_mask_backend_decodes_a_single_channel_deflate_tiff(tmp_path, compression):
    """cuCIM opens a single-channel deflate TIFF but cannot decode it; ``auto`` must
    select a backend that reads the mask, and tiling with it must succeed."""
    tifffile = pytest.importorskip("tifffile")
    from PIL import Image

    from hs2p import FilterConfig, SlideSpec, TilingConfig, tile_slide

    slide_path = tmp_path / "slide.png"
    Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(slide_path)
    mask_path = tmp_path / "mask-deflate.tif"
    tifffile.imwrite(
        mask_path,
        np.ones((64, 64), dtype=np.uint8),
        tile=(16, 16),
        compression=compression,
        photometric="minisblack",
        resolution=(10000, 10000),
        resolutionunit="CENTIMETER",
    )
    if not _some_auto_backend_decodes(mask_path):
        pytest.skip(f"no installed auto-chain reader decodes a {compression} mask")

    selection = reader_mod.resolve_mask_backend("auto", mask_path=mask_path)
    with reader_mod.open_slide(
        mask_path, selection.backend, require_spacing=False
    ) as mask_reader:
        region = mask_reader.read_region((0, 0), 0, (16, 16))
    assert region.shape[:2] == (16, 16)

    result = tile_slide(
        SlideSpec(
            sample_id="s",
            image_path=slide_path,
            mask_path=mask_path,
            spacing_at_level_0=1.0,
        ),
        tiling=TilingConfig(
            requested_spacing_um=1.0,
            requested_tile_size_px=16,
            tolerance=0.01,
            overlap=0,
            min_coverage={"tissue": 0.5},
            backend="pil",
            mask_backend="auto",
        ),
        filtering=FilterConfig(a_t=0, a_h=0),
    )

    assert result.mask_backend == selection.backend
    assert sorted(zip(result.x.tolist(), result.y.tolist())) == sorted(
        (x, y) for x in (0, 16, 32, 48) for y in (0, 16, 32, 48)
    )


def _fake_backends(monkeypatch, *, order, decodes, undecodable_levels=None):
    """Register fake readers for ``order``; ``decodes(backend, path)`` says whether a
    reader opened on ``path`` can decode it, and ``undecodable_levels(backend, path)``
    names the levels it cannot decode even then (every fake opens every path)."""
    opened: dict[tuple[str, str], _FakeReader] = {}

    def _spec(backend):
        def _opener(path, *, spacing_override=None, require_spacing=True):
            del spacing_override, require_spacing
            reader = _FakeReader(
                decodes=decodes(backend, str(path)),
                undecodable_levels=(
                    undecodable_levels(backend, str(path)) if undecodable_levels else ()
                ),
            )
            opened[(backend, str(path))] = reader
            return reader

        return reader_mod._BackendSpec(opener=_opener, supports_path=lambda path: True)

    monkeypatch.setattr(reader_mod, "_BACKENDS", {name: _spec(name) for name in order})
    monkeypatch.setattr(reader_mod, "AUTO_BACKEND_ORDER", tuple(order))
    reader_mod._probe_backend.cache_clear()
    return opened


def test_auto_skips_a_backend_that_opens_but_cannot_decode(monkeypatch):
    opened = _fake_backends(
        monkeypatch,
        order=("cucim", "openslide"),
        decodes=lambda backend, path: backend != "cucim",
    )

    selection = reader_mod.resolve_backend("auto", wsi_path=Path("undecodable-mask.tif"))

    assert selection.backend == "openslide"
    assert selection.tried == ("cucim", "openslide")
    assert selection.reason == (
        "cuCIM could not decode the source; selected openslide for auto backend"
    )
    # one small read per level, coarsest first, stopping at the first failure; every
    # probe reader is closed
    cucim = opened[("cucim", "undecodable-mask.tif")]
    assert cucim.reads == [((0, 0), 1, (64, 64))]
    openslide = opened[("openslide", "undecodable-mask.tif")]
    assert openslide.reads == [((0, 0), 1, (64, 64)), ((0, 0), 0, (64, 64))]
    assert all(reader.closed for reader in opened.values())


def test_auto_skips_a_backend_that_decodes_the_coarsest_level_but_not_level_0(
    monkeypatch,
):
    """#268: TIFF codecs are per directory, so a reader can decode the overview but lack
    the full-resolution codec. The probe must decode every level, not the coarsest."""
    opened = _fake_backends(
        monkeypatch,
        order=("cucim", "openslide"),
        decodes=lambda backend, path: True,
        undecodable_levels=lambda backend, path: (0,) if backend == "cucim" else (),
    )

    selection = reader_mod.resolve_backend("auto", wsi_path=Path("jp2k-level0.tif"))

    assert selection.backend == "openslide"
    assert selection.tried == ("cucim", "openslide")
    assert selection.reason == (
        "cuCIM could not decode the source; selected openslide for auto backend"
    )
    assert all(reader.closed for reader in opened.values())


def test_auto_skips_a_backend_that_cannot_decode_level_0_of_the_companion(monkeypatch):
    """The companion mask is probed on every level too, and every reader of the skipped
    backend, the decodable source included, is closed."""
    opened = _fake_backends(
        monkeypatch,
        order=("cucim", "openslide"),
        decodes=lambda backend, path: True,
        undecodable_levels=lambda backend, path: (
            (0,) if backend == "cucim" and path == "mask.tif" else ()
        ),
    )

    selection = reader_mod.resolve_backend(
        "auto", wsi_path=Path("slide.svs"), mask_path=Path("mask.tif")
    )

    assert selection.backend == "openslide"
    assert selection.reason == (
        "cuCIM could not decode the source; selected openslide for auto backend"
    )
    every_level = [((0, 0), 1, (64, 64)), ((0, 0), 0, (64, 64))]
    assert opened[("cucim", "slide.svs")].reads == every_level
    assert opened[("cucim", "mask.tif")].reads == every_level
    assert opened[("openslide", "mask.tif")].reads == every_level
    assert len(opened) == 4
    assert all(reader.closed for reader in opened.values())


def test_slide_and_mask_roles_share_the_decode_probe(tmp_path, monkeypatch):
    """#178: one policy for both roles. cuCIM decodes the slide but not the mask, so the
    slide keeps cuCIM while the mask falls through to the next backend that decodes."""
    tifffile = pytest.importorskip("tifffile")
    mask_path = tmp_path / "mask.tif"
    # an 8-bit min-is-black header stays on the native chain (no tifffile routing)
    tifffile.imwrite(mask_path, np.ones((8, 8), dtype=np.uint8), photometric="minisblack")
    _fake_backends(
        monkeypatch,
        order=("cucim", "openslide"),
        decodes=lambda backend, path: backend != "cucim" or path.endswith("slide.svs"),
    )

    resolved = reader_mod.resolve_backends(
        requested_slide_backend="auto",
        requested_mask_backend="auto",
        wsi_path=Path("slide.svs"),
        mask_path=mask_path,
    )

    assert resolved.slide == reader_mod.BackendSelection(
        backend="cucim", reason="selected cuCIM for auto backend", tried=("cucim",)
    )
    assert resolved.mask == reader_mod.BackendSelection(
        backend="openslide",
        reason="cuCIM could not decode the source; selected openslide for auto backend",
        tried=("cucim", "openslide"),
    )


def test_explicit_backends_are_authoritative_and_never_probed(monkeypatch):
    monkeypatch.setattr(
        reader_mod,
        "_probe_backend",
        lambda **kwargs: pytest.fail(f"an explicit backend was probed: {kwargs}"),
    )

    resolved = reader_mod.resolve_backends(
        requested_slide_backend="cucim",
        requested_mask_backend="cucim",
        wsi_path=Path("slide.svs"),
        mask_path=Path("undecodable-mask.tif"),
    )

    assert resolved.slide == reader_mod.BackendSelection(backend="cucim", tried=("cucim",))
    assert resolved.mask == reader_mod.BackendSelection(backend="cucim", tried=("cucim",))
