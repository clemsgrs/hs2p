import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _import_sampling_module():
    try:
        return importlib.import_module("hs2p.sampling")
    except ModuleNotFoundError as exc:
        if exc.name != "seaborn":
            raise
        sys.modules["seaborn"] = types.SimpleNamespace(
            color_palette=lambda name: [(1.0, 0.0, 0.0)] * 20
        )
        return importlib.import_module("hs2p.sampling")


sampling_mod = _import_sampling_module()


def test_independent_sampling_no_visualization_does_not_crash(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        visualize=False,
        output_dir=str(tmp_path),
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params=SimpleNamespace(),
            filter_params=SimpleNamespace(),
            visu_params=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(independant_sampling=True),
        ),
    )
    sampling_params = sampling_mod.SamplingParameters(
        pixel_mapping={"tumor": 1},
        color_mapping=None,
        tissue_percentage={"tumor": 0.1},
    )

    def _fake_sample_coordinates(**kwargs):
        return SimpleNamespace(
            coordinates=[(0, 0)],
            contour_indices=[0],
            read_level=0,
            read_spacing_um=0.5,
            read_tile_size_px=256,
            tile_size_lv0=256,
        )

    monkeypatch.setattr(sampling_mod, "sample_coordinates", _fake_sample_coordinates)
    monkeypatch.setattr(
        sampling_mod, "_save_sampling_coordinates", lambda **kwargs: None
    )
    monkeypatch.setattr(sampling_mod, "visualize_coordinates", lambda **kwargs: None)

    _, status_info = sampling_mod.process_slide(
        sample_id="sample-1",
        wsi_path=Path("fake-wsi.tif"),
        mask_path=Path("fake-mask.tif"),
        cfg=cfg,
        tiling_config=sampling_mod.TilingConfig(
            target_spacing_um=0.5,
            target_tile_size_px=256,
            tolerance=0.05,
            overlap=0.0,
            tissue_threshold=0.1,
            drop_holes=False,
            use_padding=True,
            backend="asap",
        ),
        segmentation_config=sampling_mod.SegmentationConfig(
            downsample=64,
            sthresh=8,
            sthresh_up=255,
            mthresh=7,
            close=4,
            use_otsu=False,
            use_hsv=True,
        ),
        filter_config=sampling_mod.FilterConfig(
            ref_tile_size=16,
            a_t=4,
            a_h=2,
            max_n_holes=8,
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
        mask_visualize_dir=None,
        sampling_visualize_dir=None,
        sampling_params=sampling_params,
    )

    assert status_info["status"] == "success"


def test_sampling_main_uses_shared_cli_config_builder(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        seed=0,
        output_dir=str(tmp_path),
        speed=SimpleNamespace(num_workers=1),
        resume=False,
        visualize=False,
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
                drop_holes=False,
                use_padding=True,
            ),
            seg_params={},
            filter_params={},
            visu_params=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(
                pixel_mapping=[{"tumor": 1}],
                tissue_percentage=[{"tumor": 0.1}],
                color_mapping=None,
                independant_sampling=True,
            ),
        ),
    )
    called = {}

    monkeypatch.setattr(sampling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(sampling_mod, "fix_random_seeds", lambda seed: None)
    monkeypatch.setattr(sampling_mod, "load_csv", lambda cfg: [])
    monkeypatch.setattr(sampling_mod.mp, "cpu_count", lambda: 1)

    def _fake_build_cli_configs(seen_cfg):
        called["cfg"] = seen_cfg
        return (
            sampling_mod.TilingConfig(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
                drop_holes=False,
                use_padding=True,
                backend="asap",
            ),
            sampling_mod.SegmentationConfig(
                downsample=64,
                sthresh=8,
                sthresh_up=255,
                mthresh=7,
                close=4,
                use_otsu=False,
                use_hsv=True,
            ),
            sampling_mod.FilterConfig(
                ref_tile_size=16,
                a_t=4,
                a_h=2,
                max_n_holes=8,
                filter_white=False,
                filter_black=False,
                white_threshold=220,
                black_threshold=25,
                fraction_threshold=0.9,
            ),
        )

    monkeypatch.setattr(sampling_mod, "_build_cli_configs", _fake_build_cli_configs)

    sampling_mod.main(SimpleNamespace())

    assert called["cfg"] is cfg


def test_process_slide_uses_extraction_preview_instead_of_reopening_overlay(
    monkeypatch,
    tmp_path,
):
    captured = {}
    cfg = SimpleNamespace(
        visualize=True,
        output_dir=str(tmp_path),
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params=SimpleNamespace(),
            filter_params=SimpleNamespace(),
            visu_params=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(independant_sampling=False),
        ),
    )
    sampling_params = sampling_mod.SamplingParameters(
        pixel_mapping={"background": 0, "tissue": 1},
        color_mapping={"background": None, "tissue": [157, 219, 129]},
        tissue_percentage={"background": None, "tissue": 0.1},
    )
    mask_visualize_dir = tmp_path / "visualization" / "mask"
    mask_visualize_dir.mkdir(parents=True, exist_ok=True)

    def _fake_extract_coordinates(**kwargs):
        captured["extract_kwargs"] = kwargs
        preview_path = kwargs["mask_visu_path"]
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_bytes(b"preview")
        return SimpleNamespace(
            coordinates=[(0, 0)],
            contour_indices=[0],
            tissue_percentages=[1.0],
            read_level=0,
            read_spacing_um=0.5,
            read_tile_size_px=256,
            tile_size_lv0=256,
        )

    monkeypatch.setattr(sampling_mod, "extract_coordinates", _fake_extract_coordinates)
    monkeypatch.setattr(
        sampling_mod,
        "filter_coordinates",
        lambda **kwargs: ({"tissue": [(0, 0)]}, {"tissue": [0]}),
    )
    monkeypatch.setattr(
        sampling_mod, "_save_sampling_coordinates", lambda **kwargs: None
    )
    monkeypatch.setattr(sampling_mod, "visualize_coordinates", lambda **kwargs: None)

    _, status_info = sampling_mod.process_slide(
        sample_id="sample-2",
        wsi_path=Path("fake-wsi.tif"),
        mask_path=Path("fake-mask.tif"),
        cfg=cfg,
        tiling_config=sampling_mod.TilingConfig(
            target_spacing_um=0.5,
            target_tile_size_px=256,
            tolerance=0.05,
            overlap=0.0,
            tissue_threshold=0.1,
            drop_holes=False,
            use_padding=True,
            backend="asap",
        ),
        segmentation_config=sampling_mod.SegmentationConfig(
            downsample=64,
            sthresh=8,
            sthresh_up=255,
            mthresh=7,
            close=4,
            use_otsu=False,
            use_hsv=True,
        ),
        filter_config=sampling_mod.FilterConfig(
            ref_tile_size=16,
            a_t=4,
            a_h=2,
            max_n_holes=8,
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
        mask_visualize_dir=mask_visualize_dir,
        sampling_visualize_dir=None,
        sampling_params=sampling_params,
    )

    assert status_info["status"] == "success"
    assert (mask_visualize_dir / "sample-2.png").is_file()
    assert captured["extract_kwargs"]["preview_downsample"] == 32
    assert (
        captured["extract_kwargs"]["preview_pixel_mapping"]
        == sampling_params.pixel_mapping
    )
    assert (
        captured["extract_kwargs"]["preview_color_mapping"]
        == sampling_params.color_mapping
    )
    assert captured["extract_kwargs"]["preview_palette"] is not None


def test_sampling_main_defaults_inner_slide_workers_to_one(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        seed=0,
        output_dir=str(tmp_path),
        speed=SimpleNamespace(num_workers=4),
        resume=False,
        visualize=False,
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
                drop_holes=False,
                use_padding=True,
            ),
            seg_params={},
            filter_params={},
            visu_params=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(
                pixel_mapping=[{"tumor": 1}],
                tissue_percentage=[{"tumor": 0.1}],
                color_mapping=None,
                independant_sampling=True,
            ),
        ),
    )
    seen = {}

    monkeypatch.setattr(sampling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(sampling_mod, "fix_random_seeds", lambda seed: None)
    monkeypatch.setattr(
        sampling_mod,
        "load_csv",
        lambda cfg: [
            SimpleNamespace(
                sample_id="slide-1", image_path=Path("slide-1.svs"), mask_path=None
            ),
        ],
    )
    monkeypatch.setattr(
        sampling_mod,
        "_build_cli_configs",
        lambda cfg: (
            sampling_mod.TilingConfig(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
                drop_holes=False,
                use_padding=True,
                backend="asap",
            ),
            sampling_mod.SegmentationConfig(64, 8, 255, 7, 4, False, True),
            sampling_mod.FilterConfig(16, 4, 2, 8, False, False, 220, 25, 0.9),
        ),
    )
    monkeypatch.setattr(sampling_mod.mp, "cpu_count", lambda: 8)
    monkeypatch.setattr(sampling_mod.tqdm, "tqdm", lambda iterable, **kwargs: iterable)

    class _FakePool:
        def __init__(self, processes):
            seen["pool_processes"] = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap(self, fn, args_list):
            seen["args_list"] = list(args_list)
            for args in seen["args_list"]:
                yield fn(args)

    monkeypatch.setattr(sampling_mod.mp, "Pool", _FakePool)
    monkeypatch.setattr(
        sampling_mod,
        "process_slide_wrapper",
        lambda kwargs: (kwargs["sample_id"], {"status": "success"}),
    )

    sampling_mod.main(SimpleNamespace())

    assert seen["pool_processes"] == 1
    assert [args["num_workers"] for args in seen["args_list"]] == [1]


def test_sampling_main_rejects_explicit_inner_slide_workers_override(
    monkeypatch, tmp_path
):
    cfg = SimpleNamespace(
        seed=0,
        output_dir=str(tmp_path),
        speed=SimpleNamespace(num_workers=4, inner_workers=2),
        resume=False,
        visualize=False,
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
                drop_holes=False,
                use_padding=True,
            ),
            seg_params={},
            filter_params={},
            visu_params=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(
                pixel_mapping=[{"tumor": 1}],
                tissue_percentage=[{"tumor": 0.1}],
                color_mapping=None,
                independant_sampling=True,
            ),
        ),
    )
    seen = {}

    monkeypatch.setattr(sampling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(sampling_mod, "fix_random_seeds", lambda seed: None)
    monkeypatch.setattr(
        sampling_mod,
        "load_csv",
        lambda cfg: [
            SimpleNamespace(
                sample_id="slide-1", image_path=Path("slide-1.svs"), mask_path=None
            ),
        ],
    )
    monkeypatch.setattr(
        sampling_mod,
        "_build_cli_configs",
        lambda cfg: (
            sampling_mod.TilingConfig(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
                drop_holes=False,
                use_padding=True,
                backend="asap",
            ),
            sampling_mod.SegmentationConfig(64, 8, 255, 7, 4, False, True),
            sampling_mod.FilterConfig(16, 4, 2, 8, False, False, 220, 25, 0.9),
        ),
    )
    monkeypatch.setattr(sampling_mod.mp, "cpu_count", lambda: 8)
    monkeypatch.setattr(sampling_mod.tqdm, "tqdm", lambda iterable, **kwargs: iterable)

    class _FakePool:
        def __init__(self, processes):
            seen["pool_processes"] = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap(self, fn, args_list):
            seen["args_list"] = list(args_list)
            for args in seen["args_list"]:
                yield fn(args)

    monkeypatch.setattr(sampling_mod.mp, "Pool", _FakePool)
    monkeypatch.setattr(
        sampling_mod,
        "process_slide_wrapper",
        lambda kwargs: (kwargs["sample_id"], {"status": "success"}),
    )

    with pytest.raises(ValueError, match="cfg.speed.inner_workers is no longer supported"):
        sampling_mod.main(SimpleNamespace())


def test_save_sampling_coordinates_uses_annotation_threshold_and_sampling_mode_in_hash(
    monkeypatch,
    tmp_path,
):
    captured = {}
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        tiling=SimpleNamespace(
            sampling_params=SimpleNamespace(independant_sampling=False),
        ),
    )

    def _fake_save_tiling_result(result, output_dir, coordinates_dir=None):
        captured["result"] = result
        captured["output_dir"] = output_dir
        captured["coordinates_dir"] = coordinates_dir
        return SimpleNamespace(
            sample_id=result.sample_id,
            tiles_npz_path=Path(coordinates_dir) / f"{result.sample_id}.tiles.npz",
            tiles_meta_path=Path(coordinates_dir)
            / f"{result.sample_id}.tiles.meta.json",
            num_tiles=result.num_tiles,
        )

    monkeypatch.setattr(sampling_mod, "save_tiling_result", _fake_save_tiling_result)

    tiling_config = sampling_mod.TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.1,
        drop_holes=False,
        use_padding=True,
        backend="asap",
    )
    segmentation_config = sampling_mod.SegmentationConfig(
        downsample=64,
        sthresh=8,
        sthresh_up=255,
        mthresh=7,
        close=4,
        use_otsu=False,
        use_hsv=True,
    )
    filter_config = sampling_mod.FilterConfig(
        ref_tile_size=16,
        a_t=4,
        a_h=2,
        max_n_holes=8,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )
    sampling_params = sampling_mod.SamplingParameters(
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2},
        color_mapping={"background": None, "tumor": None, "stroma": None},
        tissue_percentage={"background": None, "tumor": 0.7, "stroma": 0.2},
    )
    extraction = SimpleNamespace(
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=256,
        tile_size_lv0=256,
    )

    sampling_mod._save_sampling_coordinates(
        sample_id="sample-1",
        image_path=Path("slide.svs"),
        mask_path=Path("mask.tif"),
        backend="asap",
        cfg=cfg,
        tiling_config=tiling_config,
        segmentation_config=segmentation_config,
        filter_config=filter_config,
        annotation="tumor",
        coordinates=[(0, 0), (256, 0)],
        extraction=extraction,
        sampling_params=sampling_params,
    )

    result = captured["result"]
    assert result.tissue_threshold == 0.7
    assert result.config_hash == sampling_mod.compute_config_hash(
        tiling=tiling_config,
        segmentation=segmentation_config,
        filtering=filter_config,
        extra={
            "annotation": "tumor",
            "sampling": {
                "pixel_mapping": sampling_params.pixel_mapping,
                "tissue_percentage": sampling_params.tissue_percentage,
                "independant_sampling": False,
            },
        },
    )
    assert captured["coordinates_dir"] == tmp_path / "coordinates" / "tumor"
