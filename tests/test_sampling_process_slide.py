from pathlib import Path
from types import SimpleNamespace

import pytest


sampling_mod = pytest.importorskip("hs2p.sampling")


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
    monkeypatch.setattr(sampling_mod, "_save_sampling_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(sampling_mod, "visualize_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(sampling_mod, "overlay_mask_on_slide", lambda **kwargs: None)

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
    monkeypatch.setattr(sampling_mod, "_save_sampling_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(sampling_mod, "visualize_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(
        sampling_mod,
        "overlay_mask_on_slide",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("overlay_mask_on_slide should not run")),
    )

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
    assert captured["extract_kwargs"]["preview_pixel_mapping"] == sampling_params.pixel_mapping
    assert captured["extract_kwargs"]["preview_color_mapping"] == sampling_params.color_mapping
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
            SimpleNamespace(sample_id="slide-1", image_path=Path("slide-1.svs"), mask_path=None),
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
    monkeypatch.setattr(sampling_mod, "process_slide_wrapper", lambda kwargs: (kwargs["sample_id"], {"status": "success"}))

    sampling_mod.main(SimpleNamespace())

    assert seen["pool_processes"] == 4
    assert [args["num_workers"] for args in seen["args_list"]] == [1]


def test_sampling_main_allows_explicit_inner_slide_workers(monkeypatch, tmp_path):
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
            SimpleNamespace(sample_id="slide-1", image_path=Path("slide-1.svs"), mask_path=None),
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
    monkeypatch.setattr(sampling_mod, "process_slide_wrapper", lambda kwargs: (kwargs["sample_id"], {"status": "success"}))

    sampling_mod.main(SimpleNamespace())

    assert seen["pool_processes"] == 4
    assert [args["num_workers"] for args in seen["args_list"]] == [2]
