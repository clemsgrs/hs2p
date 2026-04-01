import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hs2p.wsi.streaming.plans import resolve_read_step_px


def _import_sampling_module():
    try:
        return importlib.import_module("hs2p.cli.sampling")
    except ModuleNotFoundError as exc:
        if exc.name != "seaborn":
            raise
        sys.modules["seaborn"] = types.SimpleNamespace(
            color_palette=lambda name: [(1.0, 0.0, 0.0)] * 20
        )
        return importlib.import_module("hs2p.cli.sampling")


sampling_mod = _import_sampling_module()
sampling_support_mod = importlib.import_module("hs2p.cli.sampling")


def _resolved_sampling_spec(
    *,
    pixel_mapping: dict[str, int],
    tissue_percentage: dict[str, float | None],
    color_mapping: dict[str, list[int] | None] | None = None,
):
    return sampling_mod.SamplingSpec(
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        tissue_percentage=tissue_percentage,
        active_annotations=tuple(
            annotation
            for annotation, threshold in tissue_percentage.items()
            if annotation != "background" and threshold is not None
        ),
    )


def test_independent_sampling_without_previews_does_not_crash(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        save_previews=False,
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
            preview=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(independent_sampling=True),
        ),
    )
    resolved_sampling_spec = _resolved_sampling_spec(
        pixel_mapping={"background": 0, "tumor": 1},
        color_mapping=None,
        tissue_percentage={"background": None, "tumor": 0.1},
    )

    def _fake_execute_coordinate_request(request):
        return SimpleNamespace(
            per_annotation_results={
                "tumor": SimpleNamespace(
                    coordinates=[(0, 0)],
                    contour_indices=[0],
                    read_level=0,
                    read_spacing_um=0.5,
                    read_tile_size_px=256,
                    tile_size_lv0=256,
                )
            }
        )

    monkeypatch.setattr(
        sampling_mod, "execute_coordinate_request", _fake_execute_coordinate_request
    )
    monkeypatch.setattr(sampling_mod, "save_sampling_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(sampling_mod, "write_coordinate_preview", lambda **kwargs: None)

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
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
        mask_preview_dir=None,
        sampling_preview_dir=None,
        resolved_sampling_spec=resolved_sampling_spec,
    )

    assert status_info["status"] == "success"


def test_process_slide_accepts_resolved_sampling_spec(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        save_previews=False,
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
            preview=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(independent_sampling=True),
        ),
    )
    resolved_sampling_spec = sampling_mod.SamplingSpec(
        pixel_mapping={"background": 0, "tumor": 1},
        color_mapping={"background": None, "tumor": None},
        tissue_percentage={"background": None, "tumor": 0.1},
        active_annotations=("tumor",),
    )

    def _fake_execute_coordinate_request(request):
        return SimpleNamespace(
            per_annotation_results={
                "tumor": SimpleNamespace(
                    coordinates=[(0, 0)],
                    contour_indices=[0],
                    read_level=0,
                    read_spacing_um=0.5,
                    read_tile_size_px=256,
                    tile_size_lv0=256,
                )
            }
        )

    monkeypatch.setattr(
        sampling_mod, "execute_coordinate_request", _fake_execute_coordinate_request
    )
    monkeypatch.setattr(sampling_mod, "save_sampling_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(sampling_mod, "write_coordinate_preview", lambda **kwargs: None)

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
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
        mask_preview_dir=None,
        sampling_preview_dir=None,
        resolved_sampling_spec=resolved_sampling_spec,
    )

    assert status_info["status"] == "success"


def test_sampling_main_uses_shared_config_resolvers(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        seed=0,
        output_dir=str(tmp_path),
        speed=SimpleNamespace(num_workers=1),
        resume=False,
        save_previews=False,
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params={},
            filter_params={},
            preview=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(
                pixel_mapping=[{"background": 0}, {"tumor": 1}],
                tissue_percentage=[{"background": None}, {"tumor": 0.1}],
                color_mapping=None,
                independent_sampling=True,
            ),
        ),
    )
    called = {}

    monkeypatch.setattr(sampling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(sampling_mod, "load_csv", lambda cfg, **kwargs: [])
    monkeypatch.setattr(sampling_mod.mp, "cpu_count", lambda: 1)

    def _fake_resolve_tiling_config(seen_cfg):
        called["cfg"] = seen_cfg
        return sampling_mod.TilingConfig(
            target_spacing_um=0.5,
            target_tile_size_px=256,
            tolerance=0.05,
            overlap=0.0,
            tissue_threshold=0.1,
            backend="asap",
        )

    monkeypatch.setattr(
        sampling_mod, "resolve_tiling_config", _fake_resolve_tiling_config
    )
    monkeypatch.setattr(
        sampling_mod,
        "resolve_segmentation_config",
        lambda cfg: sampling_mod.SegmentationConfig(
            downsample=64,
            sthresh=8,
            sthresh_up=255,
            mthresh=7,
            close=4,
            use_otsu=False,
            use_hsv=True,
        ),
    )
    monkeypatch.setattr(
        sampling_mod,
        "resolve_filter_config",
        lambda cfg: sampling_mod.FilterConfig(
            ref_tile_size=16,
            a_t=4,
            a_h=2,
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
    )

    sampling_mod.main(SimpleNamespace())

    assert called["cfg"] is cfg


def test_sampling_main_rejects_partial_sampling_config(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        seed=0,
        output_dir=str(tmp_path),
        speed=SimpleNamespace(num_workers=1),
        resume=False,
        save_previews=False,
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params={},
            filter_params={},
            preview=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(
                pixel_mapping=[{"background": 0}, {"tumor": 1}],
                tissue_percentage=None,
                color_mapping=None,
                independent_sampling=False,
            ),
        ),
    )

    monkeypatch.setattr(sampling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(
        sampling_mod,
        "load_csv",
        lambda cfg, **kwargs: [
            SimpleNamespace(
                sample_id="slide-1",
                image_path=Path("slide.svs"),
                mask_path=Path("mask.tif"),
            )
        ],
    )

    with pytest.raises(ValueError, match="tissue_percentage"):
        sampling_mod.main(SimpleNamespace())


def test_sampling_main_rejects_missing_background_label(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        seed=0,
        output_dir=str(tmp_path),
        speed=SimpleNamespace(num_workers=1),
        resume=False,
        save_previews=False,
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params={},
            filter_params={},
            preview=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(
                pixel_mapping=[{"tumor": 1}],
                tissue_percentage=[{"tumor": 0.1}],
                color_mapping=None,
                independent_sampling=False,
            ),
        ),
    )

    monkeypatch.setattr(sampling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(
        sampling_mod,
        "load_csv",
        lambda cfg, **kwargs: [
            SimpleNamespace(
                sample_id="slide-1",
                image_path=Path("slide.svs"),
                mask_path=Path("mask.tif"),
            )
        ],
    )

    with pytest.raises(ValueError, match="background"):
        sampling_mod.main(SimpleNamespace())


def test_process_slide_uses_extraction_preview_instead_of_reopening_overlay(
    monkeypatch,
    tmp_path,
):
    captured = {}
    cfg = SimpleNamespace(
        save_previews=True,
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
            preview=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(independent_sampling=False),
        ),
    )
    resolved_sampling_spec = _resolved_sampling_spec(
        pixel_mapping={"background": 0, "tissue": 1},
        color_mapping={"background": None, "tissue": [157, 219, 129]},
        tissue_percentage={"background": None, "tissue": 0.1},
    )
    mask_preview_dir = tmp_path / "preview" / "mask"
    mask_preview_dir.mkdir(parents=True, exist_ok=True)

    def _fake_execute_coordinate_request(request):
        captured["request"] = request
        preview_path = request.mask_preview_path
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_bytes(b"preview")
        return SimpleNamespace(
            per_annotation_results={
                "tissue": SimpleNamespace(
                    coordinates=[(0, 0)],
                    contour_indices=[0],
                    tissue_percentages=[1.0],
                    read_level=0,
                    read_spacing_um=0.5,
                    read_tile_size_px=256,
                    tile_size_lv0=256,
                )
            },
        )

    monkeypatch.setattr(
        sampling_mod, "execute_coordinate_request", _fake_execute_coordinate_request
    )
    monkeypatch.setattr(sampling_mod, "save_sampling_coordinates", lambda **kwargs: None)
    monkeypatch.setattr(sampling_mod, "write_coordinate_preview", lambda **kwargs: None)

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
            filter_white=False,
            filter_black=False,
            white_threshold=220,
            black_threshold=25,
            fraction_threshold=0.9,
        ),
        mask_preview_dir=mask_preview_dir,
        sampling_preview_dir=None,
        resolved_sampling_spec=resolved_sampling_spec,
    )

    assert status_info["status"] == "success"
    assert (mask_preview_dir / "sample-2.png").is_file()
    assert captured["request"].preview_downsample == 32
    assert captured["request"].preview_pixel_mapping == resolved_sampling_spec.pixel_mapping
    assert captured["request"].preview_color_mapping == resolved_sampling_spec.color_mapping
    assert captured["request"].preview_palette is not None


def test_sampling_main_defaults_inner_slide_workers_to_one(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        seed=0,
        output_dir=str(tmp_path),
        speed=SimpleNamespace(num_workers=4),
        resume=False,
        save_previews=False,
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params={},
            filter_params={},
            preview=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(
                pixel_mapping=[{"background": 0}, {"tumor": 1}],
                tissue_percentage=[{"background": None}, {"tumor": 0.1}],
                color_mapping=None,
                independent_sampling=True,
            ),
        ),
    )
    seen = {}

    monkeypatch.setattr(sampling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(
        sampling_mod,
        "load_csv",
        lambda cfg, **kwargs: [
            SimpleNamespace(
                sample_id="slide-1", image_path=Path("slide-1.svs"), mask_path=None
            ),
        ],
    )
    monkeypatch.setattr(
        sampling_mod,
        "resolve_tiling_config",
        lambda cfg: sampling_mod.TilingConfig(
            target_spacing_um=0.5,
            target_tile_size_px=256,
            tolerance=0.05,
            overlap=0.0,
            tissue_threshold=0.1,
            backend="asap",
        ),
    )
    monkeypatch.setattr(
        sampling_mod,
        "resolve_segmentation_config",
        lambda cfg: sampling_mod.SegmentationConfig(64, 8, 255, 7, 4, False, True),
    )
    monkeypatch.setattr(
        sampling_mod,
        "resolve_filter_config",
        lambda cfg: sampling_mod.FilterConfig(16, 4, 2, False, False, 220, 25, 0.9),
    )
    monkeypatch.setattr(sampling_mod.mp, "cpu_count", lambda: 8)

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
        lambda kwargs: (
            kwargs["sample_id"],
            {
                "status": "success",
                "rows": [
                    {
                        "sample_id": kwargs["sample_id"],
                        "annotation": "tumor",
                        "image_path": str(kwargs["wsi_path"]),
                        "annotation_mask_path": (
                            str(kwargs["annotation_mask_path"])
                            if kwargs["annotation_mask_path"] is not None
                            else None
                        ),
                    "sampling_status": "success",
                    "num_tiles": 0,
                    "coordinates_npz_path": np.nan,
                    "coordinates_meta_path": np.nan,
                    "error": np.nan,
                    "traceback": np.nan,
                }
                ],
            },
        ),
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
        save_previews=False,
        tiling=SimpleNamespace(
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params={},
            filter_params={},
            preview=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(
                pixel_mapping=[{"background": 0}, {"tumor": 1}],
                tissue_percentage=[{"background": None}, {"tumor": 0.1}],
                color_mapping=None,
                independent_sampling=True,
            ),
        ),
    )
    seen = {}

    monkeypatch.setattr(sampling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(
        sampling_mod,
        "load_csv",
        lambda cfg, **kwargs: [
            SimpleNamespace(
                sample_id="slide-1", image_path=Path("slide-1.svs"), mask_path=None
            ),
        ],
    )
    monkeypatch.setattr(
        sampling_mod,
        "resolve_tiling_config",
        lambda cfg: sampling_mod.TilingConfig(
            target_spacing_um=0.5,
            target_tile_size_px=256,
            tolerance=0.05,
            overlap=0.0,
            tissue_threshold=0.1,
            backend="asap",
        ),
    )
    monkeypatch.setattr(
        sampling_mod,
        "resolve_segmentation_config",
        lambda cfg: sampling_mod.SegmentationConfig(64, 8, 255, 7, 4, False, True),
    )
    monkeypatch.setattr(
        sampling_mod,
        "resolve_filter_config",
        lambda cfg: sampling_mod.FilterConfig(16, 4, 2, False, False, 220, 25, 0.9),
    )
    monkeypatch.setattr(sampling_mod.mp, "cpu_count", lambda: 8)

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
        lambda kwargs: (
            kwargs["sample_id"],
            {
                "status": "success",
                "rows": [
                    {
                        "sample_id": kwargs["sample_id"],
                        "annotation": "tumor",
                        "image_path": str(kwargs["wsi_path"]),
                        "annotation_mask_path": (
                            str(kwargs["annotation_mask_path"])
                            if kwargs["annotation_mask_path"] is not None
                            else None
                        ),
                    "sampling_status": "success",
                    "num_tiles": 0,
                    "coordinates_npz_path": np.nan,
                    "coordinates_meta_path": np.nan,
                    "error": np.nan,
                    "traceback": np.nan,
                }
                ],
            },
        ),
    )

    with pytest.raises(ValueError, match="cfg.speed.inner_workers is no longer supported"):
        sampling_mod.main(SimpleNamespace())


def test_save_sampling_coordinates_uses_annotation_threshold_and_sampling_mode(
    monkeypatch,
    tmp_path,
):
    captured = {}
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        tiling=SimpleNamespace(
            sampling_params=SimpleNamespace(independent_sampling=False),
        ),
    )

    def _fake_save_tiling_result(result, output_dir, tiles_dir=None):
        captured["result"] = result
        captured["output_dir"] = output_dir
        captured["tiles_dir"] = tiles_dir
        return SimpleNamespace(
            sample_id=result.sample_id,
            coordinates_npz_path=Path(tiles_dir) / f"{result.sample_id}.coordinates.npz",
            coordinates_meta_path=Path(tiles_dir)
            / f"{result.sample_id}.coordinates.meta.json",
            num_tiles=len(result.coordinates),
        )

    monkeypatch.setattr(sampling_mod, "save_tiling_result", _fake_save_tiling_result)

    tiling_config = sampling_mod.TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.1,
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
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )
    resolved_sampling_spec = _resolved_sampling_spec(
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2},
        color_mapping={"background": None, "tumor": None, "stroma": None},
        tissue_percentage={"background": None, "tumor": 0.7, "stroma": 0.2},
    )
    extraction = sampling_mod.CoordinateExtractionResult(
        contour_indices=[],
        tissue_percentages=[],
        x=np.array([], dtype=np.int64),
        y=np.array([], dtype=np.int64),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=256,
        read_step_px=256,
        resize_factor=1.0,
        tile_size_lv0=256,
        step_px_lv0=256,
    )
    sampling_support_mod.save_sampling_coordinates(
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
        resolved_sampling_spec=resolved_sampling_spec,
        save_tiling_result=sampling_mod.save_tiling_result,
    )

    result = captured["result"]
    assert result.min_tissue_fraction == 0.7
    assert resolve_read_step_px(result) == 256
    assert result.step_px_lv0 == 256
    assert result.annotation == "tumor"
    assert result.selection_strategy == sampling_mod.CoordinateSelectionStrategy.JOINT_SAMPLING
    assert result.output_mode == sampling_mod.CoordinateOutputMode.PER_ANNOTATION
    assert captured["tiles_dir"] == tmp_path / "tiles" / "tumor"


def test_save_sampling_coordinates_writes_sampling_metadata_fields(tmp_path):
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        tiling=SimpleNamespace(
            sampling_params=SimpleNamespace(independent_sampling=False),
        ),
    )
    tiling_config = sampling_mod.TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.1,
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
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )
    resolved_sampling_spec = _resolved_sampling_spec(
        pixel_mapping={"background": 0, "tumor": 1},
        color_mapping={"background": None, "tumor": None},
        tissue_percentage={"background": None, "tumor": 0.7},
    )
    extraction = sampling_mod.CoordinateExtractionResult(
        contour_indices=[],
        tissue_percentages=[],
        x=np.array([], dtype=np.int64),
        y=np.array([], dtype=np.int64),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=256,
        read_step_px=256,
        resize_factor=1.0,
        tile_size_lv0=256,
        step_px_lv0=256,
    )
    artifacts = sampling_support_mod.save_sampling_coordinates(
        sample_id="sample-1",
        image_path=Path("slide.svs"),
        mask_path=Path("mask.tif"),
        backend="asap",
        cfg=cfg,
        tiling_config=tiling_config,
        segmentation_config=segmentation_config,
        filter_config=filter_config,
        annotation="tumor",
        coordinates=[(0, 0)],
        extraction=extraction,
        resolved_sampling_spec=resolved_sampling_spec,
        save_tiling_result=sampling_mod.save_tiling_result,
    )

    meta = json.loads(artifacts.coordinates_meta_path.read_text())
    assert meta["artifact"]["annotation"] == "tumor"
    assert meta["artifact"]["selection_strategy"] == "joint_sampling"
    assert meta["artifact"]["output_mode"] == "per_annotation"
    assert meta["tiling"]["step_px_lv0"] == 256


def test_validate_sampling_artifact_row_accepts_matching_metadata(tmp_path):
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        tiling=SimpleNamespace(
            sampling_params=SimpleNamespace(independent_sampling=False),
        ),
    )
    tiling_config = sampling_mod.TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.1,
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
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )
    resolved_sampling_spec = _resolved_sampling_spec(
        pixel_mapping={"background": 0, "tumor": 1},
        color_mapping={"background": None, "tumor": None},
        tissue_percentage={"background": None, "tumor": 0.7},
    )
    extraction = sampling_mod.CoordinateExtractionResult(
        contour_indices=[],
        tissue_percentages=[],
        x=np.array([0], dtype=np.int64),
        y=np.array([0], dtype=np.int64),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=256,
        read_step_px=256,
        resize_factor=1.0,
        tile_size_lv0=256,
        step_px_lv0=256,
    )
    artifacts = sampling_support_mod.save_sampling_coordinates(
        sample_id="sample-1",
        image_path=Path("slide.svs"),
        mask_path=Path("mask.tif"),
        backend="asap",
        cfg=cfg,
        tiling_config=tiling_config,
        segmentation_config=segmentation_config,
        filter_config=filter_config,
        annotation="tumor",
        coordinates=[(0, 0)],
        extraction=extraction,
        resolved_sampling_spec=resolved_sampling_spec,
        save_tiling_result=sampling_mod.save_tiling_result,
    )

    sampling_support_mod.validate_sampling_artifact_row(
        row={
            "sample_id": "sample-1",
            "annotation": "tumor",
            "image_path": "slide.svs",
            "annotation_mask_path": "mask.tif",
            "sampling_status": "success",
            "num_tiles": 1,
            "coordinates_npz_path": str(artifacts.coordinates_npz_path),
            "coordinates_meta_path": str(artifacts.coordinates_meta_path),
        },
        whole_slide=SimpleNamespace(
            sample_id="sample-1",
            image_path=Path("slide.svs"),
            mask_path=Path("mask.tif"),
        ),
        tiling_config=tiling_config,
        segmentation_config=segmentation_config,
        filter_config=filter_config,
        expected_tissue_threshold=0.7,
        selection_strategy=sampling_mod.CoordinateSelectionStrategy.JOINT_SAMPLING,
    )


def test_validate_sampling_artifact_row_ignores_disabled_filter_threshold_mismatches(
    tmp_path,
):
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        tiling=SimpleNamespace(
            sampling_params=SimpleNamespace(independent_sampling=False),
        ),
    )
    tiling_config = sampling_mod.TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.1,
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
    stored_filter_config = sampling_mod.FilterConfig(
        ref_tile_size=16,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )
    resolved_sampling_spec = _resolved_sampling_spec(
        pixel_mapping={"background": 0, "tumor": 1},
        color_mapping={"background": None, "tumor": None},
        tissue_percentage={"background": None, "tumor": 0.7},
    )
    extraction = sampling_mod.CoordinateExtractionResult(
        contour_indices=[],
        tissue_percentages=[],
        x=np.array([0], dtype=np.int64),
        y=np.array([0], dtype=np.int64),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=256,
        read_step_px=256,
        resize_factor=1.0,
        tile_size_lv0=256,
        step_px_lv0=256,
    )
    artifacts = sampling_support_mod.save_sampling_coordinates(
        sample_id="sample-1",
        image_path=Path("slide.svs"),
        mask_path=Path("mask.tif"),
        backend="asap",
        cfg=cfg,
        tiling_config=tiling_config,
        segmentation_config=segmentation_config,
        filter_config=stored_filter_config,
        annotation="tumor",
        coordinates=[(0, 0)],
        extraction=extraction,
        resolved_sampling_spec=resolved_sampling_spec,
        save_tiling_result=sampling_mod.save_tiling_result,
    )
    compatibility_filter_config = sampling_mod.FilterConfig(
        ref_tile_size=16,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=111,
        black_threshold=9,
        fraction_threshold=0.25,
        filter_grayspace=False,
        grayspace_saturation_threshold=0.99,
        grayspace_fraction_threshold=0.12,
        filter_blur=False,
        blur_threshold=5.0,
        qc_spacing_um=9.0,
    )

    sampling_support_mod.validate_sampling_artifact_row(
        row={
            "sample_id": "sample-1",
            "annotation": "tumor",
            "image_path": "slide.svs",
            "annotation_mask_path": "mask.tif",
            "sampling_status": "success",
            "num_tiles": 1,
            "coordinates_npz_path": str(artifacts.coordinates_npz_path),
            "coordinates_meta_path": str(artifacts.coordinates_meta_path),
        },
        whole_slide=SimpleNamespace(
            sample_id="sample-1",
            image_path=Path("slide.svs"),
            mask_path=Path("mask.tif"),
        ),
        tiling_config=tiling_config,
        segmentation_config=segmentation_config,
        filter_config=compatibility_filter_config,
        expected_tissue_threshold=0.7,
        selection_strategy=sampling_mod.CoordinateSelectionStrategy.JOINT_SAMPLING,
    )


def test_validate_sampling_artifact_row_rejects_mismatched_tiling_config(tmp_path):
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        tiling=SimpleNamespace(
            sampling_params=SimpleNamespace(independent_sampling=False),
        ),
    )
    tiling_config = sampling_mod.TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.1,
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
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )
    resolved_sampling_spec = _resolved_sampling_spec(
        pixel_mapping={"background": 0, "tumor": 1},
        color_mapping={"background": None, "tumor": None},
        tissue_percentage={"background": None, "tumor": 0.7},
    )
    extraction = sampling_mod.CoordinateExtractionResult(
        contour_indices=[],
        tissue_percentages=[],
        x=np.array([0], dtype=np.int64),
        y=np.array([0], dtype=np.int64),
        read_level=0,
        read_spacing_um=0.5,
        read_tile_size_px=256,
        read_step_px=256,
        resize_factor=1.0,
        tile_size_lv0=256,
        step_px_lv0=256,
    )
    artifacts = sampling_support_mod.save_sampling_coordinates(
        sample_id="sample-1",
        image_path=Path("slide.svs"),
        mask_path=Path("mask.tif"),
        backend="asap",
        cfg=cfg,
        tiling_config=tiling_config,
        segmentation_config=segmentation_config,
        filter_config=filter_config,
        annotation="tumor",
        coordinates=[(0, 0)],
        extraction=extraction,
        resolved_sampling_spec=resolved_sampling_spec,
        save_tiling_result=sampling_mod.save_tiling_result,
    )
    incompatible_tiling = sampling_mod.TilingConfig(
        target_spacing_um=0.75,
        target_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.1,
        backend="asap",
    )

    with pytest.raises(ValueError, match="target_spacing_um mismatch"):
        sampling_support_mod.validate_sampling_artifact_row(
            row={
                "sample_id": "sample-1",
                "annotation": "tumor",
                "image_path": "slide.svs",
                "annotation_mask_path": "mask.tif",
                "sampling_status": "success",
                "num_tiles": 1,
                "coordinates_npz_path": str(artifacts.coordinates_npz_path),
                "coordinates_meta_path": str(artifacts.coordinates_meta_path),
            },
            whole_slide=SimpleNamespace(
                sample_id="sample-1",
                image_path=Path("slide.svs"),
                mask_path=Path("mask.tif"),
            ),
            tiling_config=incompatible_tiling,
            segmentation_config=segmentation_config,
            filter_config=filter_config,
            expected_tissue_threshold=0.7,
            selection_strategy=sampling_mod.CoordinateSelectionStrategy.JOINT_SAMPLING,
        )


def test_coordinate_extraction_result_requires_stride_metadata():
    with pytest.raises(TypeError):
        sampling_mod.CoordinateExtractionResult(
            contour_indices=[],
            tissue_percentages=[],
            x=np.array([], dtype=np.int64),
            y=np.array([], dtype=np.int64),
            read_level=0,
            read_spacing_um=0.5,
            read_tile_size_px=256,
            resize_factor=1.0,
            tile_size_lv0=256,
        )
