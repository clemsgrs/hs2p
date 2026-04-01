import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hs2p.api import SlideSpec, TilingArtifacts
import hs2p.cli.tiling as tiling_mod


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


def _write_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text(
        "sample_id,image_path,tissue_mask_path,annotation_mask_path\n"
        "slide-1,slide-1.svs,slide-1-mask.png,slide-1-mask.png\n"
    )
    return csv_path


def _base_cfg(tmp_path: Path, csv_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        csv=str(csv_path),
        seed=0,
        output_dir=str(tmp_path / "output"),
        resume=False,
        save_previews=False,
        save_tiles=False,
        speed=SimpleNamespace(num_workers=1, jpeg_backend="turbojpeg"),
        tiling=SimpleNamespace(
            read_coordinates_from=None,
            backend="asap",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params={
                "downsample": 64,
                "sthresh": 8,
                "sthresh_up": 255,
                "mthresh": 7,
                "close": 4,
                "use_otsu": False,
                "use_hsv": True,
            },
            filter_params={
                "ref_tile_size": 16,
                "a_t": 4,
                "a_h": 2,
                "filter_white": False,
                "filter_black": False,
                "white_threshold": 220,
                "black_threshold": 25,
                "fraction_threshold": 0.9,
            },
            preview=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(
                independent_sampling=True,
                pixel_mapping=[{"background": 0}, {"tumor": 1}],
                tissue_percentage=[{"background": None}, {"tumor": 0.1}],
                color_mapping=None,
            ),
        ),
    )


def test_tiling_main_smoke_uses_current_schema_and_manifest(
    monkeypatch, tmp_path: Path
):
    csv_path = _write_csv(tmp_path)
    cfg = _base_cfg(tmp_path, csv_path)
    captured = {}

    monkeypatch.setattr(tiling_mod, "setup", lambda args: cfg)

    def _fake_tile_slides(
        whole_slides,
        *,
        tiling,
        segmentation,
        filtering,
        preview,
        output_dir,
        num_workers,
        resume,
        read_coordinates_from,
        save_tiles,
        jpeg_backend,
    ):
        del tiling, segmentation, filtering, preview, num_workers, resume, read_coordinates_from
        captured["whole_slides"] = whole_slides
        captured["save_tiles"] = save_tiles
        captured["jpeg_backend"] = jpeg_backend
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        process_df = pd.DataFrame(
            [
                {
                    "sample_id": "slide-1",
                    "image_path": "slide-1.svs",
                    "tissue_mask_path": "slide-1-mask.png",
                    "tiling_status": "success",
                    "num_tiles": 2,
                    "coordinates_npz_path": str(
                        output_dir / "tiles" / "slide-1.coordinates.npz"
                    ),
                    "coordinates_meta_path": str(
                        output_dir / "tiles" / "slide-1.coordinates.meta.json"
                    ),
                    "tiles_tar_path": np.nan,
                    "error": np.nan,
                    "traceback": np.nan,
                }
            ]
        )
        process_df.to_csv(output_dir / "process_list.csv", index=False)
        return [
            TilingArtifacts(
                sample_id="slide-1",
                coordinates_npz_path=output_dir / "tiles" / "slide-1.coordinates.npz",
                coordinates_meta_path=output_dir / "tiles" / "slide-1.coordinates.meta.json",
                num_tiles=2,
            )
        ]

    monkeypatch.setattr(tiling_mod, "tile_slides", _fake_tile_slides)

    tiling_mod.main(SimpleNamespace())

    assert captured["whole_slides"] == [
        SlideSpec(
            sample_id="slide-1",
            image_path=Path("slide-1.svs"),
            mask_path=Path("slide-1-mask.png"),
        )
    ]
    assert captured["save_tiles"] is False
    assert captured["jpeg_backend"] == "turbojpeg"
    process_df = pd.read_csv(Path(cfg.output_dir) / "process_list.csv")
    assert list(process_df.columns) == [
        "sample_id",
        "image_path",
        "tissue_mask_path",
        "tiling_status",
        "num_tiles",
        "coordinates_npz_path",
        "coordinates_meta_path",
        "tiles_tar_path",
        "error",
        "traceback",
    ]
    row = process_df.to_dict(orient="records")[0]
    assert row["sample_id"] == "slide-1"
    assert row["image_path"] == "slide-1.svs"
    assert row["tissue_mask_path"] == "slide-1-mask.png"
    assert row["tiling_status"] == "success"
    assert row["num_tiles"] == 2


def test_sampling_main_smoke_uses_current_schema_and_manifest(
    monkeypatch, tmp_path: Path
):
    sampling_mod = _import_sampling_module()
    csv_path = _write_csv(tmp_path)
    cfg = _base_cfg(tmp_path, csv_path)

    monkeypatch.setattr(sampling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(sampling_mod.mp, "cpu_count", lambda: 1)

    class _FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap(self, fn, args_list):
            for args in args_list:
                yield fn(args)

    monkeypatch.setattr(sampling_mod.mp, "Pool", _FakePool)

    def _fake_process_slide_wrapper(kwargs):
        annotation_dir = Path(kwargs["cfg"].output_dir) / "tiles" / "tumor"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            annotation_dir / f"{kwargs['sample_id']}.coordinates.npz",
            tile_index=np.array([0], dtype=np.int32),
            x=np.array([10], dtype=np.int64),
            y=np.array([20], dtype=np.int64),
        )
        meta_path = annotation_dir / f"{kwargs['sample_id']}.coordinates.meta.json"
        meta_path.write_text(
            '{"provenance":{"sample_id":"slide-1","image_path":"slide-1.svs","mask_path":"slide-1-mask.png","backend":"asap","requested_backend":"asap"},"slide":{"dimensions":[256,256],"base_spacing_um":0.5,"level_downsamples":[1.0]},"tiling":{"requested_tile_size_px":256,"requested_spacing_um":0.5,"read_level":0,"effective_tile_size_px":256,"effective_spacing_um":0.5,"tile_size_lv0":256,"tolerance":0.05,"step_px_lv0":256,"overlap":0.0,"min_tissue_fraction":0.1,"is_within_tolerance":true,"n_tiles":1},"segmentation":{"tissue_method":"unknown","seg_downsample":64,"seg_level":0,"seg_spacing_um":0.5,"sthresh":8,"sthresh_up":255,"mthresh":7,"close":4,"use_otsu":false,"use_hsv":true,"mask_path":"slide-1-mask.png","ref_tile_size_px":16,"tissue_mask_tissue_value":null,"mask_level":null,"mask_spacing_um":null},"filtering":{"a_t":4,"a_h":2,"filter_white":false,"filter_black":false,"white_threshold":220,"black_threshold":25,"fraction_threshold":0.9},"artifact":{"coordinate_space":"level0_px","tile_order":"x_then_y","annotation":"tumor","selection_strategy":"independent_sampling","output_mode":"per_annotation"}}\n'
        )
        return kwargs["sample_id"], {
            "status": "success",
            "rows": [
                {
                    "sample_id": kwargs["sample_id"],
                    "annotation": "tumor",
                    "image_path": "slide-1.svs",
                    "annotation_mask_path": "slide-1-mask.png",
                    "sampling_status": "success",
                    "num_tiles": 1,
                    "coordinates_npz_path": str(annotation_dir / f"{kwargs['sample_id']}.coordinates.npz"),
                    "coordinates_meta_path": str(meta_path),
                    "error": np.nan,
                    "traceback": np.nan,
                }
            ],
        }

    monkeypatch.setattr(
        sampling_mod, "process_slide_wrapper", _fake_process_slide_wrapper
    )

    sampling_mod.main(SimpleNamespace())

    process_df = pd.read_csv(Path(cfg.output_dir) / "process_list.csv")
    assert list(process_df.columns) == [
        "sample_id",
        "annotation",
        "image_path",
        "annotation_mask_path",
        "sampling_status",
        "num_tiles",
        "coordinates_npz_path",
        "coordinates_meta_path",
        "error",
        "traceback",
    ]
    row = process_df.to_dict(orient="records")[0]
    assert row["sample_id"] == "slide-1"
    assert row["annotation"] == "tumor"
    assert row["image_path"] == "slide-1.svs"
    assert row["annotation_mask_path"] == "slide-1-mask.png"
    assert row["sampling_status"] == "success"
    assert row["num_tiles"] == 1
    assert row["coordinates_npz_path"].endswith("tiles/tumor/slide-1.coordinates.npz")
    assert row["coordinates_meta_path"].endswith("tiles/tumor/slide-1.coordinates.meta.json")
    assert pd.isna(row["error"])
    assert pd.isna(row["traceback"])
    assert (
        Path(cfg.output_dir) / "tiles" / "tumor" / "slide-1.coordinates.npz"
    ).is_file()
