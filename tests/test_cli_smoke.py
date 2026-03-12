from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hs2p.api import SlideSpec, TilingArtifacts
import hs2p.tiling as tiling_mod


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


def _write_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text(
        "sample_id,image_path,mask_path\n"
        "slide-1,slide-1.svs,slide-1-mask.png\n"
    )
    return csv_path


def _base_cfg(tmp_path: Path, csv_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        csv=str(csv_path),
        seed=0,
        output_dir=str(tmp_path / "output"),
        resume=False,
        visualize=False,
        speed=SimpleNamespace(num_workers=1),
        tiling=SimpleNamespace(
            read_tiles_from=None,
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
                "max_n_holes": 8,
                "filter_white": False,
                "filter_black": False,
                "white_threshold": 220,
                "black_threshold": 25,
                "fraction_threshold": 0.9,
            },
            visu_params=SimpleNamespace(downsample=32),
            sampling_params=SimpleNamespace(
                independant_sampling=True,
                pixel_mapping=[{"tumor": 1}],
                tissue_percentage=[{"tumor": 0.1}],
                color_mapping=None,
            ),
        ),
    )


def test_tiling_main_smoke_uses_current_schema_and_manifest(monkeypatch, tmp_path: Path):
    csv_path = _write_csv(tmp_path)
    cfg = _base_cfg(tmp_path, csv_path)
    captured = {}

    monkeypatch.setattr(tiling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(tiling_mod, "fix_random_seeds", lambda seed: None)

    def _fake_tile_slides(
        whole_slides,
        *,
        tiling,
        segmentation,
        filtering,
        qc,
        output_dir,
        num_workers,
        resume,
        read_tiles_from,
    ):
        captured["whole_slides"] = whole_slides
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        process_df = pd.DataFrame(
            [
                {
                    "sample_id": "slide-1",
                    "image_path": "slide-1.svs",
                    "mask_path": "slide-1-mask.png",
                    "tiling_status": "success",
                    "num_tiles": 2,
                    "tiles_npz_path": str(output_dir / "coordinates" / "slide-1.tiles.npz"),
                    "tiles_meta_path": str(
                        output_dir / "coordinates" / "slide-1.tiles.meta.json"
                    ),
                    "error": np.nan,
                    "traceback": np.nan,
                }
            ]
        )
        process_df.to_csv(output_dir / "process_list.csv", index=False)
        return [
            TilingArtifacts(
                sample_id="slide-1",
                tiles_npz_path=output_dir / "coordinates" / "slide-1.tiles.npz",
                tiles_meta_path=output_dir / "coordinates" / "slide-1.tiles.meta.json",
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
    process_df = pd.read_csv(Path(cfg.output_dir) / "process_list.csv")
    assert list(process_df.columns) == [
        "sample_id",
        "image_path",
        "mask_path",
        "tiling_status",
        "num_tiles",
        "tiles_npz_path",
        "tiles_meta_path",
        "error",
        "traceback",
    ]
    row = process_df.to_dict(orient="records")[0]
    assert row["sample_id"] == "slide-1"
    assert row["image_path"] == "slide-1.svs"
    assert row["mask_path"] == "slide-1-mask.png"
    assert row["tiling_status"] == "success"
    assert row["num_tiles"] == 2


def test_sampling_main_smoke_uses_current_schema_and_manifest(monkeypatch, tmp_path: Path):
    sampling_mod = _import_sampling_module()
    csv_path = _write_csv(tmp_path)
    cfg = _base_cfg(tmp_path, csv_path)

    monkeypatch.setattr(sampling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(sampling_mod, "fix_random_seeds", lambda seed: None)
    monkeypatch.setattr(sampling_mod.mp, "cpu_count", lambda: 1)
    monkeypatch.setattr(sampling_mod.tqdm, "tqdm", lambda iterable, **kwargs: iterable)

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
        annotation_dir = Path(kwargs["cfg"].output_dir) / "coordinates" / "tumor"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            annotation_dir / f"{kwargs['sample_id']}.tiles.npz",
            tile_index=np.array([0], dtype=np.int32),
            x=np.array([10], dtype=np.int64),
            y=np.array([20], dtype=np.int64),
        )
        return kwargs["sample_id"], {"status": "success"}

    monkeypatch.setattr(sampling_mod, "process_slide_wrapper", _fake_process_slide_wrapper)

    sampling_mod.main(SimpleNamespace())

    process_df = pd.read_csv(Path(cfg.output_dir) / "process_list.csv")
    assert list(process_df.columns) == [
        "sample_id",
        "image_path",
        "mask_path",
        "sampling_status",
        "error",
        "traceback",
    ]
    row = process_df.to_dict(orient="records")[0]
    assert row["sample_id"] == "slide-1"
    assert row["image_path"] == "slide-1.svs"
    assert row["mask_path"] == "slide-1-mask.png"
    assert row["sampling_status"] == "success"
    assert pd.isna(row["error"])
    assert pd.isna(row["traceback"])
    assert (
        Path(cfg.output_dir) / "coordinates" / "tumor" / "slide-1.tiles.npz"
    ).is_file()
