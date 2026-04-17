from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hs2p.api import SlideSpec, TilingArtifacts
import hs2p.__main__ as tiling_mod


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
        save_tiles=False,
        speed=SimpleNamespace(num_workers=1, jpeg_backend="turbojpeg"),
        tiling=SimpleNamespace(
            read_coordinates_from=None,
            backend="asap",
            params=SimpleNamespace(
                requested_spacing_um=0.5,
                requested_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            preview=SimpleNamespace(save=False, downsample=32),
            seg_params={
                "method": "hsv",
                "downsample": 64,
                "sthresh": 8,
                "sthresh_up": 255,
                "mthresh": 7,
                "close": 4,
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
                    "annotation": "tissue",
                    "image_path": "slide-1.svs",
                    "mask_path": "slide-1-mask.png",
                    "requested_backend": "asap",
                    "backend": "asap",
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
        "annotation",
        "image_path",
        "mask_path",
        "requested_backend",
        "backend",
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
    assert row["mask_path"] == "slide-1-mask.png"
    assert row["tiling_status"] == "success"
    assert row["num_tiles"] == 2
