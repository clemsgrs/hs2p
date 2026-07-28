from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

from hs2p.api import BatchPartialFailureWarning, SlideSpec, TilingArtifacts
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
        speed=SimpleNamespace(num_workers=1),
        tiling=SimpleNamespace(
            read_coordinates_from=None,
            backend="asap",
            independent_sampling=False,
            params=SimpleNamespace(
                requested_spacing_um=0.5,
                requested_tile_size_px=256,
                tolerance=0.05,
                overlap=0.0,
            ),
            preview=SimpleNamespace(
                save_mask_preview=False,
                save_tiling_preview=False,
                downsample=32,
                tissue_contour_color=[37, 94, 59],
                mask_overlay_alpha=0.5,
            ),
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
            masks=SimpleNamespace(
                pixel_mapping=[{"background": 0}, {"tissue": 1}],
                min_coverage=[{"background": None}, {"tissue": 0.01}],
                colors=None,
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
    assert captured["jpeg_backend"] == "pil"
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


def test_cli_parse_args_accepts_positional_config_file(tmp_path: Path):
    config_path = tmp_path / "config.yaml"

    args = tiling_mod.parse_args(
        [str(config_path), "output_dir=/tmp/out", "speed.num_workers=4"]
    )

    assert args.config_file == str(config_path)
    assert args.opts == ["output_dir=/tmp/out", "speed.num_workers=4"]


def test_cli_entrypoint_invokes_main(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    captured = {}

    def _fake_main(args):
        captured["args"] = args

    monkeypatch.setattr(tiling_mod, "main", _fake_main)

    exit_code = tiling_mod.entrypoint([str(config_path), "output_dir=/tmp/out"])

    assert exit_code == 0
    assert captured["args"].config_file == str(config_path)
    assert captured["args"].opts == ["output_dir=/tmp/out"]


def test_cli_entrypoint_exits_nonzero_after_partial_failure_manifest_is_persisted(
    monkeypatch,
    capsys,
    tmp_path: Path,
):
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text(
        "sample_id,image_path,mask_path\n"
        "slide-1,slide-1.svs,slide-1-mask.png\n"
        "nan,slide-2.svs,slide-2-mask.png\n"
        "slide-3,slide-3.svs,slide-3-mask.png\n"
    )
    cfg = _base_cfg(tmp_path, csv_path)
    attempted = []

    monkeypatch.setattr(tiling_mod, "setup", lambda args: cfg)

    def _completed_partial_batch(whole_slides, *, output_dir, **kwargs):
        del kwargs
        attempted.extend(slide.sample_id for slide in whole_slides)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "sample_id": "slide-1",
                    "tiling_status": "success",
                    "error": np.nan,
                    "traceback": np.nan,
                },
                {
                    "sample_id": "nan",
                    "tiling_status": "failed",
                    "error": "damaged header",
                    "traceback": "Traceback...\nRuntimeError: damaged header",
                },
                {
                    "sample_id": "slide-3",
                    "tiling_status": "success",
                    "error": np.nan,
                    "traceback": np.nan,
                },
            ]
        ).to_csv(output_dir / "process_list.csv", index=False)
        warnings.warn(
            BatchPartialFailureWarning([("nan", "damaged header")]),
            stacklevel=2,
        )
        return [
            TilingArtifacts(
                sample_id="slide-1",
                coordinates_npz_path=output_dir / "tiles" / "slide-1.coordinates.npz",
                coordinates_meta_path=output_dir
                / "tiles"
                / "slide-1.coordinates.meta.json",
                num_tiles=2,
            ),
            TilingArtifacts(
                sample_id="slide-3",
                coordinates_npz_path=output_dir / "tiles" / "slide-3.coordinates.npz",
                coordinates_meta_path=output_dir
                / "tiles"
                / "slide-3.coordinates.meta.json",
                num_tiles=2,
            ),
        ]

    monkeypatch.setattr(tiling_mod, "tile_slides", _completed_partial_batch)

    with pytest.warns(BatchPartialFailureWarning):
        exit_code = tiling_mod.entrypoint([str(tmp_path / "config.yaml")])

    assert attempted[0] == "slide-1"
    assert pd.isna(attempted[1])
    assert attempted[2] == "slide-3"
    process_df = pd.read_csv(Path(cfg.output_dir) / "process_list.csv")
    assert process_df["tiling_status"].tolist() == ["success", "failed", "success"]
    assert pd.isna(process_df.loc[1, "sample_id"])
    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Run completed with 1 failed slide." in output
    assert "successfully" not in output


def test_cli_entrypoint_exits_zero_when_every_persisted_slide_succeeded(
    monkeypatch,
    capsys,
    tmp_path: Path,
):
    csv_path = _write_csv(tmp_path)
    cfg = _base_cfg(tmp_path, csv_path)

    monkeypatch.setattr(tiling_mod, "setup", lambda args: cfg)

    def _completed_successful_batch(whole_slides, *, output_dir, **kwargs):
        del kwargs
        assert [slide.sample_id for slide in whole_slides] == ["slide-1"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "sample_id": "slide-1",
                    "tiling_status": "success",
                    "error": np.nan,
                    "traceback": np.nan,
                }
            ]
        ).to_csv(output_dir / "process_list.csv", index=False)
        return [
            TilingArtifacts(
                sample_id="slide-1",
                coordinates_npz_path=output_dir / "tiles" / "slide-1.coordinates.npz",
                coordinates_meta_path=output_dir
                / "tiles"
                / "slide-1.coordinates.meta.json",
                num_tiles=2,
            )
        ]

    monkeypatch.setattr(tiling_mod, "tile_slides", _completed_successful_batch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exit_code = tiling_mod.entrypoint([str(tmp_path / "config.yaml")])

    assert exit_code == 0
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, BatchPartialFailureWarning)
    ]
    output = capsys.readouterr().out
    assert "Run finished successfully." in output
