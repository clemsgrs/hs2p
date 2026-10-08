from pathlib import Path
from types import SimpleNamespace

import pytest

import hs2p.__main__ as tiling_mod
from hs2p.utils.setup import get_cfg_from_args


LEGACY_TUMOR_SAMPLING_PARAMS = (
    "  sampling_params:\n"
    "    pixel_mapping:\n"
    "      - background: 0\n"
    "      - tissue: 1\n"
    "      - tumor: 2\n"
    "    color_mapping:\n"
    "      - background: null\n"
    "      - tissue: null\n"
    "      - tumor: [255, 0, 0]\n"
    "    tissue_percentage:\n"
    "      - background: null\n"
    "      - tissue: null\n"
    "      - tumor: 0.5\n"
)


def _cli_args(config_path: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config_file=str(config_path),
        output_dir=str(output_dir),
        opts=[],
        skip_datetime=True,
        skip_logging=True,
    )


def test_get_cfg_from_args_merges_mask_coverage_mapping(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "csv: slides.csv\n"
        "tiling:\n"
        "  masks:\n"
        "    min_coverage:\n"
        "      tissue: 0.1\n"
    )

    cfg = get_cfg_from_args(
        SimpleNamespace(
            config_file=str(config_path),
            output_dir=None,
            opts=[],
            skip_datetime=True,
            skip_logging=True,
        )
    )

    assert cfg.tiling.masks.min_coverage["background"] is None
    assert cfg.tiling.masks.min_coverage["tissue"] == 0.1


def test_cli_rejects_legacy_tumor_sampling_params_before_tile_extraction(
    monkeypatch, tmp_path: Path
):
    """A legacy ``tiling.sampling_params`` tumor-sampling config used to be silently
    ignored (the merged default masks section won), tiling binary tissue instead of
    tumor. The CLI must refuse it, naming the retired key and its replacement."""
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text("sample_id,image_path,mask_path\nslide-1,slide-1.svs,slide-1.tif\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"csv: {csv_path}\ntiling:\n" + LEGACY_TUMOR_SAMPLING_PARAMS)

    # Real file + CLI merge, without the logging/output side effects of setup().
    monkeypatch.setattr(tiling_mod, "setup", get_cfg_from_args)

    def _tile_slides_must_not_run(*args, **kwargs):
        raise AssertionError("tile extraction started for a rejected config")

    monkeypatch.setattr(tiling_mod, "tile_slides", _tile_slides_must_not_run)

    with pytest.raises(ValueError, match=r"tiling\.sampling_params") as excinfo:
        tiling_mod.main(_cli_args(config_path, tmp_path / "output"))
    assert "tiling.masks" in str(excinfo.value)
