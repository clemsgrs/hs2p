from pathlib import Path
from types import SimpleNamespace

from hs2p.utils.setup import get_cfg_from_args


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
