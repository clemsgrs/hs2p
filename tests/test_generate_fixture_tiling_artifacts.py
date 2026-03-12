from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hs2p.api import TilingArtifacts


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_test_fixture_tiling_artifacts.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "generate_test_fixture_tiling_artifacts",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_fixture_artifacts_uses_public_tiling_api(monkeypatch, tmp_path: Path):
    module = _load_script_module()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "test-wsi.tif").write_bytes(b"wsi")
    (input_dir / "test-mask.tif").write_bytes(b"mask")
    captured = {}

    def _fake_tile_slide(whole_slide, *, tiling, segmentation, filtering, num_workers):
        captured["whole_slide"] = whole_slide
        captured["tiling"] = tiling
        captured["segmentation"] = segmentation
        captured["filtering"] = filtering
        captured["num_workers"] = num_workers
        return object()

    def _fake_save_tiling_result(result, *, output_dir):
        captured["result"] = result
        captured["output_dir"] = output_dir
        return TilingArtifacts(
            sample_id="generated-sample",
            tiles_npz_path=Path(output_dir) / "coordinates" / "generated-sample.tiles.npz",
            tiles_meta_path=Path(output_dir)
            / "coordinates"
            / "generated-sample.tiles.meta.json",
            num_tiles=2,
        )

    monkeypatch.setattr(module, "tile_slide", _fake_tile_slide)
    monkeypatch.setattr(module, "save_tiling_result", _fake_save_tiling_result)

    artifacts = module.generate_fixture_artifacts(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_id="generated-sample",
        backend="openslide",
        tissue_threshold=0.25,
        num_workers=3,
    )

    assert artifacts.sample_id == "generated-sample"
    assert captured["whole_slide"].sample_id == "generated-sample"
    assert captured["whole_slide"].image_path == input_dir / "test-wsi.tif"
    assert captured["whole_slide"].mask_path == input_dir / "test-mask.tif"
    assert captured["tiling"].backend == "openslide"
    assert captured["tiling"].target_spacing_um == pytest.approx(0.5)
    assert captured["tiling"].target_tile_size_px == 224
    assert captured["tiling"].tissue_threshold == pytest.approx(0.25)
    assert captured["segmentation"].downsample == 64
    assert captured["filtering"].ref_tile_size == 224
    assert captured["num_workers"] == 3
    assert captured["result"] is not None
    assert captured["output_dir"] == output_dir


def test_generate_fixture_artifacts_raises_when_fixture_inputs_are_missing(tmp_path: Path):
    module = _load_script_module()
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing required fixture inputs"):
        module.generate_fixture_artifacts(input_dir=input_dir, output_dir=tmp_path / "out")
