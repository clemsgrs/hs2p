from pathlib import Path

import pytest

import hs2p.segmentation as segmentation_mod


def test_sam2_predictor_prefers_explicit_local_checkpoint(tmp_path: Path, monkeypatch):
    checkpoint_path = tmp_path / "sam2-model.pth"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path = tmp_path / "sam2.yaml"
    config_path.write_text("model: {}\n")

    captured = {}

    monkeypatch.setattr(
        segmentation_mod._Sam2Predictor,
        "_load_predictor",
        lambda self, *, checkpoint_path, config_path, device, mask_threshold: captured.update(
            {
                "checkpoint_path": checkpoint_path,
                "config_path": config_path,
                "device": device,
                "mask_threshold": mask_threshold,
            }
        )
        or object(),
    )

    predictor = segmentation_mod._Sam2Predictor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
        input_size=1024,
        mask_threshold=0.0,
    )

    assert predictor.checkpoint_path == checkpoint_path
    assert captured["checkpoint_path"] == checkpoint_path
    assert captured["config_path"] == config_path


def test_sam2_predictor_downloads_default_checkpoint_when_local_path_is_missing(
    tmp_path: Path, monkeypatch
):
    config_path = tmp_path / "sam2.yaml"
    config_path.write_text("model: {}\n")
    downloaded_path = tmp_path / "downloaded-model.pth"
    downloaded_path.write_bytes(b"checkpoint")
    captured = {}

    class _FakeHub:
        @staticmethod
        def hf_hub_download(*, repo_id, filename):
            captured["repo_id"] = repo_id
            captured["filename"] = filename
            return str(downloaded_path)

    monkeypatch.setattr(
        segmentation_mod._Sam2Predictor,
        "_load_predictor",
        lambda self, *, checkpoint_path, config_path, device, mask_threshold: object(),
    )
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHub)

    predictor = segmentation_mod._Sam2Predictor(
        checkpoint_path=None,
        config_path=config_path,
        device="cpu",
        input_size=1024,
        mask_threshold=0.0,
    )

    assert predictor.checkpoint_path == downloaded_path
    assert captured == {
        "repo_id": segmentation_mod.DEFAULT_SAM2_MODEL_REPO,
        "filename": segmentation_mod.DEFAULT_SAM2_MODEL_FILENAME,
    }


def test_sam2_predictor_requires_huggingface_hub_for_automatic_download(
    tmp_path: Path, monkeypatch
):
    config_path = tmp_path / "sam2.yaml"
    config_path.write_text("model: {}\n")

    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("missing huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="huggingface-hub"):
        segmentation_mod._Sam2Predictor(
            checkpoint_path=None,
            config_path=config_path,
            device="cpu",
            input_size=1024,
            mask_threshold=0.0,
        )
