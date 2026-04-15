import logging
from pathlib import Path

import pytest

import hs2p.segmentation as segmentation_mod


def test_sam2_predictor_prefers_explicit_local_paths(tmp_path: Path, monkeypatch):
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
    )

    assert predictor.checkpoint_path == checkpoint_path
    assert predictor.config_path == config_path
    assert captured["checkpoint_path"] == checkpoint_path
    assert captured["config_path"] == config_path
    assert captured["mask_threshold"] == segmentation_mod.DEFAULT_SAM2_MASK_THRESHOLD


def test_sam2_predictor_downloads_default_assets_when_local_paths_are_missing(
    tmp_path: Path, monkeypatch
):
    downloaded_checkpoint = tmp_path / "downloaded-model.pth"
    downloaded_checkpoint.write_bytes(b"checkpoint")
    downloaded_config = tmp_path / "downloaded-config.yaml"
    downloaded_config.write_text("model: {}\n")
    captured = []

    class _FakeHub:
        @staticmethod
        def hf_hub_download(*, repo_id, filename):
            captured.append((repo_id, filename))
            if filename == segmentation_mod.DEFAULT_SAM2_MODEL_FILENAME:
                return str(downloaded_checkpoint)
            if filename == segmentation_mod.DEFAULT_SAM2_CONFIG_FILENAME:
                return str(downloaded_config)
            raise AssertionError(f"Unexpected filename {filename}")

    monkeypatch.setattr(
        segmentation_mod._Sam2Predictor,
        "_load_predictor",
        lambda self, *, checkpoint_path, config_path, device, mask_threshold: object(),
    )
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHub)

    predictor = segmentation_mod._Sam2Predictor(
        checkpoint_path=None,
        config_path=None,
        device="cpu",
    )

    assert predictor.checkpoint_path == downloaded_checkpoint
    assert predictor.config_path == downloaded_config
    assert captured == [
        (
            segmentation_mod.DEFAULT_SAM2_MODEL_REPO,
            segmentation_mod.DEFAULT_SAM2_MODEL_FILENAME,
        ),
        (
            segmentation_mod.DEFAULT_SAM2_MODEL_REPO,
            segmentation_mod.DEFAULT_SAM2_CONFIG_FILENAME,
        ),
    ]


def test_sam2_predictor_requires_huggingface_hub_for_automatic_asset_download(
    monkeypatch,
):
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
            config_path=None,
            device="cpu",
        )


def test_sam2_log_filter_keeps_predictor_messages_but_silences_httpx_noise():
    predictor_record = logging.LogRecord(
        name="root",
        level=logging.INFO,
        pathname="/tmp/site-packages/sam2/sam2_image_predictor.py",
        lineno=102,
        msg="For numpy array image, we assume (HxWxC) format",
        args=(),
        exc_info=None,
    )
    httpx_record = logging.LogRecord(
        name="httpx._client",
        level=logging.INFO,
        pathname="/tmp/site-packages/httpx/_client.py",
        lineno=1025,
        msg="HTTP Request: HEAD ...",
        args=(),
        exc_info=None,
    )
    warning_record = logging.LogRecord(
        name="httpx._client",
        level=logging.WARNING,
        pathname="/tmp/site-packages/httpx/_client.py",
        lineno=1025,
        msg="warning",
        args=(),
        exc_info=None,
    )

    assert segmentation_mod._is_sam2_predictor_log(predictor_record) is True
    assert segmentation_mod._is_sam2_predictor_log(httpx_record) is False
    assert segmentation_mod._is_sam2_predictor_log(warning_record) is True
