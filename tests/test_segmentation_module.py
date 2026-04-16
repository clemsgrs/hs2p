import logging
from pathlib import Path

import pytest
import numpy as np

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


def test_sam2_log_filter_suppresses_predictor_info_but_keeps_httpx_noise():
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

    assert segmentation_mod._should_keep_sam2_log(predictor_record) is False
    assert segmentation_mod._should_keep_sam2_log(httpx_record) is True
    assert segmentation_mod._should_keep_sam2_log(warning_record) is True


def test_sam2_predictor_is_cached_per_process(monkeypatch, tmp_path: Path):
    checkpoint_path = tmp_path / "sam2-model.pth"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path = tmp_path / "sam2.yaml"
    config_path.write_text("model: {}\n")

    segmentation_mod._build_sam2_predictor.cache_clear()
    build_calls = []

    def _fake_init(self, *, checkpoint_path, config_path, device):
        build_calls.append((checkpoint_path, config_path, device))
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.input_size = segmentation_mod.DEFAULT_SAM2_INPUT_SIZE
        self._predictor = object()

    monkeypatch.setattr(segmentation_mod._Sam2Predictor, "__init__", _fake_init)

    predictor_a = segmentation_mod._build_sam2_predictor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
    )
    predictor_b = segmentation_mod._build_sam2_predictor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
    )

    assert predictor_a is predictor_b
    assert build_calls == [(checkpoint_path, config_path, "cpu")]
    segmentation_mod._build_sam2_predictor.cache_clear()


def test_sam2_predictor_letterboxes_non_square_images_before_predicting(
    monkeypatch, tmp_path: Path
):
    checkpoint_path = tmp_path / "sam2-model.pth"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path = tmp_path / "sam2.yaml"
    config_path.write_text("model: {}\n")

    captured = {}

    class _FakePredictor:
        def set_image(self, image):
            captured["set_image_shape"] = image.shape

        def predict(
            self,
            *,
            point_coords,
            point_labels,
            box,
            multimask_output,
            return_logits,
        ):
            del point_coords, point_labels, multimask_output, return_logits
            captured["box"] = box
            height, width = captured["set_image_shape"][:2]
            mask = np.ones((1, height, width), dtype=np.float32)
            return mask, np.array([1.0], dtype=np.float32), np.zeros((1, 256, 256))

    monkeypatch.setattr(
        segmentation_mod._Sam2Predictor,
        "_load_predictor",
        lambda self, *, checkpoint_path, config_path, device, mask_threshold: _FakePredictor(),
    )

    predictor = segmentation_mod._Sam2Predictor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
    )

    image = np.zeros((2752, 768, 3), dtype=np.uint8)
    mask = predictor.predict_mask(image)

    assert captured["set_image_shape"] == (2752, 2752, 3)
    assert captured["box"].tolist() == [0.0, 0.0, 768.0, 2752.0]
    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
