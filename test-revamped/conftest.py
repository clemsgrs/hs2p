from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import hs2p.wsi.wsi as wsimod

HELPERS_DIR = Path(__file__).resolve().parent / "helpers"
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))

from fake_wsi_backend import FakeWSDFactory, make_mask_spec, make_slide_spec


@pytest.fixture
def fake_backend(monkeypatch):
    def _apply(mask_l0: np.ndarray):
        factory = FakeWSDFactory(
            slide_spec=make_slide_spec(),
            mask_spec=make_mask_spec(mask_l0),
        )
        monkeypatch.setattr(wsimod.wsd, "WholeSlideImage", factory)
        return factory

    return _apply


@pytest.fixture
def real_fixture_paths() -> tuple[Path, Path]:
    base = Path(__file__).resolve().parent.parent / "test" / "input"
    wsi_path = base / "test-wsi.tif"
    mask_path = base / "test-mask.tif"
    if not wsi_path.is_file() or not mask_path.is_file():
        pytest.skip("Real fixture TIFF files are not present")
    return wsi_path, mask_path
