"""Smoke-test tile saving from an installed base hs2p wheel."""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import io
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import hs2p
from hs2p.api import extract_tiles_to_tar
from hs2p.configs import default_config


def main() -> None:
    assert "site-packages" in str(Path(hs2p.__file__).resolve())
    assert importlib.util.find_spec("turbojpeg") is None
    assert default_config.speed.jpeg_backend == "pil"

    metadata = importlib.metadata.metadata("hs2p")
    extras = set(metadata.get_all("Provides-Extra") or [])
    requirements = metadata.get_all("Requires-Dist") or []
    assert "turbojpeg" in extras
    assert any(
        requirement.startswith("PyTurboJPEG")
        and 'extra == "turbojpeg"' in requirement
        for requirement in requirements
    )
    assert any(
        requirement.startswith("PyTurboJPEG") and 'extra == "all"' in requirement
        for requirement in requirements
    )

    tile = np.empty((8, 8, 3), dtype=np.uint8)
    tile[:] = (12, 34, 56)
    result = SimpleNamespace(
        sample_id="wheel-smoke",
        read_tile_size_px=8,
        requested_tile_size_px=8,
        tile_index=np.array([0], dtype=np.int32),
        x=np.array([0], dtype=np.int64),
        y=np.array([0], dtype=np.int64),
    )
    record = SimpleNamespace(tile_arr=tile, tile_index=0, x=0, y=0)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        with patch(
            "hs2p.tiling.tar.iter_tile_records_from_result",
            return_value=iter([record]),
        ):
            tar_path, output_result = extract_tiles_to_tar(result, output_dir)

        assert output_result is result
        with tarfile.open(tar_path) as archive:
            assert archive.getnames() == ["000000.jpg"]
            member = archive.extractfile("000000.jpg")
            assert member is not None
            with Image.open(io.BytesIO(member.read())) as image:
                assert image.format == "JPEG"
                assert image.mode == "RGB"
                assert image.size == (8, 8)

        manifest_path = output_dir / "tiles" / "wheel-smoke.tiles.manifest.csv"
        with manifest_path.open(newline="") as handle:
            assert list(csv.DictReader(handle)) == [
                {"tile_index": "0", "x": "0", "y": "0"}
            ]

    print("Clean-wheel Pillow tile smoke check passed.")


if __name__ == "__main__":
    main()
