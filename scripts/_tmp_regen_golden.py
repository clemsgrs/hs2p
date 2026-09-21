"""TEMPORARY (#210): regenerate the golden tiling artifacts inside the CI Docker image.

Reuses the golden regression test's own config so the two cannot drift. Removed before merge.
"""
import json
import sys
from pathlib import Path

from hs2p.wsi.reader import open_slide
from tests.test_fixture_artifacts_regression import (
    GOLDEN_MASK_BACKEND,
    _run_and_save_tiles,
)

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
base = Path("/workspace/tests/fixtures/input")
wsi_path, mask_path = base / "test-wsi.tif", base / "test-mask.tif"

_run_and_save_tiles(
    wsi_path=wsi_path,
    mask_path=mask_path,
    backend="asap",
    tissue_pct=0.1,
    output_dir=out_dir,
)

report = {}
for name, path, backend in (
    ("slide_asap", wsi_path, "asap"),
    ("mask_cucim", mask_path, GOLDEN_MASK_BACKEND),
):
    reader = open_slide(path, backend=backend)
    report[name] = {
        "spacing": float(reader.spacing),
        "level_downsamples": [
            [float(v) for v in ds] if isinstance(ds, (tuple, list)) else float(ds)
            for ds in reader.level_downsamples
        ],
        "level_dimensions": [[int(v) for v in dims] for dims in reader.level_dimensions],
    }
    reader.close()
(out_dir / "reader-report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps(report, indent=2, sort_keys=True))
print((out_dir / "test-wsi.coordinates.meta.json").read_text())
