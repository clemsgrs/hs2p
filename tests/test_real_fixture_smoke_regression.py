import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from hs2p.api import FilterConfig, SegmentationConfig, TilingConfig
import hs2p.wsi as wsi_api
from hs2p.wsi import ResolvedSamplingSpec
from hs2p.wsi.utils import HasEnoughTissue, ResolvedTileGeometry
from hs2p.wsi.wsi import WholeSlideImage

pytestmark = pytest.mark.integration


def _choose_backend(wsi_path: Path) -> str:
    wsd = pytest.importorskip("wholeslidedata")
    for backend in ("asap", "openslide"):
        try:
            wsd.WholeSlideImage(wsi_path, backend=backend)
            return backend
        except Exception:
            continue
    pytest.skip("No supported WholeSlideData backend is available for TIFF fixtures")


def _run_extract(wsi_path: Path, mask_path: Path, backend: str, tissue_pct: float):
    tiling, segmentation, filtering, sampling_spec = _build_runtime_inputs(
        backend=backend,
        tissue_pct=tissue_pct,
    )
    return wsi_api.extract_coordinates(
        wsi_path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        segment_params=segmentation,
        tiling_params=tiling,
        filter_params=filtering,
        sampling_spec=sampling_spec,
        disable_tqdm=True,
        num_workers=1,
    )


def _build_runtime_inputs(
    *, backend: str, tissue_pct: float
) -> tuple[TilingConfig, SegmentationConfig, FilterConfig, ResolvedSamplingSpec]:
    tiling = TilingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.07,
        overlap=0.0,
        tissue_threshold=tissue_pct,
        drop_holes=False,
        use_padding=True,
        backend=backend,
    )
    segmentation = SegmentationConfig(
        downsample=64,
        sthresh=8,
        sthresh_up=255,
        mthresh=7,
        close=4,
        use_otsu=False,
        use_hsv=True,
    )
    filtering = FilterConfig(
        ref_tile_size=224,
        a_t=4,
        a_h=2,
        max_n_holes=8,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )
    sampling_spec = ResolvedSamplingSpec(
        pixel_mapping={"background": 0, "tissue": 1},
        color_mapping={"background": None, "tissue": None},
        tissue_percentage={"background": None, "tissue": tissue_pct},
        active_annotations=("tissue",),
    )
    return tiling, segmentation, filtering, sampling_spec


def _collect_extract_diagnostics(
    wsi_path: Path,
    mask_path: Path,
    *,
    backend: str,
    tissue_pct: float,
) -> dict[str, object]:
    tiling, segmentation, filtering, sampling_spec = _build_runtime_inputs(
        backend=backend,
        tissue_pct=tissue_pct,
    )
    wsi = WholeSlideImage(
        path=wsi_path,
        backend=backend,
        mask_path=mask_path,
        segment=False,
        segment_params=segmentation,
        sampling_spec=sampling_spec,
    )
    contours, holes = wsi.detect_contours(
        target_spacing=tiling.target_spacing_um,
        tolerance=tiling.tolerance,
        filter_params=filtering,
        annotation=None,
    )
    tile_level, tile_spacing, resize_factor = wsi._resolve_tile_read_metadata(tiling)
    target_tile_size = tiling.target_tile_size_px
    tile_size_resized = int(round(target_tile_size * resize_factor, 0))
    step_size = int(tile_size_resized * (1.0 - tiling.overlap))
    tile_downsample = (
        int(wsi.level_downsamples[tile_level][0]),
        int(wsi.level_downsamples[tile_level][1]),
    )
    tile_size_at_level_0 = (
        tile_size_resized * tile_downsample[0],
        tile_size_resized * tile_downsample[1],
    )
    mask = wsi.annotation_mask["tissue"]
    pct = wsi.annotation_pct["tissue"]

    candidate_total = 0
    retained_after_tissue = 0
    holes_per_contour: list[int] = []

    for contour, contour_holes in zip(contours, holes):
        holes_per_contour.append(len(contour_holes))
        start_x, start_y, width, height = cv2.boundingRect(contour)
        img_w, img_h = wsi.level_dimensions[0]
        if tiling.use_padding:
            stop_y = int(start_y + height)
            stop_x = int(start_x + width)
        else:
            stop_y = min(start_y + height, img_h - tile_size_at_level_0[1] + 1)
            stop_x = min(start_x + width, img_w - tile_size_at_level_0[0] + 1)

        scale = wsi.level_downsamples[wsi.seg_level]
        contour_at_seg_level = wsi.scaleContourDim(
            [contour], (1.0 / scale[0], 1.0 / scale[1])
        )[0]
        tissue_checker = HasEnoughTissue(
            contour=contour_at_seg_level,
            contour_holes=contour_holes,
            tissue_mask=mask,
            geometry=ResolvedTileGeometry(
                target_tile_size_px=target_tile_size,
                read_spacing_um=tile_spacing,
                resize_factor=resize_factor,
                seg_spacing_um=wsi.get_level_spacing(wsi.seg_level),
                level0_spacing_um=wsi.get_level_spacing(0),
            ),
            pct=pct,
        )

        ref_step_size_x = int(round(step_size * tile_downsample[0], 0))
        ref_step_size_y = int(round(step_size * tile_downsample[1], 0))
        x_range = np.arange(start_x, stop_x, step=ref_step_size_x)
        y_range = np.arange(start_y, stop_y, step=ref_step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()
        keep_flags, _ = tissue_checker.check_coordinates(coord_candidates)

        candidate_total += int(coord_candidates.shape[0])
        retained_after_tissue += int(np.asarray(keep_flags, dtype=np.uint8).sum())

    return {
        "contour_count": len(contours),
        "holes_per_contour": holes_per_contour,
        "hole_count": int(sum(holes_per_contour)),
        "candidate_tile_count": candidate_total,
        "retained_after_tissue": retained_after_tissue,
        "read_level": tile_level,
        "resize_factor": resize_factor,
    }


def _run_extract_in_subprocess(wsi_path: Path, mask_path: Path) -> dict[str, object]:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from pathlib import Path
from tests.test_real_fixture_smoke_regression import (
    _choose_backend,
    _collect_extract_diagnostics,
    _run_extract,
)

wsi_path = Path(__import__("sys").argv[1])
mask_path = Path(__import__("sys").argv[2])
backend = _choose_backend(wsi_path)
result = _run_extract(wsi_path, mask_path, backend, tissue_pct=0.10)
diagnostics = _collect_extract_diagnostics(
    wsi_path,
    mask_path,
    backend=backend,
    tissue_pct=0.10,
)
payload = {
    "backend": backend,
    "coordinates": [list(coord) for coord in result.coordinates],
    "contour_indices": [int(idx) for idx in result.contour_indices],
    "num_tiles": len(result.coordinates),
    "read_level": int(result.read_level),
    "read_spacing_um": float(result.read_spacing_um),
    "read_tile_size_px": int(result.read_tile_size_px),
    "resize_factor": float(result.resize_factor),
    "tile_size_lv0": int(result.tile_size_lv0),
    "diagnostics": diagnostics,
}
print(json.dumps(payload, sort_keys=True))
""",
            str(wsi_path),
            str(mask_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def _load_golden_num_tiles(wsi_path: Path) -> int:
    gt_meta_path = wsi_path.parent.parent / "gt" / "test-wsi.coordinates.meta.json"
    if not gt_meta_path.is_file():
        pytest.skip(f"Missing golden metadata: {gt_meta_path}")
    return int(json.loads(gt_meta_path.read_text())["num_tiles"])


def test_real_fixture_is_deterministic_in_fresh_processes(real_fixture_paths):
    wsi_path, mask_path = real_fixture_paths

    run1 = _run_extract_in_subprocess(wsi_path, mask_path)
    run2 = _run_extract_in_subprocess(wsi_path, mask_path)

    assert run1 == run2
    assert run1["read_level"] == 0
    assert run1["read_tile_size_px"] == 444
    assert run1["tile_size_lv0"] == 444
    assert run1["resize_factor"] == pytest.approx(1.984126953125)
    assert len(run1["coordinates"]) == run1["num_tiles"]
    assert len(run1["contour_indices"]) == run1["num_tiles"]


def test_real_fixture_outputs_sane_level0_coordinates(real_fixture_paths):
    wsi_path, mask_path = real_fixture_paths
    backend = _choose_backend(wsi_path)

    result = _run_extract(wsi_path, mask_path, backend, tissue_pct=0.10)

    assert len(result.coordinates) > 0
    assert len(result.coordinates) == len(result.contour_indices)
    assert len(result.coordinates) == len(result.tissue_percentages)
    assert len(result.coordinates) == len(set(result.coordinates))
    assert result.read_level >= 0
    assert result.read_spacing_um > 0
    assert result.read_tile_size_px > 0
    assert result.resize_factor > 0
    assert result.tile_size_lv0 > 0
    np.testing.assert_array_equal(
        result.x,
        np.array([x for x, _ in result.coordinates], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result.y,
        np.array([y for _, y in result.coordinates], dtype=np.int64),
    )


def test_real_fixture_stage_diagnostics_match_expected_baseline(real_fixture_paths):
    wsi_path, mask_path = real_fixture_paths
    payload = _run_extract_in_subprocess(wsi_path, mask_path)
    diagnostics = payload["diagnostics"]
    expected_num_tiles = _load_golden_num_tiles(wsi_path)

    assert diagnostics["contour_count"] == 1
    assert diagnostics["holes_per_contour"] == [1]
    assert diagnostics["hole_count"] == 1
    assert diagnostics["candidate_tile_count"] == 899
    assert diagnostics["retained_after_tissue"] == expected_num_tiles
    assert payload["num_tiles"] == expected_num_tiles
    assert diagnostics["read_level"] == 0
    assert diagnostics["resize_factor"] == pytest.approx(1.984126953125)
