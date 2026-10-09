from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hs2p.tiling.orchestration as orchestration_mod
import hs2p.preprocessing as preprocessing_mod
from hs2p.api import (
    BatchPartialFailureWarning,
    SlideSpec,
    TilingArtifacts,
    tile_slides,
)
from hs2p.configs import FilterConfig, SegmentationConfig, TilingConfig


class RecordingReporter:
    def __init__(self):
        self.events = []
        self.log_lines = []

    def emit(self, event):
        self.events.append(event)

    def close(self):
        return None

    def write_log(self, message: str, *, stream=None):
        self.log_lines.append(message)


def _tiling_config() -> TilingConfig:
    return TilingConfig(
        backend="asap",
        requested_spacing_um=0.5,
        requested_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        min_coverage={"tissue": 0.1},
    )


def _segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(
        method="hsv",
        downsample=64,
        sthresh=8,
        sthresh_up=255,
        mthresh=7,
        close=4,
    )


def _filter_config() -> FilterConfig:
    return FilterConfig(
        ref_tile_size=16,
        a_t=4,
        a_h=2,
        filter_white=False,
        filter_black=False,
        white_threshold=220,
        black_threshold=25,
        fraction_threshold=0.9,
    )


def test_tile_slides_emits_progress_for_reused_success_and_failure(
    monkeypatch, tmp_path: Path
):
    import hs2p.progress as progress

    reporter = RecordingReporter()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    slide_a = SlideSpec(sample_id="slide-a", image_path=Path("slide-a.svs"))
    slide_b = SlideSpec(sample_id="slide-b", image_path=Path("slide-b.svs"))
    slide_c = SlideSpec(sample_id="slide-c", image_path=Path("slide-c.svs"))

    pd.DataFrame(
        [
            {
                "sample_id": "slide-a",
                "annotation": "tissue",
                "image_path": "slide-a.svs",
                "mask_path": np.nan,
                "requested_backend": "asap",
                "backend": "asap",
                "requested_mask_backend": np.nan,
                "mask_backend": np.nan,
                "tiling_status": "success",
                "num_tiles": 2,
                "coordinates_npz_path": str(run_dir / "tiles" / "slide-a.coordinates.npz"),
                "coordinates_meta_path": str(
                    run_dir / "tiles" / "slide-a.coordinates.meta.json"
                ),
                "error": np.nan,
                "traceback": np.nan,
            }
        ]
    ).to_csv(run_dir / "process_list.csv", index=False)

    def _fake_validate_tiling_artifacts(**kwargs):
        whole_slide = kwargs["whole_slide"]
        return TilingArtifacts(
            sample_id=whole_slide.sample_id,
            coordinates_npz_path=run_dir / "tiles" / f"{whole_slide.sample_id}.coordinates.npz",
            coordinates_meta_path=run_dir
            / "tiles"
            / f"{whole_slide.sample_id}.coordinates.meta.json",
            num_tiles=2,
        )

    def _fake_compute_and_save(request):
        if request.whole_slide.sample_id == "slide-b":
            return orchestration_mod._ComputeResponse(
                input_index=request.input_index,
                whole_slide=request.whole_slide,
                ok=True,
                artifact=TilingArtifacts(
                    sample_id="slide-b",
                    coordinates_npz_path=run_dir / "tiles" / "slide-b.coordinates.npz",
                    coordinates_meta_path=run_dir / "tiles" / "slide-b.coordinates.meta.json",
                    num_tiles=1,
                ),
                requested_backend=request.tiling.requested_backend,
                backend=request.tiling.backend,
            )
        return orchestration_mod._ComputeResponse(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=False,
            requested_backend=request.tiling.requested_backend,
            backend=request.tiling.backend,
            error="boom",
            traceback_text="traceback",
        )

    def _fake_resolve_mask_for_request(request):
        return orchestration_mod._MaskResolutionResponse(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=True,
            resolved_mask=preprocessing_mod.ResolvedTissueMask(
                tissue_mask=np.zeros((8, 8), dtype=np.uint8),
                tissue_method="hsv",
                requested_seg_downsample=64,
                seg_downsample=64,
                seg_level=0,
                seg_spacing_um=0.5,
            ),
            requested_backend=request.tiling.requested_backend,
            backend=request.tiling.backend,
        )

    monkeypatch.setattr("hs2p.tiling.orchestration.validate_tiling_artifacts", _fake_validate_tiling_artifacts)
    monkeypatch.setattr(
        "hs2p.tiling.orchestration._compute_and_save_tiling_artifacts_from_request",
        _fake_compute_and_save,
    )
    monkeypatch.setattr("hs2p.tiling.orchestration._resolve_mask_for_request", _fake_resolve_mask_for_request)

    with progress.activate_progress_reporter(reporter):
        with pytest.warns(BatchPartialFailureWarning, match="slide-c: boom"):
            tile_slides(
                [slide_a, slide_b, slide_c],
                tiling=_tiling_config(),
                segmentation=_segmentation_config(),
                filtering=_filter_config(),
                output_dir=run_dir,
                resume=True,
            )

    assert [event.kind for event in reporter.events] == [
        "tissue.started",
        "tissue.progress",
        "tissue.progress",
        "tissue.finished",
        "tiling.started",
        "tiling.progress",
        "tiling.progress",
        "tiling.progress",
        "tiling.finished",
    ]
    progress_payloads = [
        event.payload for event in reporter.events if event.kind == "tiling.progress"
    ]
    assert progress_payloads == [
        {
            "total": 3,
            "completed": 1,
            "failed": 0,
            "pending": 2,
            "discovered_tiles": 2,
        },
        {
            "total": 3,
            "completed": 2,
            "failed": 0,
            "pending": 1,
            "discovered_tiles": 3,
        },
        {
            "total": 3,
            "completed": 2,
            "failed": 1,
            "pending": 0,
            "discovered_tiles": 3,
        },
    ]
    assert reporter.events[-1].payload == {
        "total": 3,
        "completed": 2,
        "failed": 1,
        "pending": 0,
        "discovered_tiles": 3,
        "output_dir": str(run_dir),
        "process_list_path": str(run_dir / "process_list.csv"),
        "zero_tile_successes": 0,
        "empty_masks": 0,
    }
