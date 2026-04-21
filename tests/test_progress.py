import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import hs2p.api as api_mod
import hs2p.tiling.orchestration as orchestration_mod
import hs2p.preprocessing as preprocessing_mod
from hs2p.api import SlideSpec, TilingArtifacts, tile_slides
from hs2p.configs import FilterConfig, SegmentationConfig, TilingConfig
import hs2p.__main__ as tiling_mod





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


def _install_fake_rich_console(monkeypatch, *, is_terminal: bool):
    fake_rich = types.ModuleType("rich")
    fake_console = types.ModuleType("rich.console")

    class FakeConsole:
        def __init__(self, file=None):
            self.file = file
            self.is_terminal = is_terminal

        def log(self, *args, **kwargs):
            return None

    fake_console.Console = FakeConsole
    fake_rich.console = fake_console
    monkeypatch.setitem(sys.modules, "rich", fake_rich)
    monkeypatch.setitem(sys.modules, "rich.console", fake_console)


def _install_fake_rich_progress(monkeypatch):
    fake_progress = types.ModuleType("rich.progress")

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            self.tasks = {}
            self.next_task_id = 1

        def start(self):
            return None

        def stop(self):
            return None

        def add_task(self, description, total=None):
            task_id = self.next_task_id
            self.next_task_id += 1
            self.tasks[task_id] = {
                "description": description,
                "total": total,
                "completed": 0,
            }
            return task_id

        def update(self, task_id, **kwargs):
            self.tasks[task_id].update(kwargs)

        def remove_task(self, task_id):
            self.tasks.pop(task_id, None)

    class _Identity:
        def __init__(self, *args, **kwargs):
            pass

    fake_progress.BarColumn = _Identity
    fake_progress.MofNCompleteColumn = _Identity
    fake_progress.Progress = FakeProgress
    fake_progress.SpinnerColumn = _Identity
    fake_progress.TaskProgressColumn = _Identity
    fake_progress.TextColumn = _Identity
    fake_progress.TimeElapsedColumn = _Identity
    fake_progress.TimeRemainingColumn = _Identity
    monkeypatch.setitem(sys.modules, "rich.progress", fake_progress)


def _install_fake_rich_summary_types(monkeypatch):
    fake_panel = types.ModuleType("rich.panel")
    fake_table = types.ModuleType("rich.table")

    class FakePanel:
        @staticmethod
        def fit(table, title=None, border_style=None):
            return SimpleNamespace(table=table, title=title, border_style=border_style)

    class FakeTable:
        def __init__(self, *args, **kwargs):
            self.columns = []
            self.rows = []

        def add_column(self, *args, **kwargs):
            self.columns.append((args, kwargs))

        def add_row(self, *args):
            self.rows.append(args)

        @staticmethod
        def grid(*args, **kwargs):
            return FakeTable(*args, **kwargs)

    fake_panel.Panel = FakePanel
    fake_table.Table = FakeTable
    monkeypatch.setitem(sys.modules, "rich.panel", fake_panel)
    monkeypatch.setitem(sys.modules, "rich.table", fake_table)


def _tiling_config() -> TilingConfig:
    return TilingConfig(
        backend="asap",
        requested_spacing_um=0.5,
        requested_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.1,
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


def _base_cli_cfg(tmp_path: Path, *, resume: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        resume=resume,
        speed=SimpleNamespace(num_workers=1, jpeg_backend="turbojpeg"),
        tiling=SimpleNamespace(
            backend="asap",
            read_coordinates_from=None,
            params=SimpleNamespace(
                requested_spacing_um=0.5,
                requested_tile_size_px=256,
            ),
            preview=SimpleNamespace(save=False, downsample=32),
        ),
    )


def test_create_cli_progress_reporter_uses_rich_when_terminal(monkeypatch):
    import hs2p.progress as progress

    _install_fake_rich_console(monkeypatch, is_terminal=True)
    sentinel = object()
    monkeypatch.setattr(
        progress,
        "RichReporter",
        lambda **kwargs: sentinel,
    )

    reporter = progress.create_cli_progress_reporter(output_dir="out")

    assert reporter is sentinel


def test_create_cli_progress_reporter_falls_back_when_stdout_is_not_terminal(
    monkeypatch,
):
    import hs2p.progress as progress

    _install_fake_rich_console(monkeypatch, is_terminal=False)

    reporter = progress.create_cli_progress_reporter(output_dir="out")

    assert isinstance(reporter, progress.TextReporter)


def test_rich_tiling_summary_uses_zero_tile_label_without_process_list(monkeypatch):
    import hs2p.progress as progress

    _install_fake_rich_console(monkeypatch, is_terminal=True)
    _install_fake_rich_progress(monkeypatch)
    _install_fake_rich_summary_types(monkeypatch)

    captured = {}

    class FakeConsole:
        def print(self, *args, **kwargs):
            captured["printed"] = args

        def log(self, *args, **kwargs):
            captured["logged"] = args

    reporter = progress.RichReporter(output_dir="out", console=FakeConsole())
    monkeypatch.setattr(
        reporter,
        "_print_summary",
        lambda title, rows: captured.update({"title": title, "rows": rows}),
    )

    reporter.emit(
        progress.ProgressEvent(
            kind="tiling.finished",
            payload={
                "total": 3,
                "completed": 2,
                "failed": 1,
                "zero_tile_successes": 4,
                "discovered_tiles": 7,
                "process_list_path": "ignored.csv",
            },
        )
    )

    assert captured["title"] == "Tiling Summary"
    assert captured["rows"] == [
        ("Slides", "3"),
        ("Completed", "2"),
        ("Failed", "1"),
        ("Zero-tile", "4"),
        ("Total tiles", "7"),
    ]


def test_rich_tissue_progress_uses_a_separate_task(monkeypatch):
    import hs2p.progress as progress

    _install_fake_rich_console(monkeypatch, is_terminal=True)
    _install_fake_rich_progress(monkeypatch)
    _install_fake_rich_summary_types(monkeypatch)

    class FakeConsole:
        def print(self, *args, **kwargs):
            return None

    reporter = progress.RichReporter(output_dir="out", console=FakeConsole())

    reporter.emit(
        progress.ProgressEvent(
            kind="tissue.started",
            payload={"total": 3},
        )
    )
    reporter.emit(
        progress.ProgressEvent(
            kind="tissue.progress",
            payload={
                "total": 3,
                "completed": 2,
                "failed": 1,
                "pending": 0,
            },
        )
    )

    assert reporter.progress.tasks[1]["description"] == "Tissue masks (2/3 resolved)"
    assert reporter.progress.tasks[1]["completed"] == 3


def test_tiling_main_installs_progress_reporter_only_during_pipeline_run(
    monkeypatch, tmp_path: Path
):
    import hs2p.progress as progress

    reporter = RecordingReporter()
    cfg = _base_cli_cfg(tmp_path)
    observed = {}

    monkeypatch.setattr(tiling_mod, "setup", lambda args: cfg)
    monkeypatch.setattr(
        tiling_mod,
        "load_csv",
        lambda cfg, **kwargs: [
            SlideSpec(sample_id="slide-1", image_path=Path("slide-1.svs"))
        ],
    )
    monkeypatch.setattr(tiling_mod, "resolve_tiling_config", lambda cfg: _tiling_config())
    monkeypatch.setattr(
        tiling_mod, "resolve_segmentation_config", lambda cfg: _segmentation_config()
    )
    monkeypatch.setattr(tiling_mod, "resolve_filter_config", lambda cfg: _filter_config())
    monkeypatch.setattr(tiling_mod, "resolve_preview_config", lambda cfg: None)
    monkeypatch.setattr(tiling_mod, "resolve_read_coordinates_from", lambda cfg: None)
    monkeypatch.setattr(progress, "create_cli_progress_reporter", lambda **kwargs: reporter)

    def _fake_tile_slides(*args, **kwargs):
        observed["reporter"] = progress.get_progress_reporter()
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "sample_id": "slide-1",
                    "image_path": "slide-1.svs",
                    "mask_path": np.nan,
                    "requested_backend": "asap",
                    "backend": "asap",
                    "tiling_status": "success",
                    "num_tiles": 2,
                    "coordinates_npz_path": str(output_dir / "tiles" / "slide-1.coordinates.npz"),
                    "coordinates_meta_path": str(
                        output_dir / "tiles" / "slide-1.coordinates.meta.json"
                    ),
                    "error": np.nan,
                    "traceback": np.nan,
                }
            ]
        ).to_csv(output_dir / "process_list.csv", index=False)
        return [
            TilingArtifacts(
                sample_id="slide-1",
                coordinates_npz_path=output_dir / "tiles" / "slide-1.coordinates.npz",
                coordinates_meta_path=output_dir / "tiles" / "slide-1.coordinates.meta.json",
                num_tiles=2,
            )
        ]

    monkeypatch.setattr(tiling_mod, "tile_slides", _fake_tile_slides)

    tiling_mod.main(SimpleNamespace())

    assert observed["reporter"] is reporter
    assert isinstance(progress.get_progress_reporter(), progress.NullProgressReporter)


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
            return SimpleNamespace(
                input_index=request.input_index,
                whole_slide=request.whole_slide,
                ok=True,
                artifact=TilingArtifacts(
                    sample_id="slide-b",
                    coordinates_npz_path=run_dir / "tiles" / "slide-b.coordinates.npz",
                    coordinates_meta_path=run_dir / "tiles" / "slide-b.coordinates.meta.json",
                    num_tiles=1,
                ),
                mask_preview_path=None,
                error=None,
                traceback_text=None,
            )
        return SimpleNamespace(
            input_index=request.input_index,
            whole_slide=request.whole_slide,
            ok=False,
            artifact=None,
            mask_preview_path=None,
            requested_backend=None,
            backend=None,
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
    }

def test_progress_aware_logging_routes_stdout_through_active_reporter():
    import hs2p.progress as progress
    import hs2p.utils.logging as log_utils

    reporter = RecordingReporter()
    logger_name = "hs2p.test.progress.logging"
    log_utils._configure_logger.cache_clear()

    with progress.activate_progress_reporter(reporter):
        log_utils.setup_logging(name=logger_name, level=logging.INFO, capture_warnings=False)
        logging.getLogger(logger_name).info("progress aware message")

    assert any("progress aware message" in line for line in reporter.log_lines)


def test_text_reporter_formats_backend_selection_line(capsys):
    import hs2p.progress as progress

    reporter = progress.TextReporter()

    reporter.emit(
        progress.ProgressEvent(
            kind="backend.selected",
            payload={
                "sample_id": "slide-1",
                "backend": "cucim",
                "reason": "selected cuCIM for auto backend",
            },
        )
    )

    out = capsys.readouterr().out.strip()
    assert out == "[backend] slide-1: selected cuCIM for auto backend"


def test_text_reporter_formats_preview_progress_line(capsys):
    import hs2p.progress as progress

    reporter = progress.TextReporter()

    reporter.emit(progress.ProgressEvent(kind="preview.started", payload={"total": 3}))
    reporter.emit(
        progress.ProgressEvent(
            kind="preview.progress",
            payload={"total": 3, "completed": 2, "failed": 1, "pending": 0},
        )
    )
    reporter.emit(
        progress.ProgressEvent(
            kind="preview.finished",
            payload={"total": 3, "completed": 3, "failed": 0, "pending": 0},
        )
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "Generating previews (3 total)...",
        "Preview generation: 2/3 complete, 1 failed",
        "Preview generation finished: 3/3 complete, 0 failed",
    ]


def test_rich_reporter_emits_backend_selection_line(monkeypatch):
    import hs2p.progress as progress

    _install_fake_rich_console(monkeypatch, is_terminal=True)
    _install_fake_rich_progress(monkeypatch)
    _install_fake_rich_summary_types(monkeypatch)

    class FakeConsole:
        def __init__(self):
            self.lines = []

        def print(self, message, **kwargs):
            self.lines.append((message, kwargs))

        def log(self, message, **kwargs):
            self.lines.append((message, kwargs))

    reporter = progress.RichReporter(output_dir="out", console=FakeConsole())
    reporter.emit(
        progress.ProgressEvent(
            kind="backend.selected",
            payload={
                "sample_id": "slide-1",
                "backend": "cucim",
                "reason": "selected cuCIM for auto backend",
            },
        )
    )

    assert reporter.console.lines[-1][0] == "[backend] slide-1: selected cuCIM for auto backend"


def test_rich_reporter_emits_preview_progress_lines(monkeypatch):
    import hs2p.progress as progress

    _install_fake_rich_console(monkeypatch, is_terminal=True)
    _install_fake_rich_progress(monkeypatch)
    _install_fake_rich_summary_types(monkeypatch)

    class FakeConsole:
        def __init__(self):
            self.lines = []

        def print(self, message, **kwargs):
            self.lines.append((message, kwargs))

        def log(self, message, **kwargs):
            self.lines.append((message, kwargs))

    reporter = progress.RichReporter(output_dir="out", console=FakeConsole())

    reporter.emit(progress.ProgressEvent(kind="preview.started", payload={"total": 3}))
    assert reporter.progress.tasks[1]["description"] == "Generating previews"
    reporter.emit(
        progress.ProgressEvent(
            kind="preview.progress",
            payload={"total": 3, "completed": 2, "failed": 1, "pending": 0},
        )
    )
    assert (
        reporter.progress.tasks[1]["description"]
        == "Generating previews (2/3 rendered)"
    )
    reporter.emit(
        progress.ProgressEvent(
            kind="preview.finished",
            payload={"total": 3, "completed": 3, "failed": 0, "pending": 0},
        )
    )
    assert 1 not in reporter.progress.tasks
    assert reporter.console.lines[-1][0].title == "Preview Generation"
