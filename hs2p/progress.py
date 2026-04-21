
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import sys
import time
from typing import Any


@dataclass(frozen=True)
class ProgressEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class NullProgressReporter:
    def emit(self, event: ProgressEvent) -> None:
        return None

    def close(self) -> None:
        return None

    def write_log(self, message: str, *, stream=None) -> None:
        target = stream or sys.stdout
        print(message, file=target, flush=True)


class TextReporter:
    def __init__(self, *, stream=None) -> None:
        self.stream = stream or sys.stdout
        self._last_line_by_kind: dict[str, tuple[float, str]] = {}

    def emit(self, event: ProgressEvent) -> None:
        line = self._format_line(event.kind, event.payload)
        if line is None:
            return
        if event.kind in {"tiling.progress", "preview.progress", "sampling.progress"}:
            now = time.monotonic()
            last = self._last_line_by_kind.get(event.kind)
            if last is not None and last[1] == line and (now - last[0]) < 1.0:
                return
            self._last_line_by_kind[event.kind] = (now, line)
        print(line, file=self.stream, flush=True)

    def close(self) -> None:
        return None

    def write_log(self, message: str, *, stream=None) -> None:
        print(message, file=stream or self.stream, flush=True)

    def _format_line(self, kind: str, payload: dict[str, Any]) -> str | None:
        if kind == "run.started":
            return (
                f"Starting hs2p {payload['command']}: {payload['slide_count']} slide(s), "
                f"backend={payload['backend']} spacing={payload['requested_spacing_um']}um "
                f"tile_size={payload['requested_tile_size_px']}px workers={payload['num_workers']} "
                f"output={payload['output_dir']}"
            )
        if kind == "tissue.started":
            return f"Resolving tissue masks ({payload['total']} total)..."
        if kind == "tissue.progress":
            return (
                f"Tissue resolution: {payload['completed']}/{payload['total']} complete, "
                f"{payload['failed']} failed"
            )
        if kind == "tissue.finished":
            return (
                f"Tissue resolution finished: {payload['completed']}/{payload['total']} complete, "
                f"{payload['failed']} failed"
            )
        if kind == "tiling.started":
            return f"Tiling slides ({payload['total']} total)..."
        if kind == "tiling.progress":
            return (
                f"Tiling progress: {payload['completed']}/{payload['total']} complete, "
                f"{payload['failed']} failed, {payload['discovered_tiles']} tiles discovered"
            )
        if kind == "tiling.finished":
            return (
                f"Tiling finished: {payload['completed']}/{payload['total']} complete, "
                f"{payload['failed']} failed, {payload['discovered_tiles']} tiles"
            )
        if kind == "preview.started":
            return f"Generating previews ({payload['total']} total)..."
        if kind == "preview.progress":
            return (
                f"Preview generation: {payload['completed']}/{payload['total']} complete, "
                f"{payload['failed']} failed"
            )
        if kind == "preview.finished":
            return (
                f"Preview generation finished: {payload['completed']}/{payload['total']} complete, "
                f"{payload['failed']} failed"
            )
        if kind == "sampling.started":
            return f"Sampling slides ({payload['total']} total)..."
        if kind == "sampling.progress":
            return (
                f"Sampling progress: {payload['completed']}/{payload['total']} complete, "
                f"{payload['failed']} failed, {payload['sampled_tiles']} tiles kept"
            )
        if kind == "sampling.finished":
            return (
                f"Sampling finished: {payload['completed']}/{payload['total']} complete, "
                f"{payload['failed']} failed, {payload['sampled_tiles']} tiles kept"
            )
        if kind == "backend.selected":
            reason = payload.get("reason")
            if reason:
                return f"[backend] {payload['sample_id']}: {reason}"
            return f"[backend] {payload['sample_id']}: using {payload['backend']}"
        if kind == "run.finished":
            return f"Run finished successfully. Output: {payload['output_dir']}"
        if kind == "run.failed":
            return f"Run failed during {payload['stage']}: {payload['error']}"
        return None


class RichReporter:
    def __init__(self, *, output_dir=None, console=None) -> None:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self.output_dir = output_dir
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self.progress.start()
        self._task_ids: dict[str, int] = {}

    def emit(self, event: ProgressEvent) -> None:
        kind = event.kind
        payload = event.payload
        if kind == "run.started":
            self.console.print(
                f"[bold]hs2p[/bold] {payload['command']} on {payload['slide_count']} slide(s) "
                f"backend={payload['backend']} spacing={payload['requested_spacing_um']}um "
                f"tile_size={payload['requested_tile_size_px']}px workers={payload['num_workers']}"
            )
            self.console.print(f"Output: {payload['output_dir']}")
            return
        if kind == "tissue.started":
            self._task_ids["tissue"] = self.progress.add_task(
                "Tissue masks", total=payload["total"]
            )
            return
        if kind == "tissue.progress":
            task_id = self._task_ids.get("tissue")
            if task_id is not None:
                self.progress.update(
                    task_id,
                    completed=payload["completed"] + payload["failed"],
                    description=(
                        f"Tissue masks ({payload['completed']}/{payload['total']} resolved)"
                    ),
                )
            return
        if kind == "tissue.finished":
            task_id = self._task_ids.get("tissue")
            if task_id is not None:
                self.progress.update(
                    task_id, completed=payload["completed"] + payload["failed"]
                )
            return
        if kind == "tiling.started":
            self._task_ids["tiling"] = self.progress.add_task(
                "Tiling slides", total=payload["total"]
            )
            return
        if kind == "tiling.progress":
            task_id = self._task_ids.get("tiling")
            if task_id is not None:
                self.progress.update(
                    task_id,
                    completed=payload["completed"] + payload["failed"],
                    description=f"Tiling slides ({payload['completed']}/{payload['total']} resolved)",
                )
            return
        if kind == "tiling.finished":
            task_id = self._task_ids.get("tiling")
            if task_id is not None:
                self.progress.update(
                    task_id, completed=payload["completed"] + payload["failed"]
                )
            self._print_summary(
                "Tiling Summary",
                [
                    ("Slides", str(payload["total"])),
                    ("Completed", str(payload["completed"])),
                    ("Failed", str(payload["failed"])),
                    ("Zero-tile", str(payload["zero_tile_successes"])),
                    ("Total tiles", str(payload["discovered_tiles"])),
                ],
            )
            return
        if kind == "preview.started":
            self._task_ids["preview"] = self.progress.add_task(
                "Generating previews", total=payload["total"]
            )
            return
        if kind == "preview.progress":
            task_id = self._task_ids.get("preview")
            if task_id is not None:
                self.progress.update(
                    task_id,
                    completed=payload["completed"] + payload["failed"],
                    description=(
                        f"Generating previews ({payload['completed']}/{payload['total']} rendered)"
                    ),
                )
            return
        if kind == "preview.finished":
            task_id = self._task_ids.pop("preview", None)
            if task_id is not None:
                self.progress.update(
                    task_id, completed=payload["completed"] + payload["failed"]
                )
                if hasattr(self.progress, "remove_task"):
                    self.progress.remove_task(task_id)
            self._print_summary(
                "Preview Generation",
                [
                    ("Slides", str(payload["total"])),
                    ("Completed", str(payload["completed"])),
                    ("Failed", str(payload["failed"])),
                ],
            )
            return
        if kind == "sampling.started":
            self._task_ids["sampling"] = self.progress.add_task(
                "Sampling slides", total=payload["total"]
            )
            return
        if kind == "sampling.progress":
            task_id = self._task_ids.get("sampling")
            if task_id is not None:
                self.progress.update(
                    task_id,
                    completed=payload["completed"] + payload["failed"],
                    description=f"Sampling slides ({payload['sampled_tiles']} tiles kept)",
                )
            return
        if kind == "sampling.finished":
            zero_tile_total = sum(payload["zero_tile_successes_by_annotation"].values())
            rows = [
                ("Slides", str(payload["total"])),
                ("Completed", str(payload["completed"])),
                ("Failed", str(payload["failed"])),
                ("Zero-tile", str(zero_tile_total)),
                ("Total tiles", str(payload["sampled_tiles"])),
                ("Process list", payload["process_list_path"]),
            ]
            for annotation, count in sorted(
                payload["zero_tile_successes_by_annotation"].items()
            ):
                rows.append((f"Zero-tile {annotation}", str(count)))
            self._print_summary("Sampling Summary", rows)
            return
        if kind == "backend.selected":
            sample_id = payload["sample_id"]
            reason = payload.get("reason")
            if reason:
                self.console.print(f"[backend] {sample_id}: {reason}")
            else:
                self.console.print(f"[backend] {sample_id}: using {payload['backend']}")
            return
        if kind == "run.finished":
            self._print_summary(
                "Run Complete",
                [
                    ("Output", payload["output_dir"]),
                    ("Process list", payload["process_list_path"]),
                    ("Logs", payload["logs_dir"]),
                ],
            )
            return
        if kind == "run.failed":
            self._print_summary(
                "Run Failed",
                [
                    ("Stage", payload["stage"]),
                    ("Error", payload["error"]),
                ],
            )
            return

    def close(self) -> None:
        self.progress.stop()

    def write_log(self, message: str, *, stream=None) -> None:
        self.console.print(message, markup=False, highlight=False, soft_wrap=True)

    def _print_summary(self, title: str, rows: list[tuple[str, str]]) -> None:
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column()
        for key, value in rows:
            table.add_row(key, value)
        self.console.print(Panel.fit(table, title=title, border_style="blue"))


_NULL_REPORTER = NullProgressReporter()
_ACTIVE_REPORTER: ContextVar[Any] = ContextVar(
    "hs2p_active_progress_reporter", default=_NULL_REPORTER
)


def create_cli_progress_reporter(*, output_dir=None, stream=None):
    try:
        from rich.console import Console
    except ImportError:
        return TextReporter(stream=stream)
    console = Console(file=stream or sys.stdout)
    if not console.is_terminal:
        return TextReporter(stream=stream or sys.stdout)
    return RichReporter(output_dir=output_dir, console=console)


def get_progress_reporter():
    return _ACTIVE_REPORTER.get()


@contextmanager
def activate_progress_reporter(reporter):
    token = _ACTIVE_REPORTER.set(reporter)
    try:
        yield reporter
    finally:
        try:
            reporter.close()
        finally:
            _ACTIVE_REPORTER.reset(token)


def emit_progress(kind: str, **payload: Any) -> None:
    get_progress_reporter().emit(ProgressEvent(kind=kind, payload=payload))


def emit_progress_log(message: str, *, stream=None) -> None:
    reporter = get_progress_reporter()
    if hasattr(reporter, "write_log"):
        reporter.write_log(message, stream=stream)
        return
    target = stream or sys.stdout
    print(message, file=target, flush=True)
