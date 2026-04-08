
import os
import sys
import tempfile
from typing import Callable, TypeVar

T = TypeVar("T")

CUCIM_NO_GPU_STDERR_PATTERNS = (
    "cuInit Failed, error",
    "CUDA_ERROR_NO_DEVICE",
    "CUDA_ERROR_OPERATING_SYSTEM",
    "CUDA_ERROR_NOT_INITIALIZED",
    "cuFile initialization failed",
)


def _filter_stderr_text(
    text: str,
    suppress_patterns: tuple[str, ...],
) -> str:
    if not text:
        return text
    filtered_text = text
    for pattern in suppress_patterns:
        filtered_text = filtered_text.replace(pattern, "")
    kept_lines: list[str] = []
    for line in filtered_text.splitlines(keepends=True):
        if any(pattern in line for pattern in suppress_patterns):
            continue
        if not line.strip():
            continue
        kept_lines.append(line)
    return "".join(kept_lines)


def run_with_filtered_stderr(
    func: Callable[[], T],
    *,
    suppress_patterns: tuple[str, ...] = CUCIM_NO_GPU_STDERR_PATTERNS,
) -> T:
    """Run ``func`` while capturing fd 2 and dropping known-noisy lines."""

    original_stderr_fd = os.dup(2)
    captured_stderr_text = ""
    captured_exc: tuple[type[BaseException], BaseException, object] | None = None
    try:
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            os.dup2(tmp.fileno(), 2)
            try:
                result = func()
            except BaseException:
                captured_exc = sys.exc_info()
                result = None
            finally:
                os.dup2(original_stderr_fd, 2)
                tmp.flush()
                tmp.seek(0)
                captured_stderr_text = tmp.read().decode("utf-8", errors="replace")
        filtered = _filter_stderr_text(captured_stderr_text, suppress_patterns)
        if filtered:
            os.write(2, filtered.encode("utf-8", errors="replace"))
        if captured_exc is not None:
            exc_type, exc, tb = captured_exc
            raise exc.with_traceback(tb)
        return result  # type: ignore[return-value]
    finally:
        os.close(original_stderr_fd)
