
import os
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


def _filter_text(
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


def _run_with_filtered_fds(
    func: Callable[[], T],
    fds: tuple[int, ...],
    suppress_patterns: tuple[str, ...],
) -> T:
    originals = [os.dup(fd) for fd in fds]
    captured_exc: BaseException | None = None
    try:
        tmp_files = [tempfile.TemporaryFile(mode="w+b") for _ in fds]
        try:
            for tmp, fd in zip(tmp_files, fds):
                os.dup2(tmp.fileno(), fd)
            try:
                result = func()
            except BaseException as exc:
                captured_exc = exc
                result = None
            finally:
                for orig, fd in zip(originals, fds):
                    os.dup2(orig, fd)
                captured_texts = []
                for tmp in tmp_files:
                    tmp.seek(0)
                    captured_texts.append(tmp.read().decode("utf-8", errors="replace"))
        finally:
            for tmp in tmp_files:
                tmp.close()
        for text, fd in zip(captured_texts, fds):
            filtered = _filter_text(text, suppress_patterns)
            if filtered:
                os.write(fd, filtered.encode("utf-8", errors="replace"))
        if captured_exc is not None:
            raise captured_exc
        return result  # type: ignore[return-value]
    finally:
        for orig in originals:
            os.close(orig)


def run_with_filtered_stderr(
    func: Callable[[], T],
    *,
    suppress_patterns: tuple[str, ...] = CUCIM_NO_GPU_STDERR_PATTERNS,
) -> T:
    """Run ``func`` while capturing fd 2 and dropping known-noisy lines."""
    return _run_with_filtered_fds(func, (2,), suppress_patterns)


def run_with_filtered_stdio(
    func: Callable[[], T],
    *,
    suppress_patterns: tuple[str, ...] = CUCIM_NO_GPU_STDERR_PATTERNS,
) -> T:
    """Run ``func`` while filtering known-native noise from both fd 1 and fd 2."""
    return _run_with_filtered_fds(func, (1, 2), suppress_patterns)
