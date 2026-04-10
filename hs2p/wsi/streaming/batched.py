
import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class BatchedReadRequest:
    location: tuple[int, int]
    size: tuple[int, int]
    payload: Any = None


def group_batched_read_requests_by_size(
    requests: Sequence[BatchedReadRequest],
) -> dict[tuple[int, int], list[BatchedReadRequest]]:
    grouped: dict[tuple[int, int], list[BatchedReadRequest]] = {}
    for request in requests:
        size = (int(request.size[0]), int(request.size[1]))
        grouped.setdefault(size, []).append(request)
    return grouped


@contextlib.contextmanager
def _suppress_native_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


def iter_cucim_batched_read_regions(
    *,
    image_path: Path,
    requests: Sequence[BatchedReadRequest],
    level: int,
    num_workers: int,
    spacing_override: float | None = None,
    gpu_decode: bool = False,
):
    from hs2p.wsi.backends.cucim import CuCIMReader

    with _suppress_native_stderr():
        reader = CuCIMReader(
            str(image_path),
            spacing_override=spacing_override,
            gpu_decode=gpu_decode,
        )
    grouped_requests = group_batched_read_requests_by_size(requests)

    def _iter_regions():
        for size, size_requests in grouped_requests.items():
            locations = [request.location for request in size_requests]
            with _suppress_native_stderr():
                regions = list(
                    reader.read_regions(
                        locations,
                        int(level),
                        size,
                        num_workers=int(num_workers),
                    )
                )
            for request, region in zip(size_requests, regions):
                yield request, region

    return _iter_regions()
