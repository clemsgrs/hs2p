from __future__ import annotations

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


def iter_cucim_batched_read_regions(
    *,
    image_path: Path,
    requests: Sequence[BatchedReadRequest],
    level: int,
    num_workers: int,
    gpu_decode: bool = False,
):
    from hs2p.wsi.backends.cucim import CuImageReader

    reader = CuImageReader(image_path, gpu_decode=gpu_decode)
    reader._ensure_open()
    grouped_requests = group_batched_read_requests_by_size(requests)

    def _iter_regions():
        for size, size_requests in grouped_requests.items():
            locations = [request.location for request in size_requests]
            regions = reader.read_region(
                locations,
                size,
                level=int(level),
                num_workers=int(num_workers),
            )
            for request, region in zip(size_requests, regions):
                yield request, region

    return _iter_regions()
