
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ProgressCallback = Callable[[int, int], None]


@dataclass
class ExtractionPhaseRecorder:
    progress_callback: ProgressCallback | None = None
    read_s: float = 0.0
    encode_s: float = 0.0
    write_s: float = 0.0
    tile_count: int = 0
    jpeg_bytes: int = 0

    def record(
        self,
        phase: str,
        duration_s: float,
        *,
        tile_count: int = 0,
        jpeg_bytes: int = 0,
    ) -> None:
        if phase == "read":
            self.read_s += float(duration_s)
            self.tile_count += int(tile_count)
            if self.progress_callback is not None:
                for _ in range(int(tile_count)):
                    self.progress_callback(1, 1)
            return
        if phase == "encode":
            self.encode_s += float(duration_s)
            self.jpeg_bytes += int(jpeg_bytes)
            return
        if phase == "write":
            self.write_s += float(duration_s)
            return
        raise ValueError(f"Unsupported phase: {phase}")


def resolve_jpeg_backend(
    *,
    config_file: Path,
    cli_jpeg_backend: str | None = None,
) -> str:
    from hs2p.utils.setup import get_cfg_from_file

    if cli_jpeg_backend is not None:
        return str(cli_jpeg_backend)

    cfg = get_cfg_from_file(config_file)
    return str(getattr(cfg.speed, "jpeg_backend", "turbojpeg"))


def build_result_row(
    *,
    sample_id: str,
    image_path: str,
    repeat_index: int,
    tiles: int,
    jpeg_quality: int,
    jpeg_backend: str,
    num_workers: int,
    read_s: float,
    encode_s: float,
    write_s: float,
    total_s: float,
    jpeg_bytes: int,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "repeat_index": repeat_index,
        "tiles": tiles,
        "jpeg_quality": jpeg_quality,
        "jpeg_backend": jpeg_backend,
        "num_workers": num_workers,
        "read_s": round(read_s, 6),
        "encode_s": round(encode_s, 6),
        "write_s": round(write_s, 6),
        "total_s": round(total_s, 6),
        "read_pct": round(100 * read_s / total_s, 2) if total_s > 0 else 0.0,
        "encode_pct": round(100 * encode_s / total_s, 2) if total_s > 0 else 0.0,
        "write_pct": round(100 * write_s / total_s, 2) if total_s > 0 else 0.0,
        "tiles_per_second": round(tiles / total_s, 2) if total_s > 0 else 0.0,
        "jpeg_bytes": jpeg_bytes,
        "jpeg_mb_per_second": round(
            (jpeg_bytes / 1_000_000) / total_s, 2
        ) if total_s > 0 else 0.0,
    }


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    def _mean(vals):
        return round(statistics.mean(vals), 6)

    def _pstdev(vals):
        return round(statistics.pstdev(vals), 6) if len(vals) > 1 else 0.0

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["num_workers"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for nw, nw_rows in grouped.items():
        read_s = [float(r["read_s"]) for r in nw_rows]
        encode_s = [float(r["encode_s"]) for r in nw_rows]
        write_s = [float(r["write_s"]) for r in nw_rows]
        total_s = [float(r["total_s"]) for r in nw_rows]
        read_pct = [float(r["read_pct"]) for r in nw_rows]
        encode_pct = [float(r["encode_pct"]) for r in nw_rows]
        write_pct = [float(r["write_pct"]) for r in nw_rows]
        tps = [float(r["tiles_per_second"]) for r in nw_rows]
        summary.append(
            {
                "num_workers": nw,
                "tiles": int(nw_rows[0]["tiles"]),
                "jpeg_quality": int(nw_rows[0]["jpeg_quality"]),
                "jpeg_backend": str(nw_rows[0]["jpeg_backend"]),
                "mean_read_s": _mean(read_s),
                "mean_encode_s": _mean(encode_s),
                "mean_write_s": _mean(write_s),
                "mean_total_s": _mean(total_s),
                "mean_read_pct": round(statistics.mean(read_pct), 2),
                "mean_encode_pct": round(statistics.mean(encode_pct), 2),
                "mean_write_pct": round(statistics.mean(write_pct), 2),
                "mean_tiles_per_second": round(statistics.mean(tps), 2),
                "std_tiles_per_second": round(_pstdev(tps), 2),
            }
        )
    return summary
