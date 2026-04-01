import importlib.util
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.script


def _load_benchmark_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_tile_store.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_tile_store", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_result_row_computes_percentages_and_throughput():
    mod = _load_benchmark_module()
    row = mod.build_result_row(
        sample_id="slide-1",
        image_path="/slides/slide-1.svs",
        repeat_index=0,
        tiles=100,
        jpeg_quality=90,
        jpeg_backend="turbojpeg",
        num_workers=4,
        read_s=1.0,
        encode_s=2.0,
        write_s=1.0,
        total_s=4.0,
        jpeg_bytes=500_000,
    )
    assert row["sample_id"] == "slide-1"
    assert row["image_path"] == "/slides/slide-1.svs"
    assert row["repeat_index"] == 0
    assert row["tiles"] == 100
    assert row["jpeg_quality"] == 90
    assert row["jpeg_backend"] == "turbojpeg"
    assert row["num_workers"] == 4
    assert row["read_s"] == 1.0
    assert row["encode_s"] == 2.0
    assert row["write_s"] == 1.0
    assert row["total_s"] == 4.0
    assert row["read_pct"] == 25.0
    assert row["encode_pct"] == 50.0
    assert row["write_pct"] == 25.0
    assert row["tiles_per_second"] == 25.0
    assert row["jpeg_bytes"] == 500_000
    assert row["jpeg_mb_per_second"] == 0.12


def test_summarize_results_computes_mean_and_std():
    mod = _load_benchmark_module()
    rows = [
        mod.build_result_row(
            sample_id="s",
            image_path="/s.svs",
            repeat_index=i,
            tiles=100,
            jpeg_quality=90,
            jpeg_backend="turbojpeg",
            num_workers=4,
            read_s=r,
            encode_s=e,
            write_s=w,
            total_s=r + e + w,
            jpeg_bytes=500_000,
        )
        for i, (r, e, w) in enumerate([(1.0, 2.0, 1.0), (1.2, 2.2, 0.8), (0.8, 1.8, 1.2)])
    ]
    summary = mod.summarize_results(rows)
    assert len(summary) == 1
    s = summary[0]
    assert s["tiles"] == 100
    assert s["jpeg_quality"] == 90
    assert s["jpeg_backend"] == "turbojpeg"
    assert s["num_workers"] == 4

    expected_totals = [4.0, 4.2, 3.8]
    expected_tps = [100 / t for t in expected_totals]
    assert s["mean_total_s"] == pytest.approx(statistics.mean(expected_totals), abs=1e-4)
    assert s["mean_read_s"] == pytest.approx(statistics.mean([1.0, 1.2, 0.8]), abs=1e-4)
    assert s["mean_encode_s"] == pytest.approx(statistics.mean([2.0, 2.2, 1.8]), abs=1e-4)
    assert s["mean_write_s"] == pytest.approx(statistics.mean([1.0, 0.8, 1.2]), abs=1e-4)
    assert s["mean_tiles_per_second"] == pytest.approx(statistics.mean(expected_tps), abs=0.1)
    assert s["std_tiles_per_second"] == pytest.approx(statistics.pstdev(expected_tps), abs=0.1)


def test_benchmark_tile_store_accumulates_per_phase_times(tmp_path, monkeypatch):
    mod = _load_benchmark_module()

    result_stub = MagicMock()
    result_stub.effective_tile_size_px = 64
    result_stub.requested_tile_size_px = 64
    result_stub.x = np.empty(0, dtype=np.int64)
    result_stub.y = np.empty(0, dtype=np.int64)
    result_stub.sample_id = "slide-1"

    def _fake_extract_tiles_to_tar(*args, **kwargs):
        recorder = kwargs["phase_recorder"]
        recorder.record("read", 0.1, tile_count=1)
        recorder.record("encode", 0.2, tile_count=1, jpeg_bytes=100)
        recorder.record("write", 0.05, tile_count=1)
        recorder.record("read", 0.15, tile_count=1)
        recorder.record("encode", 0.25, tile_count=1, jpeg_bytes=150)
        recorder.record("write", 0.07, tile_count=1)
        recorder.record("read", 0.12, tile_count=1)
        recorder.record("encode", 0.22, tile_count=1, jpeg_bytes=125)
        recorder.record("write", 0.06, tile_count=1)
        return tmp_path / "tiles" / "slide-1.tiles.tar", args[0]

    monkeypatch.setattr(mod, "extract_tiles_to_tar", _fake_extract_tiles_to_tar)

    metrics = mod.benchmark_tile_store(
        result=result_stub,
        jpeg_quality=90,
        num_workers=4,
        output_dir=tmp_path,
    )

    assert metrics["tile_count"] == 3
    assert metrics["jpeg_bytes"] > 0
    assert metrics["read_s"] >= 0
    assert metrics["encode_s"] > 0
    assert metrics["write_s"] > 0
    assert metrics["total_s"] >= metrics["read_s"] + metrics["encode_s"] + metrics["write_s"]


def test_benchmark_tile_store_forwards_supertile_sizes(tmp_path, monkeypatch):
    mod = _load_benchmark_module()
    seen: dict[str, tuple[int, ...] | None] = {}

    def _fake_extract_tiles_to_tar(*args, **kwargs):
        seen["supertile_sizes"] = kwargs.get("supertile_sizes")
        seen["num_workers"] = kwargs.get("num_workers")
        seen["jpeg_backend"] = kwargs.get("jpeg_backend")
        seen["jpeg_quality"] = kwargs.get("jpeg_quality")
        seen["phase_recorder"] = kwargs.get("phase_recorder") is not None
        return tmp_path / "tiles" / "slide-1.tiles.tar", args[0]

    monkeypatch.setattr(mod, "extract_tiles_to_tar", _fake_extract_tiles_to_tar)

    result_stub = MagicMock()
    result_stub.effective_tile_size_px = 64
    result_stub.requested_tile_size_px = 64
    result_stub.x = np.empty(0, dtype=np.int64)
    result_stub.y = np.empty(0, dtype=np.int64)
    result_stub.sample_id = "slide-1"

    mod.benchmark_tile_store(
        result=result_stub,
        jpeg_quality=90,
        num_workers=4,
        output_dir=tmp_path,
        supertile_sizes=(16, 8, 4, 2),
    )

    assert seen["supertile_sizes"] == (16, 8, 4, 2)
    assert seen["num_workers"] == 4
    assert seen["jpeg_backend"] == "turbojpeg"
    assert seen["jpeg_quality"] == 90
    assert seen["phase_recorder"] is True


def test_benchmark_tile_store_handles_empty_iterator(tmp_path, monkeypatch):
    mod = _load_benchmark_module()

    monkeypatch.setattr(
        mod,
        "extract_tiles_to_tar",
        lambda *args, **kwargs: (tmp_path / "tiles" / "slide-1.tiles.tar", args[0]),
    )

    result_stub = MagicMock()
    result_stub.effective_tile_size_px = 64
    result_stub.requested_tile_size_px = 64
    result_stub.x = np.empty(0, dtype=np.int64)
    result_stub.y = np.empty(0, dtype=np.int64)
    result_stub.sample_id = "slide-1"

    metrics = mod.benchmark_tile_store(
        result=result_stub,
        jpeg_quality=90,
        num_workers=4,
        output_dir=tmp_path,
    )

    assert metrics["tile_count"] == 0
    assert metrics["jpeg_bytes"] == 0
    assert metrics["read_s"] == 0.0
    assert metrics["encode_s"] == 0.0
    assert metrics["write_s"] == 0.0
    assert metrics["total_s"] >= 0.0


def test_progress_callback_called_per_tile(tmp_path, monkeypatch):
    mod = _load_benchmark_module()

    result_stub = MagicMock()
    result_stub.effective_tile_size_px = 64
    result_stub.requested_tile_size_px = 64
    result_stub.x = np.empty(0, dtype=np.int64)
    result_stub.y = np.empty(0, dtype=np.int64)
    result_stub.sample_id = "slide-1"

    def _fake_extract_tiles_to_tar(*args, **kwargs):
        recorder = kwargs["phase_recorder"]
        for _ in range(5):
            recorder.record("read", 0.01, tile_count=1)
        return tmp_path / "tiles" / "slide-1.tiles.tar", args[0]

    monkeypatch.setattr(mod, "extract_tiles_to_tar", _fake_extract_tiles_to_tar)

    callback = MagicMock()
    mod.benchmark_tile_store(
        result=result_stub,
        jpeg_quality=90,
        num_workers=4,
        output_dir=tmp_path,
        progress_callback=callback,
    )

    assert callback.call_count == 5
    for call in callback.call_args_list:
        assert call == ((1, 1),)


def test_resolve_jpeg_backend_prefers_config(tmp_path):
    mod = _load_benchmark_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "csv: slides.csv\n"
        "output_dir: out\n"
        "speed:\n"
        "  jpeg_backend: pil\n"
    )

    assert mod.resolve_jpeg_backend(config_file=config_path) == "pil"


def test_resolve_jpeg_backend_defaults_to_turbojpeg_when_unspecified(tmp_path):
    mod = _load_benchmark_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "csv: slides.csv\n"
        "output_dir: out\n"
        "speed:\n"
        "  num_workers: 4\n"
    )

    assert mod.resolve_jpeg_backend(config_file=config_path) == "turbojpeg"


def test_resolve_jpeg_backend_prefers_cli_override(tmp_path):
    mod = _load_benchmark_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "csv: slides.csv\n"
        "output_dir: out\n"
        "speed:\n"
        "  jpeg_backend: pil\n"
    )

    assert (
        mod.resolve_jpeg_backend(config_file=config_path, cli_jpeg_backend="turbojpeg")
        == "turbojpeg"
    )
