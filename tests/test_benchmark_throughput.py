import importlib.util
from pathlib import Path


def _load_benchmark_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_throughput.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_throughput", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_workloads_returns_balanced_and_skewed_sets():
    benchmark_mod = _load_benchmark_module()
    slides = [
        {
            "sample_id": f"slide-{idx}",
            "image_path": Path(f"slide-{idx}.svs"),
            "mask_path": None,
            "spacing_at_level_0": None,
            "size_bytes": size,
        }
        for idx, size in enumerate([10, 20, 30, 40, 500, 600], start=1)
    ]

    workloads = benchmark_mod.build_workloads(slides, n_slides=4, seed=7)

    assert [workload["name"] for workload in workloads] == ["balanced", "skewed"]
    assert len(workloads[0]["slides"]) == 4
    assert len(workloads[1]["slides"]) == 4
    skewed_ids = [slide["sample_id"] for slide in workloads[1]["slides"]]
    assert "slide-5" in skewed_ids
    assert "slide-6" in skewed_ids


def test_benchmark_records_workload_label(monkeypatch, tmp_path: Path):
    benchmark_mod = _load_benchmark_module()
    calls = []

    def _fake_run_one(**kwargs):
        calls.append(kwargs["workload"])
        return {
            "num_workers": kwargs["num_workers"],
            "elapsed_seconds": 2.0,
            "exit_code": 0,
            "num_slides": len(kwargs["slides"]),
            "total_tiles": 20,
            "slides_with_tiles": 1,
            "failed": 0,
            "tiles_per_second": 10.0,
        }

    monkeypatch.setattr(benchmark_mod, "run_one", _fake_run_one)

    records = benchmark_mod.benchmark(
        workloads=[
            {"name": "balanced", "slides": [{"sample_id": "a"}]},
            {"name": "skewed", "slides": [{"sample_id": "b"}]},
        ],
        workers=[2],
        repeat=1,
        python_executable="python",
        backend="asap",
        target_spacing=0.5,
        tile_size=256,
        output_root=tmp_path,
    )

    assert calls == ["balanced", "skewed"]
    assert [record["workload"] for record in records] == ["balanced", "skewed"]
