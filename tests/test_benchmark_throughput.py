import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


def _load_benchmark_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_throughput.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_throughput", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_balanced_sample_returns_requested_slide_count():
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

    sampled = benchmark_mod.build_balanced_sample(slides, n_slides=4, seed=7)

    assert len(sampled) == 4
    assert {slide["sample_id"] for slide in sampled}.issubset(
        {f"slide-{idx}" for idx in range(1, 7)}
    )


def test_benchmark_records_worker_sweep_without_workload_label(
    monkeypatch, tmp_path: Path
):
    benchmark_mod = _load_benchmark_module()
    seen_workers = []

    def _fake_run_one(**kwargs):
        seen_workers.append(kwargs["num_workers"])
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
        slides=[{"sample_id": "a"}],
        workers=[2, 4],
        repeat=1,
        python_executable="python",
        backend="asap",
        target_spacing=0.5,
        tile_size=256,
        output_root=tmp_path,
    )

    assert seen_workers == [2, 4]
    assert [record["num_workers"] for record in records] == [2, 4]


def test_write_config_uses_current_preview_and_sampling_keys(tmp_path: Path):
    benchmark_mod = _load_benchmark_module()
    config_path = tmp_path / "config.yaml"

    benchmark_mod.write_config(
        csv_path=tmp_path / "slides.csv",
        output_dir=tmp_path / "output",
        num_workers=4,
        backend="asap",
        target_spacing=0.5,
        tile_size=256,
        config_path=config_path,
    )

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    assert cfg["save_previews"] is False
    assert "preview" in cfg["tiling"]
    assert "visu_params" not in cfg["tiling"]
    assert cfg["tiling"]["sampling_params"]["independent_sampling"] is False
    assert "independant_sampling" not in cfg["tiling"]["sampling_params"]
