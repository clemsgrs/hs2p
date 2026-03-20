#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark hs2p tiling CLI wall-clock speed on the current checkout "
            "versus a legacy git ref such as main."
        )
    )
    parser.add_argument(
        "--slides-csv",
        required=True,
        help="CSV with wsi_path, mask_path, and optional sample_id columns.",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Base tiling config in either the current or legacy hs2p schema.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where benchmark inputs, logs, outputs, and summaries are written.",
    )
    parser.add_argument(
        "--legacy-ref",
        default="main",
        help="Git ref to benchmark as the legacy implementation (default: main).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to run each branch (default: 1).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for the current checkout (default: current interpreter).",
    )
    parser.add_argument(
        "--legacy-python",
        default=None,
        help="Optional Python executable for the legacy worktree (defaults to --python).",
    )
    parser.add_argument(
        "--disable-visualize",
        action="store_true",
        help="Force visualize=false in both generated configs to benchmark tiling without preview I/O.",
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Keep the temporary legacy worktree after the benchmark completes.",
    )
    return parser.parse_args()


def _string_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned == "":
        return None
    return cleaned


def _make_unique_sample_id(raw_id: str, seen: dict[str, int]) -> str:
    count = seen.get(raw_id, 0) + 1
    seen[raw_id] = count
    if count == 1:
        return raw_id
    return f"{raw_id}-{count}"


def load_benchmark_rows(csv_path: Path) -> list[dict[str, str | None]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV {csv_path} is empty.")
        fieldnames = set(reader.fieldnames)
        if "wsi_path" in fieldnames:
            slide_key = "wsi_path"
        elif "image_path" in fieldnames:
            slide_key = "image_path"
        else:
            raise ValueError(
                "Input CSV must contain either 'wsi_path' or 'image_path'."
            )

        seen_ids: dict[str, int] = {}
        rows: list[dict[str, str | None]] = []
        for index, row in enumerate(reader, start=1):
            raw_wsi_path = _string_or_none(row.get(slide_key))
            if raw_wsi_path is None:
                raise ValueError(f"Row {index} is missing {slide_key}.")
            sample_id = _string_or_none(row.get("sample_id"))
            if sample_id is None:
                sample_id = Path(raw_wsi_path).stem
            sample_id = _make_unique_sample_id(sample_id, seen_ids)
            rows.append(
                {
                    "sample_id": sample_id,
                    "wsi_path": raw_wsi_path,
                    "mask_path": _string_or_none(row.get("mask_path")),
                }
            )
    if not rows:
        raise ValueError(f"Input CSV {csv_path} does not contain any slides.")
    return rows


def branch_csv_rows(
    rows: list[dict[str, str | None]], *, schema: str
) -> list[dict[str, str | None]]:
    if schema == "current":
        return [
            {
                "sample_id": row["sample_id"],
                "image_path": row["wsi_path"],
                "mask_path": row["mask_path"],
            }
            for row in rows
        ]
    if schema == "legacy":
        return [
            {
                "wsi_path": row["wsi_path"],
                "mask_path": row["mask_path"],
            }
            for row in rows
        ]
    raise ValueError(f"Unsupported schema: {schema}")


def write_branch_csv(
    rows: list[dict[str, str | None]], *, schema: str, csv_path: Path
) -> Path:
    schema_rows = branch_csv_rows(rows, schema=schema)
    fieldnames = list(schema_rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(schema_rows)
    return csv_path


def build_branch_config(
    base_config: dict[str, Any],
    *,
    schema: str,
    csv_path: Path,
    output_dir: Path,
    visualize: bool | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg["csv"] = str(csv_path)
    cfg["output_dir"] = str(output_dir)
    cfg["resume"] = False
    if visualize is not None:
        cfg["visualize"] = visualize

    tiling = cfg.setdefault("tiling", {})
    params = tiling.setdefault("params", {})

    if schema == "current":
        replacements = {
            "tiling.read_coordinates_from": "tiling.read_tiles_from",
            "tiling.params.spacing": "tiling.params.target_spacing_um",
            "tiling.params.tile_size": "tiling.params.target_tile_size_px",
            "tiling.params.min_tissue_percentage": "tiling.params.tissue_threshold",
        }
        if "read_coordinates_from" in tiling:
            tiling["read_tiles_from"] = tiling.pop("read_coordinates_from")
        if "spacing" in params:
            params["target_spacing_um"] = params.pop("spacing")
        if "tile_size" in params:
            params["target_tile_size_px"] = params.pop("tile_size")
        if "min_tissue_percentage" in params:
            params["tissue_threshold"] = params.pop("min_tissue_percentage")
    elif schema == "legacy":
        replacements = {
            "tiling.read_tiles_from": "tiling.read_coordinates_from",
            "tiling.params.target_spacing_um": "tiling.params.spacing",
            "tiling.params.target_tile_size_px": "tiling.params.tile_size",
            "tiling.params.tissue_threshold": "tiling.params.min_tissue_percentage",
        }
        if "read_tiles_from" in tiling:
            tiling["read_coordinates_from"] = tiling.pop("read_tiles_from")
        if "target_spacing_um" in params:
            params["spacing"] = params.pop("target_spacing_um")
        if "target_tile_size_px" in params:
            params["tile_size"] = params.pop("target_tile_size_px")
        if "tissue_threshold" in params:
            params["min_tissue_percentage"] = params.pop("tissue_threshold")
    else:
        raise ValueError(f"Unsupported schema: {schema}")

    return _rewrite_config_strings(cfg, replacements)


def _rewrite_config_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_config_strings(nested_value, replacements)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_config_strings(item, replacements) for item in value]
    if isinstance(value, str):
        updated = value
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        return updated
    return value


def write_branch_config(config: dict[str, Any], config_path: Path) -> Path:
    OmegaConf.save(config=OmegaConf.create(config), f=config_path)
    return config_path


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> int:
    with log_path.open("w") as log_handle:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def parse_process_list(process_list_path: Path) -> dict[str, Any]:
    if not process_list_path.is_file():
        return {
            "num_slides": 0,
            "failed_slides": 0,
            "slides_with_tiles": None,
        }

    with process_list_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    failed_slides = sum(row.get("tiling_status") == "failed" for row in rows)
    slides_with_tiles: int | None = None
    if rows and "num_tiles" in rows[0]:
        slides_with_tiles = sum(
            int(float(row.get("num_tiles", "0") or 0)) > 0 for row in rows
        )
    return {
        "num_slides": len(rows),
        "failed_slides": failed_slides,
        "slides_with_tiles": slides_with_tiles,
    }


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=True,
    ).strip()


def resolve_git_ref(repo_root: Path, git_ref: str) -> str:
    return git_output(repo_root, "rev-parse", "--short", git_ref)


def create_legacy_worktree(
    *, repo_root: Path, legacy_ref: str, workspace_root: Path
) -> Path:
    worktree_path = Path(
        tempfile.mkdtemp(prefix="legacy-worktree-", dir=str(workspace_root))
    )
    subprocess.check_call(
        ["git", "worktree", "add", "--detach", str(worktree_path), legacy_ref],
        cwd=repo_root,
    )
    return worktree_path


def remove_legacy_worktree(*, repo_root: Path, worktree_path: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        cwd=repo_root,
        check=False,
    )
    shutil.rmtree(worktree_path, ignore_errors=True)


def benchmark_branch(
    *,
    branch_label: str,
    schema: str,
    repo_root: Path,
    python_executable: str,
    csv_rows: list[dict[str, str | None]],
    base_config: dict[str, Any],
    output_root: Path,
    repeat: int,
    visualize_override: bool | None,
    git_ref: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    branch_root = output_root / branch_label
    branch_root.mkdir(parents=True, exist_ok=True)

    for run_index in range(1, repeat + 1):
        run_root = branch_root / f"run-{run_index:02d}"
        inputs_dir = run_root / "inputs"
        cli_output_dir = run_root / "tiling-output"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        cli_output_dir.mkdir(parents=True, exist_ok=True)

        branch_csv_path = write_branch_csv(
            csv_rows,
            schema=schema,
            csv_path=inputs_dir / "slides.csv",
        )
        branch_config = build_branch_config(
            base_config,
            schema=schema,
            csv_path=branch_csv_path,
            output_dir=cli_output_dir,
            visualize=visualize_override,
        )
        branch_config_path = write_branch_config(
            branch_config,
            inputs_dir / "config.yaml",
        )

        command = [
            python_executable,
            "-m",
            "hs2p.tiling",
            "--config-file",
            str(branch_config_path),
            "--skip-datetime",
            "--skip-logging",
        ]
        print(
            f"[{branch_label}] run {run_index}/{repeat}: "
            + " ".join(command)
        )
        start = time.perf_counter()
        exit_code = run_command(
            command,
            cwd=repo_root,
            log_path=run_root / "tiling-cli.log",
        )
        elapsed_seconds = time.perf_counter() - start
        process_list_stats = parse_process_list(cli_output_dir / "process_list.csv")
        num_slides = process_list_stats["num_slides"]
        records.append(
            {
                "branch_label": branch_label,
                "schema": schema,
                "git_ref": git_ref,
                "run_index": run_index,
                "elapsed_seconds": round(elapsed_seconds, 6),
                "num_slides": num_slides,
                "failed_slides": process_list_stats["failed_slides"],
                "slides_with_tiles": process_list_stats["slides_with_tiles"],
                "slides_per_second": round(num_slides / elapsed_seconds, 6)
                if elapsed_seconds > 0
                else None,
                "exit_code": exit_code,
                "output_dir": str(cli_output_dir),
                "process_list_path": str(cli_output_dir / "process_list.csv"),
                "log_path": str(run_root / "tiling-cli.log"),
            }
        )
    return records


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_branch: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_branch.setdefault(record["branch_label"], []).append(record)

    summary_rows: list[dict[str, Any]] = []
    for branch_label, branch_records in by_branch.items():
        elapsed_values = [record["elapsed_seconds"] for record in branch_records]
        slide_rates = [
            record["slides_per_second"]
            for record in branch_records
            if record["slides_per_second"] is not None
        ]
        summary_rows.append(
            {
                "branch_label": branch_label,
                "schema": branch_records[0]["schema"],
                "git_ref": branch_records[0]["git_ref"],
                "repeats": len(branch_records),
                "mean_seconds": round(statistics.mean(elapsed_values), 6),
                "min_seconds": round(min(elapsed_values), 6),
                "max_seconds": round(max(elapsed_values), 6),
                "stdev_seconds": round(statistics.stdev(elapsed_values), 6)
                if len(elapsed_values) > 1
                else 0.0,
                "mean_slides_per_second": round(statistics.mean(slide_rates), 6)
                if slide_rates
                else None,
                "failed_runs": sum(record["exit_code"] != 0 for record in branch_records),
                "failed_slides_total": sum(
                    int(record["failed_slides"]) for record in branch_records
                ),
            }
        )
    return sorted(summary_rows, key=lambda row: row["branch_label"])


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> Path:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def legacy_output_stem(wsi_path: str) -> str:
    return Path(wsi_path).stem.replace(" ", "_")


def _load_current_artifact(
    *,
    sample_id: str,
    output_dir: Path,
) -> dict[str, Any] | None:
    npz_path = output_dir / "coordinates" / f"{sample_id}.tiles.npz"
    meta_path = output_dir / "coordinates" / f"{sample_id}.tiles.meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        return None
    with npz_path.open("rb") as handle:
        tiles = np.load(handle, allow_pickle=False)
        x = tiles["x"].astype(np.int64, copy=False)
        y = tiles["y"].astype(np.int64, copy=False)
    meta = json.loads(meta_path.read_text())
    return {
        "x": x,
        "y": y,
        "num_tiles": int(meta["num_tiles"]),
        "target_spacing_um": float(meta["target_spacing_um"]),
        "target_tile_size_px": int(meta["target_tile_size_px"]),
        "read_level": int(meta["read_level"]),
        "read_tile_size_px": int(meta["read_tile_size_px"]),
        "tile_size_lv0": int(meta["tile_size_lv0"]),
    }


def _load_legacy_artifact(
    *,
    wsi_path: str,
    output_dir: Path,
) -> dict[str, Any] | None:
    npy_path = output_dir / "coordinates" / f"{legacy_output_stem(wsi_path)}.npy"
    if not npy_path.is_file():
        return None
    data = np.load(npy_path, allow_pickle=False)
    x = np.asarray(data["x"], dtype=np.int64)
    y = np.asarray(data["y"], dtype=np.int64)
    metadata: dict[str, Any] = {
        "x": x,
        "y": y,
        "num_tiles": int(len(data)),
        "target_spacing_um": None,
        "target_tile_size_px": None,
        "read_level": None,
        "read_tile_size_px": None,
        "tile_size_lv0": None,
    }
    if len(data) > 0:
        metadata.update(
            {
                "target_spacing_um": float(data["target_spacing"][0]),
                "target_tile_size_px": int(data["target_tile_size"][0]),
                "read_level": int(data["tile_level"][0]),
                "read_tile_size_px": int(data["tile_size_resized"][0]),
                "tile_size_lv0": int(data["tile_size_lv0"][0]),
            }
        )
    return metadata


def _float_matches(left: float | None, right: float | None, *, tol: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= tol


def compare_branch_outputs(
    *,
    rows: list[dict[str, str | None]],
    current_output_dir: Path,
    legacy_output_dir: Path,
) -> list[dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    metadata_keys = (
        "num_tiles",
        "target_spacing_um",
        "target_tile_size_px",
        "read_level",
        "read_tile_size_px",
        "tile_size_lv0",
    )
    for row in rows:
        current = _load_current_artifact(
            sample_id=str(row["sample_id"]),
            output_dir=current_output_dir,
        )
        legacy = _load_legacy_artifact(
            wsi_path=str(row["wsi_path"]),
            output_dir=legacy_output_dir,
        )
        detail = ""
        coordinates_match = False
        metadata_match = False
        status = "match"
        if current is None and legacy is None:
            status = "missing_both_outputs"
            detail = "missing current and legacy artifact"
        elif current is None:
            status = "missing_current_output"
            detail = "missing current artifact"
        elif legacy is None:
            status = "missing_legacy_output"
            detail = "missing legacy artifact"
        else:
            coordinates_match = bool(
                np.array_equal(current["x"], legacy["x"])
                and np.array_equal(current["y"], legacy["y"])
            )
            mismatched_meta: list[str] = []
            for key in metadata_keys:
                left = current[key]
                right = legacy[key]
                if isinstance(left, float) or isinstance(right, float):
                    if not _float_matches(left, right):
                        mismatched_meta.append(key)
                elif left != right:
                    mismatched_meta.append(key)
            metadata_match = not mismatched_meta
            if not coordinates_match or not metadata_match:
                status = "mismatch"
                details: list[str] = []
                if not coordinates_match:
                    details.append("coordinate arrays differ")
                if mismatched_meta:
                    details.append(
                        "metadata mismatch: " + ", ".join(mismatched_meta)
                    )
                detail = "; ".join(details)
        comparison_rows.append(
            {
                "sample_id": row["sample_id"],
                "wsi_path": row["wsi_path"],
                "mask_path": row["mask_path"],
                "coordinates_match": coordinates_match,
                "metadata_match": metadata_match,
                "status": status,
                "detail": detail,
                "current_num_tiles": None if current is None else current["num_tiles"],
                "legacy_num_tiles": None if legacy is None else legacy["num_tiles"],
            }
        )
    return comparison_rows


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nBenchmark summary")
    print("=================")
    for row in summary_rows:
        print(
            f"{row['branch_label']} ({row['git_ref']}): "
            f"mean={row['mean_seconds']:.3f}s, "
            f"min={row['min_seconds']:.3f}s, "
            f"max={row['max_seconds']:.3f}s, "
            f"slides/s={row['mean_slides_per_second']}"
        )
    legacy = next((row for row in summary_rows if row["branch_label"] == "legacy"), None)
    current = next((row for row in summary_rows if row["branch_label"] == "current"), None)
    if legacy is not None and current is not None and current["mean_seconds"] > 0:
        speedup = legacy["mean_seconds"] / current["mean_seconds"]
        print(f"Current-vs-legacy speedup: {speedup:.3f}x")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    slides_csv = Path(args.slides_csv).resolve()
    config_file = Path(args.config_file).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    csv_rows = load_benchmark_rows(slides_csv)
    base_config = OmegaConf.to_container(
        OmegaConf.load(config_file),
        resolve=False,
    )
    if not isinstance(base_config, dict):
        raise ValueError("Config file must resolve to a mapping.")

    visualize_override = False if args.disable_visualize else None
    legacy_python = args.legacy_python or args.python
    current_ref = git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    legacy_worktree: Path | None = None

    try:
        legacy_worktree = create_legacy_worktree(
            repo_root=repo_root,
            legacy_ref=args.legacy_ref,
            workspace_root=output_root,
        )
        legacy_ref_resolved = resolve_git_ref(repo_root, args.legacy_ref)
        current_ref_resolved = resolve_git_ref(repo_root, "HEAD")

        records = []
        records.extend(
            benchmark_branch(
                branch_label="current",
                schema="current",
                repo_root=repo_root,
                python_executable=args.python,
                csv_rows=csv_rows,
                base_config=base_config,
                output_root=output_root / "runs",
                repeat=args.repeat,
                visualize_override=visualize_override,
                git_ref=f"{current_ref}@{current_ref_resolved}",
            )
        )
        records.extend(
            benchmark_branch(
                branch_label="legacy",
                schema="legacy",
                repo_root=legacy_worktree,
                python_executable=legacy_python,
                csv_rows=csv_rows,
                base_config=base_config,
                output_root=output_root / "runs",
                repeat=args.repeat,
                visualize_override=visualize_override,
                git_ref=f"{args.legacy_ref}@{legacy_ref_resolved}",
            )
        )
        summary_rows = summarize_records(records)
        runs_csv = write_csv(records, output_root / "benchmark_runs.csv")
        summary_csv = write_csv(summary_rows, output_root / "benchmark_summary.csv")
        current_run_one = next(
            record
            for record in records
            if record["branch_label"] == "current" and record["run_index"] == 1
        )
        legacy_run_one = next(
            record
            for record in records
            if record["branch_label"] == "legacy" and record["run_index"] == 1
        )
        output_comparison_rows = compare_branch_outputs(
            rows=csv_rows,
            current_output_dir=Path(current_run_one["output_dir"]),
            legacy_output_dir=Path(legacy_run_one["output_dir"]),
        )
        output_comparison_csv = write_csv(
            output_comparison_rows,
            output_root / "output_comparison.csv",
        )
        print_summary(summary_rows)
        print(f"\nPer-run results: {runs_csv}")
        print(f"Summary results: {summary_csv}")
        print(f"Output comparison: {output_comparison_csv}")

        failed_runs = [record for record in records if record["exit_code"] != 0]
        failed_output_checks = [
            row for row in output_comparison_rows if row["status"] != "match"
        ]
        if failed_runs:
            print("\nOne or more benchmark runs failed. Check the per-run log_path entries.")
            return 1
        if failed_output_checks:
            print(
                "\nCurrent and legacy outputs differ. "
                "Check output_comparison.csv for per-slide details."
            )
            return 1
        return 0
    finally:
        if legacy_worktree is not None and not args.keep_worktree:
            remove_legacy_worktree(repo_root=repo_root, worktree_path=legacy_worktree)


if __name__ == "__main__":
    raise SystemExit(main())
