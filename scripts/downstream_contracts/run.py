#!/usr/bin/env python3
"""Check a candidate hs2p checkout against the pinned slide2vec and soma contracts.

    python scripts/downstream_contracts/run.py --hs2p .

One runner for local use and CI. It reads contracts.toml (pins, CPU dependency set,
suites, skip policy, import exclusions), creates or reuses an isolated CPU virtualenv,
fetches clean downstream checkouts at the pinned (or overridden) full SHAs, installs the
candidate hs2p editable after the dependencies, then per project:

1. AST-scans the tracked production sources for hs2p imports and resolves them against
   the candidate (import_scan.py), writing an inventory;
2. runs the contract suites plus hs2p's dynamic-contract probes in one pytest process
   with the hs2p_contract_guard plugin, which verifies hs2p's import origin before and
   after downstream conftests run and records every outcome;
3. fails on unresolved imports, collection errors, zero collected cases, unexpected skips,
   failed assertions or a foreign hs2p.

Evidence (revisions, environment, setup commands, inventories, pytest logs/JUnit/session
reports and the tests' generated files) is written to --out. See docs/downstream-contracts.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: pytest depends on tomli there
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "contracts.toml"
PLUGIN_DIR = HERE / "plugin"
IMPORT_SCAN = HERE / "import_scan.py"
ENV_SCHEMA = 1
ENV_MARKER = "hs2p-contracts-env.json"
EVIDENCE_MARKER = ".hs2p-downstream-contracts-evidence"
_SHA = re.compile(r"^[0-9a-f]{40}$")


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _full_sha(value: str) -> str:
    if not _SHA.match(value):
        raise argparse.ArgumentTypeError(f"{value!r} is not a full 40-character lowercase commit SHA")
    return value


def _default_work_dir() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "hs2p-downstream-contracts"


def parse_args(argv: list[str] | None = None, manifest: dict[str, Any] | None = None) -> argparse.Namespace:
    manifest = manifest or load_manifest()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hs2p", type=Path, help="candidate hs2p checkout to test (required)")
    for project in manifest["projects"]:
        parser.add_argument(
            f"--{project['name']}-rev",
            dest=f"rev_{project['name']}",
            type=_full_sha,
            metavar="SHA",
            help=f"full commit SHA overriding the pinned {project['name']} revision ({project['revision'][:12]})",
        )
    parser.add_argument("--work-dir", type=Path, default=_default_work_dir(), help="environment + sources (reused)")
    parser.add_argument("--out", type=Path, help="evidence directory (default: <work-dir>/evidence)")
    parser.add_argument("--python", default=sys.executable, help="base interpreter for the virtualenv")
    parser.add_argument("--recreate-env", action="store_true", help="discard and rebuild the environment")
    parser.add_argument(
        "--print-env-key", action="store_true", help="print the environment cache key (for CI caches) and exit"
    )
    args = parser.parse_args(argv)
    if args.hs2p is None and not args.print_env_key:
        parser.error("--hs2p is required: pass the candidate hs2p checkout explicitly")
    return args


def resolve_projects(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    projects = []
    for project in manifest["projects"]:
        override = getattr(args, f"rev_{project['name']}", None)
        projects.append({**project, "revision": override or project["revision"], "pinned": override is None})
    return projects


# --------------------------------------------------------------------------- policy


def session_problems(
    report: dict[str, Any] | None,
    *,
    returncode: int,
    required_files: list[str],
    allowed_skips: list[dict[str, Any]],
) -> list[str]:
    """Everything that makes one downstream pytest session unacceptable as evidence.

    ``returncode`` is the pytest process's own exit status: the guard writes ``report`` at
    sessionfinish, so a later failure (``pytest_unconfigure``, interpreter shutdown) shows
    only there.
    """
    if report is None:
        return ["no session report: the hs2p contract guard plugin did not run"]
    problems: list[str] = []
    for check in report["origin_checks"]:
        if not check["ok"]:
            problems.append(
                f"hs2p import origin mismatch at {check['checkpoint']}: resolves to {check.get('resolves_to')}, "
                f"imported {check.get('imported')}, expected {check.get('expected')}, "
                f"foreign hs2p modules {check.get('foreign_modules', [])}, "
                f"tests modules from outside the rootdir {check.get('foreign_tests_modules', [])}"
            )
    for error in report["collect_errors"]:
        problems.append(f"collection error in {error['nodeid'] or '<session>'}")
    allowed = {skip["nodeid"] for skip in allowed_skips}
    for result in report["results"]:
        if result["outcome"] == "failed":
            problems.append(f"{result['nodeid']}: failed")
        elif result["outcome"] in {"skipped", "xfailed"} and result["nodeid"] not in allowed:
            problems.append(f"{result['nodeid']}: unexpected skip ({result['outcome']}: {result['reason']})")
        elif result["outcome"] == "xpassed":
            problems.append(f"{result['nodeid']}: unexpected xpass")
    ran = {result["nodeid"] for result in report["results"]}
    for item in report["collected"]:
        if item["nodeid"] not in ran:
            problems.append(f"{item['nodeid']}: collected but never ran")
    collected = [item["path"] for item in report["collected"]]
    for required in required_files:
        if collected.count(required) == 0:
            problems.append(f"{required}: zero collected required cases")
    if report["exitstatus"] != 0 and not problems:
        problems.append(f"pytest exit status {report['exitstatus']}")
    if returncode != 0 and not problems:
        problems.append(f"pytest process exited {returncode}")
    return problems


# --------------------------------------------------------------------------- evidence directory


def prepare_evidence_dir(out: Path, *, protected: list[Path]) -> None:
    """Create ``out`` empty and mark it as runner-owned, or refuse to touch it.

    The previous contents are deleted only when ``out`` is not, and does not contain, a
    protected directory (the candidate checkout, the work dir) and is either empty or a
    directory an earlier run marked as its own. Anything else is left untouched.
    """
    out = out.resolve()
    for path in protected:
        path = path.resolve()
        if path == out or out in path.parents:
            raise SystemExit(f"refusing to use {out} as --out: it holds {path}, which the runner must not delete")
    if out.exists():
        if not out.is_dir():
            raise SystemExit(f"refusing to use {out} as --out: it is not a directory")
        if any(out.iterdir()) and not (out / EVIDENCE_MARKER).is_file():
            raise SystemExit(
                f"refusing to empty {out}: it is not a downstream-contracts evidence directory "
                f"(no {EVIDENCE_MARKER}); pass an empty or new --out"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / EVIDENCE_MARKER).write_text("Created by scripts/downstream_contracts/run.py; emptied on every run.\n")


# --------------------------------------------------------------------------- environment


class Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def run(self, cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True):
        line = f"$ {'cd ' + shlex.quote(str(cwd)) + ' && ' if cwd else ''}{shlex.join(str(c) for c in cmd)}"
        print(line, flush=True)
        with open(self.path, "a") as handle:
            handle.write(line + "\n")
            handle.flush()
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if check and proc.returncode != 0:
            raise RuntimeError(f"command failed ({proc.returncode}): {line}\nsee {self.path}")
        return proc


def env_key(manifest: dict[str, Any], python: str) -> str:
    probe = subprocess.run(
        [python, "-c", "import platform, sys; print(sys.version, platform.machine(), sys.platform)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = json.dumps({"schema": ENV_SCHEMA, "environment": manifest["environment"], "python": probe}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _env_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _constraints(environment: dict[str, Any]) -> list[str]:
    return [req for req in [*environment["torch_requirements"], *environment["requirements"]] if "[" not in req]


def ensure_environment(manifest: dict[str, Any], *, env_dir: Path, python: str, recreate: bool, log: Log) -> Path:
    environment = manifest["environment"]
    key = env_key(manifest, python)
    marker = env_dir / ENV_MARKER
    envpy = _env_python(env_dir)
    if not recreate and marker.exists() and envpy.exists():
        if json.loads(marker.read_text()).get("key") == key:
            print(f"reusing environment {env_dir} (key {key})")
            return envpy
    shutil.rmtree(env_dir, ignore_errors=True)
    if log.run([python, "-m", "venv", str(env_dir)], check=False).returncode != 0:
        # Some interpreters ship without ensurepip; bootstrap pip into the venv from outside.
        shutil.rmtree(env_dir, ignore_errors=True)
        log.run([python, "-m", "venv", "--without-pip", str(env_dir)])
        log.run([python, "-m", "pip", "--python", str(envpy), "install", "pip"])
    pip = [str(envpy), "-m", "pip", "install", "--disable-pip-version-check"]
    log.run([*pip, "--upgrade", "pip", "setuptools>=64", "wheel"])
    log.run([*pip, "--index-url", environment["torch_index_url"], *environment["torch_requirements"]])
    log.run([*pip, *environment["requirements"]])
    marker.write_text(json.dumps({"key": key, "environment": environment}, indent=2) + "\n")
    return envpy


def fetch_source(project: dict[str, Any], *, src_dir: Path, log: Log) -> Path:
    # One parent directory per checkout: soma's conftest prepends sibling ``slide2vec`` and
    # ``hs2p`` checkouts to sys.path, so no downstream checkout may have siblings.
    repo = src_dir / project["name"] / project["name"]
    if not (repo / ".git").exists():
        shutil.rmtree(repo, ignore_errors=True)
        repo.mkdir(parents=True)
        log.run(["git", "init", "-q"], cwd=repo)
        log.run(["git", "remote", "add", "origin", project["repository"]], cwd=repo)
    log.run(["git", "fetch", "-q", "--depth", "1", "origin", project["revision"]], cwd=repo)
    log.run(["git", "checkout", "-q", "--detach", "--force", project["revision"]], cwd=repo)
    log.run(["git", "clean", "-q", "-ffdx"], cwd=repo)
    head = _git_out(repo, "rev-parse", "HEAD")
    if head != project["revision"]:
        raise RuntimeError(f"{project['name']} checkout is at {head}, expected {project['revision']}")
    return repo


def _git_out(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _clean(repo: Path) -> bool:
    return _git_out(repo, "status", "--porcelain", "--untracked-files=no") == ""


def install_candidate(
    manifest: dict[str, Any], *, envpy: Path, hs2p: Path, sources: dict[str, Path], out: Path, log: Log
) -> str:
    environment = manifest["environment"]
    constraints = out / "constraints.txt"
    constraints.write_text("\n".join(_constraints(environment)) + "\n")
    pip = [str(envpy), "-m", "pip", "install", "--disable-pip-version-check"]
    # Downstream packages at the pinned revisions, without their (GPU/model-heavy)
    # dependency trees. Not editable: an editable install would put a checkout root, and
    # with it that project's ``tests`` package, on every process's sys.path.
    log.run([*pip, "--no-deps", "--force-reinstall", *(str(repo) for repo in sources.values())])
    # The candidate last, so nothing installed afterwards can replace it.
    extras = ",".join(environment["hs2p_extras"])
    log.run([*pip, "-c", str(constraints), "-e", f"{hs2p}[{extras}]"])
    origin = subprocess.run(
        [str(envpy), "-I", "-c", "import hs2p; print(hs2p.__file__)"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if not Path(origin).resolve().is_relative_to(hs2p / "hs2p"):
        raise RuntimeError(f"the environment imports hs2p from {origin}, not the candidate {hs2p}")
    return origin


def describe_environment(envpy: Path, out: Path) -> dict[str, Any]:
    script = (
        "import json, platform, sys, torch, openslide, PIL, numpy\n"
        "print(json.dumps({'python': sys.version, 'platform': platform.platform(),"
        " 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,"
        " 'openslide_python': openslide.__version__, 'openslide_library': openslide.__library_version__,"
        " 'pillow': PIL.__version__, 'numpy': numpy.__version__}))\n"
    )
    info = json.loads(subprocess.run([str(envpy), "-I", "-c", script], check=True, capture_output=True, text=True).stdout)
    freeze = subprocess.run([str(envpy), "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    (out / "pip-freeze.txt").write_text(freeze)
    return info


# --------------------------------------------------------------------------- per project


def stage_probes(probes: list[str], dest: Path) -> list[str]:
    """Copy hs2p's probe modules to ``dest`` and return their paths for pytest.

    pytest imports the ``__init__.py`` of every package above a test file and prepends the
    package root to sys.path. Run in place under ``scripts/`` that would put the runner's
    own hs2p checkout (and its ``tests`` package) on the downstream sys.path, so probes run
    from a directory with no package above it. The copies double as evidence.
    """
    dest = dest.resolve()
    for parent in [dest, *dest.parents]:
        if (parent / "__init__.py").exists():
            raise RuntimeError(f"cannot stage probes under {dest}: {parent / '__init__.py'} makes it a package")
    dest.mkdir(parents=True, exist_ok=True)
    staged = []
    for probe in probes:
        target = dest / Path(probe).name
        shutil.copyfile(HERE / probe, target)
        staged.append(str(target))
    return staged


def contract_session_env(inherited: Mapping[str, str], *, hs2p: Path, project_out: Path) -> dict[str, str]:
    """Environment for a downstream pytest process: the caller's, minus anything that changes
    what Python imports or what pytest runs (``PYTEST_ADDOPTS`` could add ``--collect-only``
    or ``-k``), so the runner's command line alone decides the session."""
    dropped = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
    return {
        **{k: v for k, v in inherited.items() if k not in dropped and not k.startswith("PYTEST_")},
        "PYTHONPATH": str(PLUGIN_DIR),
        "HS2P_EXPECTED_ORIGIN": str(hs2p / "hs2p"),
        "HS2P_CONTRACT_REPORT": str(project_out / "session.json"),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(project_out / "mpl"),
        "CUDA_VISIBLE_DEVICES": "",
    }


def run_project(
    project: dict[str, Any], *, repo: Path, envpy: Path, hs2p: Path, out: Path, log: Log
) -> dict[str, Any]:
    name = project["name"]
    project_out = out / name
    project_out.mkdir(parents=True, exist_ok=True)

    inventory_path = project_out / "import-inventory.json"
    scan_cmd = [
        str(envpy), "-I", str(IMPORT_SCAN),
        "--repo", str(repo), "--project", name, "--revision", project["revision"],
        "--exclusions", json.dumps(project.get("import_exclusions", [])),
        "--out", str(inventory_path),
    ]  # fmt: skip
    for path in project["production_paths"]:
        scan_cmd += ["--production-path", path]
    scan = log.run(scan_cmd, cwd=project_out, check=False)
    inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else None
    import_problems = (
        ["import scan did not produce an inventory"]
        if inventory is None
        else [
            f"unresolved {r['kind']} {r['module']}{' -> ' + r['name'] if r['name'] else ''} at {r['file']}:{r['line']}"
            for r in inventory["unresolved"]
        ]
    )
    if scan.returncode != 0 and not import_problems:
        import_problems.append(f"import scan exited {scan.returncode}")

    suites = [str((repo / suite).resolve()) for suite in project["suites"]]
    probes = stage_probes(project.get("probes", []), project_out / "probes")
    session_path = project_out / "session.json"
    session_path.unlink(missing_ok=True)
    env = contract_session_env(os.environ, hs2p=hs2p, project_out=project_out)
    pytest_cmd = [
        str(envpy), "-m", "pytest", "-p", "no:cacheprovider", "-p", "hs2p_contract_guard",
        "-o", "addopts=", "-c", str(repo / "pyproject.toml"), "--rootdir", str(repo),
        "-rA", "--junitxml", str(project_out / "junit.xml"), "--basetemp", str(project_out / "generated"),
        *suites, *probes,
    ]  # fmt: skip
    pytest_log = project_out / "pytest.log"
    print(f"$ cd {repo} && {shlex.join(pytest_cmd)}  > {pytest_log}", flush=True)
    with open(log.path, "a") as handle:
        handle.write(f"$ cd {shlex.quote(str(repo))} && {shlex.join(pytest_cmd)}\n")
    with open(pytest_log, "w") as handle:
        pytest_run = subprocess.run(pytest_cmd, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
    session = json.loads(session_path.read_text()) if session_path.exists() else None
    test_problems = session_problems(
        session,
        returncode=pytest_run.returncode,
        required_files=suites + probes,
        allowed_skips=project.get("allowed_skips", []),
    )
    if not _clean(repo) or _git_out(repo, "rev-parse", "HEAD") != project["revision"]:
        test_problems.append(f"{name} checkout changed during the run")

    outcomes: dict[str, int] = {}
    for result in (session or {}).get("results", []):
        outcomes[result["outcome"]] = outcomes.get(result["outcome"], 0) + 1
    return {
        "name": name,
        "repository": project["repository"],
        "revision": project["revision"],
        "pinned": project["pinned"],
        "suites": project["suites"],
        "probes": project.get("probes", []),
        "import_contracts": len(inventory["records"]) if inventory else 0,
        "import_resolved": sum(r["status"] == "resolved" for r in inventory["records"]) if inventory else 0,
        "import_excluded": len(inventory["excluded"]) if inventory else 0,
        "import_problems": import_problems,
        "hs2p_origin": [c.get("imported") or c.get("resolves_to") for c in (session or {}).get("origin_checks", [])][-1:],
        "outcomes": outcomes,
        "per_file": {
            path: sum(1 for item in (session or {}).get("collected", []) if item["path"] == path)
            for path in suites + probes
        },
        "test_problems": test_problems,
        "ok": not import_problems and not test_problems,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"### Downstream contracts: {'PASS' if summary['ok'] else 'FAIL'}",
        "",
        f"- hs2p candidate: `{summary['hs2p']['revision']}` (dirty files: {summary['hs2p']['dirty_files']}), "
        f"imported from `{summary['hs2p']['origin']}`",
        f"- environment: python {summary['environment']['python'].split()[0]}, torch {summary['environment']['torch']}, "
        f"openslide-python {summary['environment']['openslide_python']} / OpenSlide {summary['environment']['openslide_library']}",
        "",
        "| project | revision | import contracts | tests | result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for project in summary["projects"]:
        tests = ", ".join(f"{n} {k}" for k, n in sorted(project["outcomes"].items())) or "none"
        pin = "" if project["pinned"] else " (override)"
        lines.append(
            f"| {project['name']} | `{project['revision']}`{pin} | {project['import_resolved']}/{project['import_contracts']} resolved"
            f"{', ' + str(project['import_excluded']) + ' excluded' if project['import_excluded'] else ''} "
            f"| {tests} | {'PASS' if project['ok'] else 'FAIL'} |"
        )
    for project in summary["projects"]:
        for problem in project["import_problems"] + project["test_problems"]:
            lines.append(f"- **{project['name']}**: {problem}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    manifest = load_manifest()
    args = parse_args(argv, manifest)
    if args.print_env_key:
        print(env_key(manifest, args.python))
        return 0
    hs2p = args.hs2p.resolve()
    if not (hs2p / "hs2p" / "__init__.py").is_file() or not (hs2p / "pyproject.toml").is_file():
        raise SystemExit(f"{hs2p} is not an hs2p checkout")
    work = args.work_dir.resolve()
    out = (args.out or work / "evidence").resolve()
    prepare_evidence_dir(out, protected=[hs2p, work, HERE.parents[1]])
    log = Log(out / "setup-commands.log")
    projects = resolve_projects(manifest, args)

    envpy = ensure_environment(manifest, env_dir=work / "env", python=args.python, recreate=args.recreate_env, log=log)
    sources = {p["name"]: fetch_source(p, src_dir=work / "src", log=log) for p in projects}
    origin = install_candidate(manifest, envpy=envpy, hs2p=hs2p, sources=sources, out=out, log=log)
    environment = describe_environment(envpy, out)
    candidate = {
        "path": str(hs2p),
        "revision": _git_out(hs2p, "rev-parse", "HEAD") if (hs2p / ".git").exists() else "unknown",
        "dirty_files": len(_git_out(hs2p, "status", "--porcelain").splitlines()) if (hs2p / ".git").exists() else -1,
        "origin": origin,
    }
    (out / "revisions.json").write_text(
        json.dumps(
            {"hs2p": candidate, **{p["name"]: {"repository": p["repository"], "revision": p["revision"], "pinned": p["pinned"]} for p in projects}},
            indent=2,
        )
        + "\n"
    )

    results = [run_project(p, repo=sources[p["name"]], envpy=envpy, hs2p=hs2p, out=out, log=log) for p in projects]
    summary = {
        "ok": all(r["ok"] for r in results),
        "hs2p": candidate,
        "environment": {**environment, "key": env_key(manifest, args.python), "host": platform.node()},
        "projects": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = _markdown(summary)
    (out / "summary.md").write_text(markdown)
    print("\n" + markdown + f"\nevidence: {out}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
