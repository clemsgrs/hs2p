"""Behaviour of the downstream contract runner's own checks.

The runner itself provisions an environment and runs slide2vec/soma suites, which is too
heavy for the default test run. These tests pin the parts that decide pass/fail:
the AST import scan and its resolution against hs2p, the pytest session policy
(skips, empty suites, collection errors), the hs2p import-origin guard, the manifest,
and the CI wiring.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: pytest depends on tomli there
    import tomli as tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = ROOT / "scripts" / "downstream_contracts"


def _load(name: str):
    path = CONTRACTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"downstream_contracts_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _downstream_repo(tmp_path: Path, files: dict[str, str], *, untracked: dict[str, str] | None = None) -> tuple[Path, str]:
    repo = tmp_path / "downstream"
    repo.mkdir()
    _git(repo, "init", "-q")
    for rel, source in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    for rel, source in (untracked or {}).items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    return repo, _git(repo, "rev-parse", "HEAD")


# --------------------------------------------------------------------------- import scan

PRODUCTION_SOURCE = {
    "pkg/__init__.py": "",
    "pkg/top.py": """\
        import hs2p
        from hs2p import SlideSpec as Spec, tile_slides
        from hs2p.wsi import reader as hs2p_reader
        import hs2p.progress as hs2p_progress


        def run():
            from hs2p.wsi.geometry import plan_spacing_read
            hs2p_progress.activate_progress_reporter
            hs2p_reader.open_slide
            hs2p.wsi.reader.resolve_backend
            return plan_spacing_read, Spec, tile_slides
        """,
    "pkg/broken.py": """\
        from hs2p import NoSuchSymbolInHs2p


        def later():
            import hs2p.no_such_module
            from hs2p import progress as p
            return p.no_such_progress_helper
        """,
    "pkg/guarded.py": """\
        try:
            from hs2p import OptionalThingThatDoesNotExist
        except ImportError:
            OptionalThingThatDoesNotExist = None
        """,
    "tests/test_downstream.py": "from hs2p import NotProductionCode\n",
}


def _scan(tmp_path: Path, *, exclusions=(), untracked=None):
    import_scan = _load("import_scan")
    repo, sha = _downstream_repo(tmp_path, PRODUCTION_SOURCE, untracked=untracked)
    inventory = import_scan.build_inventory(
        repo,
        project="downstream",
        expected_revision=sha,
        production_paths=["pkg"],
        exclusions=list(exclusions),
    )
    return inventory, sha


def _keys(records, status=None):
    return {
        (r["file"], r["line"], r["kind"], r["module"], r["name"])
        for r in records
        if status is None or r["status"] == status
    }


def test_inventory_records_every_hs2p_import_with_sha_file_line_and_symbol(tmp_path):
    inventory, sha = _scan(tmp_path)

    assert inventory["project"] == "downstream"
    assert inventory["revision"] == sha
    resolved = _keys(inventory["records"], "resolved")
    assert {
        ("pkg/top.py", 1, "import", "hs2p", None),
        ("pkg/top.py", 2, "from", "hs2p", "SlideSpec"),
        ("pkg/top.py", 2, "from", "hs2p", "tile_slides"),
        ("pkg/top.py", 3, "from", "hs2p.wsi", "reader"),
        ("pkg/top.py", 4, "import", "hs2p.progress", None),
        # nested inside a function
        ("pkg/top.py", 8, "from", "hs2p.wsi.geometry", "plan_spacing_read"),
        # attribute contracts on module bindings
        ("pkg/top.py", 9, "attribute", "hs2p.progress", "activate_progress_reporter"),
        ("pkg/top.py", 10, "attribute", "hs2p.wsi.reader", "open_slide"),
        ("pkg/top.py", 11, "attribute", "hs2p", "wsi.reader.resolve_backend"),
    } <= resolved
    alias_record = next(r for r in inventory["records"] if r["name"] == "SlideSpec")
    assert alias_record["alias"] == "Spec"


def test_unresolved_symbols_modules_and_module_attributes_fail_the_scan(tmp_path):
    inventory, _ = _scan(tmp_path)

    assert _keys(inventory["unresolved"]) == {
        ("pkg/broken.py", 1, "from", "hs2p", "NoSuchSymbolInHs2p"),
        ("pkg/broken.py", 5, "import", "hs2p.no_such_module", None),
        ("pkg/broken.py", 7, "attribute", "hs2p.progress", "no_such_progress_helper"),
        ("pkg/guarded.py", 2, "from", "hs2p", "OptionalThingThatDoesNotExist"),
    }
    assert inventory["ok"] is False
    guarded = next(r for r in inventory["records"] if r["file"] == "pkg/guarded.py")
    assert guarded["guarded"] is True


def test_attribute_reads_resolve_aliases_in_their_lexical_scope(tmp_path):
    import_scan = _load("import_scan")
    repo, sha = _downstream_repo(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/scoped.py": """\
                import hs2p.wsi.reader as reader


                def progress_user():
                    import hs2p.progress as api
                    return api.activate_progress_reporter


                def broken_user():
                    import hs2p.progress as api
                    return api.no_such_progress_helper


                def reader_user():
                    import hs2p.wsi.reader as api

                    def nested():
                        return api.open_slide, reader.resolve_backend

                    return nested


                def spec_user(reader):
                    from hs2p import SlideSpec as api
                    return api.anything, reader.anything
                """,
        },
    )

    inventory = import_scan.build_inventory(
        repo, project="downstream", expected_revision=sha, production_paths=["pkg"], exclusions=[]
    )

    attributes = {
        (r["line"], r["module"], r["name"], r["status"]) for r in inventory["records"] if r["kind"] == "attribute"
    }
    assert attributes == {
        (6, "hs2p.progress", "activate_progress_reporter", "resolved"),
        (11, "hs2p.progress", "no_such_progress_helper", "unresolved"),
        (18, "hs2p.wsi.reader", "open_slide", "resolved"),
        (18, "hs2p.wsi.reader", "resolve_backend", "resolved"),
    }


def test_explicit_exclusions_are_recorded_with_their_rationale_and_do_not_fail(tmp_path):
    exclusions = [
        {"module": "hs2p", "name": "NoSuchSymbolInHs2p", "reason": "removed upstream on purpose"},
        {"module": "hs2p.no_such_module", "reason": "optional plugin"},
        {"module": "hs2p.progress", "name": "no_such_progress_helper", "reason": "dead code path"},
        {"module": "hs2p", "name": "OptionalThingThatDoesNotExist", "reason": "guarded fallback"},
    ]
    inventory, _ = _scan(tmp_path, exclusions=exclusions)

    assert inventory["unresolved"] == []
    assert inventory["ok"] is True
    excluded = {(r["module"], r["name"]): r["exclusion_reason"] for r in inventory["records"] if r["status"] == "excluded"}
    assert excluded[("hs2p", "NoSuchSymbolInHs2p")] == "removed upstream on purpose"
    assert excluded[("hs2p.no_such_module", None)] == "optional plugin"


def test_only_tracked_production_sources_are_scanned(tmp_path):
    inventory, _ = _scan(tmp_path, untracked={"pkg/scratch.py": "from hs2p import UntrackedJunk\n"})

    files = {r["file"] for r in inventory["records"]}
    assert files == {"pkg/top.py", "pkg/broken.py", "pkg/guarded.py"}
    assert inventory["scanned_files"] == ["pkg/__init__.py", "pkg/broken.py", "pkg/guarded.py", "pkg/top.py"]


def test_scan_refuses_a_checkout_that_is_not_at_the_expected_revision(tmp_path):
    import_scan = _load("import_scan")
    repo, _ = _downstream_repo(tmp_path, PRODUCTION_SOURCE)

    with pytest.raises(RuntimeError, match="expected revision"):
        import_scan.build_inventory(
            repo, project="downstream", expected_revision="0" * 40, production_paths=["pkg"], exclusions=[]
        )


def test_scan_does_not_import_downstream_production_modules(tmp_path):
    import_scan = _load("import_scan")
    repo, sha = _downstream_repo(
        tmp_path,
        {"pkg/__init__.py": "raise SystemExit('downstream module was imported')\nfrom hs2p import SlideSpec\n"},
    )

    inventory = import_scan.build_inventory(
        repo, project="downstream", expected_revision=sha, production_paths=["pkg"], exclusions=[]
    )

    assert inventory["ok"] is True
    assert "pkg" not in sys.modules


# --------------------------------------------------------------------------- session policy


def _session(results, *, collected=None, collect_errors=(), origin_ok=True, exitstatus=0):
    return {
        "exitstatus": exitstatus,
        "collected": collected if collected is not None else [{"nodeid": r["nodeid"], "path": r["path"]} for r in results],
        "collect_errors": list(collect_errors),
        "results": results,
        "origin_checks": [{"checkpoint": "sessionfinish", "ok": origin_ok}],
    }


def _result(path, name, outcome, reason=""):
    return {"nodeid": f"{path}::{name}", "path": path, "outcome": outcome, "reason": reason}


def test_policy_accepts_a_fully_passing_session():
    run = _load("run")
    report = _session([_result("/ds/tests/test_a.py", "t1", "passed"), _result("/ds/tests/test_b.py", "t2", "passed")])

    assert run.session_problems(
        report, returncode=0, required_files=["/ds/tests/test_a.py", "/ds/tests/test_b.py"], allowed_skips=[]
    ) == []


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (
            _session([_result("/ds/tests/test_a.py", "t1", "skipped", "openslide missing")]),
            "unexpected skip",
        ),
        (
            _session([_result("/ds/tests/test_a.py", "t1", "xfailed")]),
            "unexpected skip",
        ),
        (
            _session([_result("/ds/tests/test_a.py", "t1", "failed")], exitstatus=1),
            "failed",
        ),
        (
            _session([_result("/ds/tests/test_a.py", "t1", "passed")], collect_errors=[{"nodeid": "tests/test_c.py", "message": "ImportError"}], exitstatus=2),
            "collection error",
        ),
        (
            _session([_result("/ds/tests/test_a.py", "t1", "passed")], origin_ok=False),
            "hs2p import origin",
        ),
        (
            _session([_result("/ds/tests/test_a.py", "t1", "passed")], exitstatus=1),
            "pytest exit status 1",
        ),
    ],
)
def test_policy_rejects_skips_failures_collection_errors_and_foreign_hs2p(report, expected):
    run = _load("run")

    problems = run.session_problems(
        report, returncode=report["exitstatus"], required_files=["/ds/tests/test_a.py"], allowed_skips=[]
    )

    assert any(expected in p for p in problems), problems


def test_policy_rejects_a_required_suite_that_collected_nothing():
    run = _load("run")
    report = _session([_result("/ds/tests/test_a.py", "t1", "passed")])

    problems = run.session_problems(
        report, returncode=0, required_files=["/ds/tests/test_a.py", "/ds/tests/test_b.py"], allowed_skips=[]
    )

    assert problems == ["/ds/tests/test_b.py: zero collected required cases"]


def test_policy_rejects_collected_cases_that_never_ran():
    # e.g. an inherited --collect-only: pytest exits 0 with every case collected and none run.
    run = _load("run")
    collected = [{"nodeid": "/ds/tests/test_a.py::t1", "path": "/ds/tests/test_a.py"}]
    report = _session([], collected=collected)

    problems = run.session_problems(report, returncode=0, required_files=["/ds/tests/test_a.py"], allowed_skips=[])

    assert problems == ["/ds/tests/test_a.py::t1: collected but never ran"]


def test_policy_rejects_a_failed_pytest_process_even_when_its_report_passed():
    run = _load("run")
    report = _session([_result("/ds/tests/test_a.py", "t1", "passed")])

    problems = run.session_problems(report, returncode=1, required_files=["/ds/tests/test_a.py"], allowed_skips=[])

    assert problems == ["pytest process exited 1"]


def test_policy_rejects_a_missing_session_report():
    run = _load("run")

    assert run.session_problems(None, returncode=1, required_files=["/ds/tests/test_a.py"], allowed_skips=[]) == [
        "no session report: the hs2p contract guard plugin did not run"
    ]


def test_policy_permits_only_the_recorded_skip_exclusions():
    run = _load("run")
    report = _session(
        [_result("/ds/tests/test_a.py", "t1", "skipped", "needs GPU"), _result("/ds/tests/test_a.py", "t2", "passed")]
    )

    problems = run.session_problems(
        report,
        returncode=0,
        required_files=["/ds/tests/test_a.py"],
        allowed_skips=[{"nodeid": "/ds/tests/test_a.py::t1", "reason": "GPU-only case"}],
    )

    assert problems == []


# --------------------------------------------------------------------------- origin guard


def _guarded_pytest(
    tmp_path: Path,
    *,
    conftest: str,
    conftest_dir: str = "tests",
    extra_env: dict[str, str] | None = None,
    outside_rootdir: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess, dict]:
    candidate = tmp_path / "candidate"
    (candidate / "hs2p").mkdir(parents=True)
    (candidate / "hs2p" / "__init__.py").write_text("ORIGIN = 'candidate'\n")
    shadow = tmp_path / "shadow"
    (shadow / "hs2p").mkdir(parents=True)
    (shadow / "hs2p" / "__init__.py").write_text("ORIGIN = 'shadow'\n")
    project = tmp_path / "project"
    (project / conftest_dir).mkdir(parents=True)
    (project / conftest_dir / "conftest.py").write_text(textwrap.dedent(conftest).replace("SHADOW", str(shadow)))
    (project / conftest_dir / "test_uses_hs2p.py").write_text("def test_import():\n    import hs2p\n    assert hs2p.ORIGIN\n")
    report_path = tmp_path / "session.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(CONTRACTS_DIR / "plugin"), str(candidate)]),
        "HS2P_EXPECTED_ORIGIN": str(candidate / "hs2p"),
        "HS2P_CONTRACT_REPORT": str(report_path),
        **(extra_env or {}),
    }
    staged = tmp_path / "staged"
    staged.mkdir()
    for name, source in (outside_rootdir or {}).items():
        (staged / name).write_text(source)
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
            "-p", "hs2p_contract_guard", "--rootdir", str(project), "tests",
            *(str(staged / name) for name in outside_rootdir or {}),
        ],  # fmt: skip
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return proc, report


def test_guard_records_the_candidate_origin_in_each_checkpoint(tmp_path):
    proc, report = _guarded_pytest(tmp_path, conftest="")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert [c["checkpoint"] for c in report["origin_checks"]] == ["sessionstart", "collection_finish", "sessionfinish"]
    assert all(c["ok"] for c in report["origin_checks"])
    assert report["origin_checks"][-1]["imported"].startswith(str(tmp_path / "candidate" / "hs2p"))
    assert [r["outcome"] for r in report["results"]] == ["passed"]


@pytest.mark.parametrize(
    ("conftest_dir", "first_failing_checkpoint"),
    [("tests", "sessionstart"), ("tests/nested", "collection_finish")],
    ids=["initial-conftest", "conftest-loaded-during-collection"],
)
def test_guard_fails_the_session_when_a_conftest_puts_another_hs2p_first(tmp_path, conftest_dir, first_failing_checkpoint):
    proc, report = _guarded_pytest(
        tmp_path, conftest="import sys\nsys.path.insert(0, 'SHADOW')\n", conftest_dir=conftest_dir
    )

    assert proc.returncode != 0
    assert "hs2p import origin" in proc.stdout + proc.stderr
    failing = [c for c in report["origin_checks"] if not c["ok"]]
    assert failing and failing[0]["checkpoint"] == first_failing_checkpoint
    assert str(tmp_path / "shadow") in failing[0]["resolves_to"]
    assert report["results"] == []


def test_guard_reports_outcomes_under_the_same_ids_as_the_collected_cases(tmp_path):
    # hs2p's probes are staged outside the downstream rootdir; their results must still be
    # matched to their collected cases, or the runner cannot tell which cases ran.
    proc, report = _guarded_pytest(
        tmp_path, conftest="", outside_rootdir={"test_probe.py": "def test_probe():\n    import hs2p\n"}
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert {c["nodeid"] for c in report["collected"]} == {r["nodeid"] for r in report["results"]}
    probe = next(r for r in report["results"] if "test_probe" in r["nodeid"])
    assert probe["path"] == str((tmp_path / "staged" / "test_probe.py").resolve())


def test_runner_drops_inherited_pytest_options_from_contract_sessions(tmp_path):
    run = _load("run")
    inherited = {
        "PATH": "/usr/bin",
        "HOME": "/home/me",
        "PYTEST_ADDOPTS": "--collect-only -k nothing",
        "PYTEST_PLUGINS": "some_plugin",
        "PYTHONPATH": "/elsewhere",
    }

    env = run.contract_session_env(inherited, hs2p=tmp_path / "hs2p", project_out=tmp_path / "out")

    assert not any(key.startswith("PYTEST_") for key in env)
    assert env["PYTHONPATH"] == str(run.PLUGIN_DIR)
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/home/me"
    assert env["HS2P_EXPECTED_ORIGIN"] == str(tmp_path / "hs2p" / "hs2p")
    assert env["HS2P_CONTRACT_REPORT"] == str(tmp_path / "out" / "session.json")


def test_an_inherited_collect_only_session_is_not_accepted_as_evidence(tmp_path):
    run = _load("run")
    proc, report = _guarded_pytest(tmp_path, conftest="", extra_env={"PYTEST_ADDOPTS": "--collect-only"})

    assert proc.returncode == 0, proc.stdout + proc.stderr
    test_file = str((tmp_path / "project" / "tests" / "test_uses_hs2p.py").resolve())
    problems = run.session_problems(report, returncode=proc.returncode, required_files=[test_file], allowed_skips=[])

    assert problems == ["tests/test_uses_hs2p.py::test_import: collected but never ran"]


def test_a_session_that_fails_after_its_report_was_written_is_not_accepted_as_evidence(tmp_path):
    # The guard writes its report at sessionfinish; a later failure (pytest_unconfigure,
    # interpreter shutdown) only shows in the process exit status.
    run = _load("run")
    proc, report = _guarded_pytest(
        tmp_path, conftest="def pytest_unconfigure(config):\n    raise RuntimeError('late failure')\n"
    )

    assert proc.returncode != 0 and report["exitstatus"] == 0, proc.stdout + proc.stderr
    test_file = str((tmp_path / "project" / "tests" / "test_uses_hs2p.py").resolve())
    problems = run.session_problems(report, returncode=proc.returncode, required_files=[test_file], allowed_skips=[])

    assert problems == [f"pytest process exited {proc.returncode}"]


def test_guard_fails_when_another_projects_tests_package_shadows_the_downstream_tests(tmp_path):
    # slide2vec's tests/ has no __init__.py: any regular ``tests`` package on sys.path
    # (another checkout's root, an editable install's root) wins over it.
    candidate = tmp_path / "candidate"
    (candidate / "hs2p").mkdir(parents=True)
    (candidate / "hs2p" / "__init__.py").write_text("")
    foreign = tmp_path / "other_checkout"
    (foreign / "tests").mkdir(parents=True)
    (foreign / "tests" / "__init__.py").write_text("")
    (foreign / "tests" / "helpers.py").write_text("VALUE = 'foreign'\n")
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "helpers.py").write_text("VALUE = 'own'\n")
    (project / "tests" / "test_helpers.py").write_text(
        "def test_helpers():\n    from tests.helpers import VALUE\n    assert VALUE\n"
    )
    report_path = tmp_path / "session.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(CONTRACTS_DIR / "plugin"), str(candidate), str(foreign)]),
        "HS2P_EXPECTED_ORIGIN": str(candidate / "hs2p"),
        "HS2P_CONTRACT_REPORT": str(report_path),
    }

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "-p", "hs2p_contract_guard", "tests"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    report = json.loads(report_path.read_text())
    failing = [c for c in report["origin_checks"] if not c["ok"]]
    assert failing and any(str(foreign) in entry for entry in failing[0]["foreign_tests_modules"])


def test_probes_are_staged_outside_any_python_package(tmp_path):
    # pytest imports the __init__.py of every package above a test file and prepends that
    # package's parent to sys.path; from scripts/ that would be the runner's hs2p checkout.
    run = _load("run")
    probes = run.stage_probes(["probes/test_soma_dynamic_contracts.py"], tmp_path / "staged")

    assert probes == [str((tmp_path / "staged" / "test_soma_dynamic_contracts.py").resolve())]
    assert Path(probes[0]).read_text() == (CONTRACTS_DIR / "probes" / "test_soma_dynamic_contracts.py").read_text()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    with pytest.raises(RuntimeError, match="__init__.py"):
        run.stage_probes(["probes/test_soma_dynamic_contracts.py"], tmp_path / "pkg" / "staged")


# --------------------------------------------------------------------------- evidence directory


def _checkout(path: Path) -> Path:
    (path / "hs2p").mkdir(parents=True)
    (path / "hs2p" / "__init__.py").write_text("")
    (path / ".git").mkdir()
    (path / "work.txt").write_text("uncommitted")
    return path


@pytest.mark.parametrize("target", ["candidate", "ancestor-of-candidate", "work-dir", "ancestor-of-work-dir"])
def test_evidence_dir_refuses_the_candidate_the_work_dir_and_their_ancestors(tmp_path, target):
    run = _load("run")
    candidate = _checkout(tmp_path / "parent" / "hs2p")
    work = tmp_path / "cache" / "work"
    (work / "env").mkdir(parents=True)
    out = {
        "candidate": candidate,
        "ancestor-of-candidate": tmp_path / "parent",
        "work-dir": work,
        "ancestor-of-work-dir": tmp_path / "cache",
    }[target]
    # Even a stray ownership marker must not make a protected directory deletable.
    (out / run.EVIDENCE_MARKER).write_text("")

    with pytest.raises(SystemExit, match="refusing"):
        run.prepare_evidence_dir(out, protected=[candidate, work])

    assert (candidate / "work.txt").read_text() == "uncommitted"
    assert (candidate / ".git").is_dir() and (work / "env").is_dir()


def test_evidence_dir_refuses_a_non_empty_directory_the_runner_did_not_create(tmp_path):
    run = _load("run")
    out = tmp_path / "results"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")

    with pytest.raises(SystemExit, match="not a downstream-contracts evidence directory"):
        run.prepare_evidence_dir(out, protected=[])

    assert (out / "notes.txt").read_text() == "keep me"


def test_evidence_dir_is_created_or_emptied_and_marked_as_runner_owned(tmp_path):
    run = _load("run")
    fresh, empty, previous = tmp_path / "fresh", tmp_path / "empty", tmp_path / "previous"
    empty.mkdir()
    run.prepare_evidence_dir(previous, protected=[])
    (previous / "soma").mkdir()
    (previous / "soma" / "session.json").write_text("{}")

    for out in (fresh, empty, previous):
        run.prepare_evidence_dir(out, protected=[])

        assert sorted(p.name for p in out.iterdir()) == [run.EVIDENCE_MARKER]


# --------------------------------------------------------------------------- manifest and CI wiring


def test_manifest_pins_full_revisions_suites_and_requirements_in_one_place():
    manifest = tomllib.loads((CONTRACTS_DIR / "contracts.toml").read_text())

    projects = {p["name"]: p for p in manifest["projects"]}
    assert set(projects) == {"slide2vec", "soma"}
    for project in projects.values():
        assert len(project["revision"]) == 40 and int(project["revision"], 16) >= 0
        assert project["repository"].startswith("https://github.com/clemsgrs/")
        for skip in project.get("allowed_skips", []):
            assert skip["reason"].strip()
        for exclusion in project.get("import_exclusions", []):
            assert exclusion["reason"].strip()
    assert projects["slide2vec"]["suites"] == ["tests/test_hs2p5_integration.py"]
    assert projects["soma"]["suites"] == ["tests/test_hs2p_4_tiling_cache.py", "tests/test_supplied_coordinates.py"]
    environment = manifest["environment"]
    assert environment["torch_index_url"].endswith("/whl/cpu")
    assert "openslide" in environment["hs2p_extras"]


@pytest.mark.parametrize("override", ["d7aa60b", "main", "g" * 40])
def test_runner_rejects_revision_overrides_that_are_not_full_shas(override):
    run = _load("run")

    with pytest.raises(SystemExit):
        run.parse_args(["--hs2p", str(ROOT), "--slide2vec-rev", override])


def test_runner_requires_an_explicit_candidate_checkout():
    run = _load("run")

    with pytest.raises(SystemExit):
        run.parse_args([])


def test_runner_applies_full_sha_overrides_on_top_of_the_pinned_defaults():
    run = _load("run")
    args = run.parse_args(["--hs2p", str(ROOT), "--soma-rev", "a" * 40])

    projects = {p["name"]: p for p in run.resolve_projects(run.load_manifest(), args)}

    assert projects["soma"]["revision"] == "a" * 40
    assert projects["slide2vec"]["revision"] == run.load_manifest()["projects"][0]["revision"]


def test_ci_runs_the_same_runner_on_every_pull_request_and_on_dispatch():
    workflow = (ROOT / ".github" / "workflows" / "downstream-contracts.yml").read_text()

    assert "pull_request:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "scripts/downstream_contracts/run.py" in workflow
    assert "label" not in workflow.lower()
    assert "paths:" not in workflow
