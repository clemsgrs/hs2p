"""pytest plugin loaded (``-p hs2p_contract_guard``) into every downstream contract process.

It proves the downstream tests exercised the candidate hs2p and records what ran:

* ``HS2P_EXPECTED_ORIGIN``: the candidate's ``hs2p`` package directory. At session start,
  after collection (so after every downstream conftest has edited ``sys.path``) and at
  session end, the guard checks both where a fresh ``import hs2p`` would resolve and where
  every already-imported ``hs2p`` module came from. A mismatch stops the session.
  The same checkpoints verify that every imported ``tests``/``tests.*`` module comes from
  the downstream rootdir, so no other checkout's test package shadows the downstream one.
* ``HS2P_CONTRACT_REPORT``: JSON file receiving the origin checks, the collected items,
  collection errors and per-test outcomes (with skip reasons) for the runner's policy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_STATE: dict = {"rootdir": None, "origin_checks": [], "collected": [], "collect_errors": [], "results": {}}


def _expected() -> Path:
    return Path(os.environ["HS2P_EXPECTED_ORIGIN"]).resolve()


def _within(path: str | None, root: Path) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _fresh_import_spec(name: str):
    """Spec a fresh ``import name`` would use now: sys.meta_path in order, cache ignored.

    This sees both sys.path entries (and conftest edits to them) and the meta-path finder
    an editable install registers.
    """
    for finder in sys.meta_path:
        find_spec = getattr(finder, "find_spec", None)
        if find_spec is None:
            continue
        try:
            spec = find_spec(name, None)
        except Exception:  # noqa: BLE001 - a broken finder is not hs2p's origin
            continue
        if spec is not None:
            return spec
    return None


def _module_locations(module) -> list[str]:
    if getattr(module, "__file__", None):
        return [module.__file__]
    return [str(entry) for entry in getattr(module, "__path__", [])]  # namespace package


def _foreign_tests_modules(rootdir: Path) -> list[str]:
    return sorted(
        f"{name}={location}"
        for name, module in list(sys.modules.items())
        if name == "tests" or name.startswith("tests.")
        for location in _module_locations(module)
        if not _within(location, rootdir)
    )


def _check(checkpoint: str) -> dict:
    expected = _expected()
    rootdir = Path(_STATE["rootdir"] or os.getcwd()).resolve()
    foreign_tests = _foreign_tests_modules(rootdir)
    spec = _fresh_import_spec("hs2p")
    resolves_to = spec.origin if spec is not None else None
    imported = getattr(sys.modules.get("hs2p"), "__file__", None)
    foreign = sorted(
        f"{name}={module.__file__}"
        for name, module in list(sys.modules.items())
        if (name == "hs2p" or name.startswith("hs2p."))
        and getattr(module, "__file__", None)
        and not _within(module.__file__, expected)
    )
    ok = (
        _within(resolves_to, expected)
        and (imported is None or _within(imported, expected))
        and not foreign
        and not foreign_tests
    )
    check = {
        "checkpoint": checkpoint,
        "expected": str(expected),
        "resolves_to": resolves_to,
        "imported": imported,
        "foreign_modules": foreign,
        "rootdir": str(rootdir),
        "foreign_tests_modules": foreign_tests,
        "sys_path": list(sys.path),
        "ok": ok,
    }
    _STATE["origin_checks"].append(check)
    return check


def _fail_message(check: dict) -> str:
    return (
        f"hs2p import origin check failed at {check['checkpoint']}: expected {check['expected']}, "
        f"import resolves to {check['resolves_to']}, imported module {check['imported']}, "
        f"foreign hs2p modules {check['foreign_modules']}, "
        f"tests modules from outside {check['rootdir']}: {check['foreign_tests_modules']}"
    )


def _write_report(exitstatus) -> None:
    path = os.environ.get("HS2P_CONTRACT_REPORT")
    if not path:
        return
    report = {
        "exitstatus": int(exitstatus),
        "rootdir": _STATE["rootdir"],
        "origin_checks": _STATE["origin_checks"],
        "collected": _STATE["collected"],
        "collect_errors": _STATE["collect_errors"],
        "results": list(_STATE["results"].values()),
    }
    Path(path).write_text(json.dumps(report, indent=2) + "\n")


def pytest_sessionstart(session):
    _STATE["rootdir"] = str(session.config.rootpath)
    check = _check("sessionstart")
    print(f"\nhs2p resolves to {check['resolves_to']} (expected {check['expected']})")
    if not check["ok"]:
        _write_report(4)
        pytest.exit(_fail_message(check), returncode=4)


def pytest_collectreport(report):
    if report.failed:
        _STATE["collect_errors"].append({"nodeid": report.nodeid, "message": str(report.longrepr)[-2000:]})


def pytest_collection_finish(session):
    for item in session.items:
        path = str(Path(item.path).resolve())
        _STATE["collected"].append({"nodeid": _nodeid(item.nodeid, path), "path": path})
    check = _check("collection_finish")
    if not check["ok"]:
        _write_report(4)
        pytest.exit(_fail_message(check), returncode=4)


def _nodeid(nodeid: str, path: str | None) -> str:
    # Files outside the rootdir (hs2p's staged probes) get ids like "::test_x".
    return f"{Path(path).name}{nodeid}" if nodeid.startswith("::") and path else nodeid


def pytest_runtest_logreport(report):
    path = str(Path(report.fspath).resolve()) if report.fspath else None
    nodeid = _nodeid(report.nodeid, path)
    entry = _STATE["results"].setdefault(nodeid, {"nodeid": nodeid, "path": path, "outcome": "passed", "reason": ""})
    if report.failed:
        entry["outcome"] = "failed"
        entry["reason"] = str(report.longrepr)[-2000:]
    elif hasattr(report, "wasxfail"):
        entry["outcome"] = "xfailed" if report.skipped else "xpassed"
        entry["reason"] = report.wasxfail
    elif report.skipped and entry["outcome"] != "failed":
        entry["outcome"] = "skipped"
        longrepr = report.longrepr
        entry["reason"] = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session, exitstatus):
    check = _check("sessionfinish")
    yield
    if not check["ok"]:
        session.exitstatus = 4
    _write_report(session.exitstatus)


def pytest_terminal_summary(terminalreporter):
    for check in _STATE["origin_checks"]:
        status = "ok" if check["ok"] else "MISMATCH"
        terminalreporter.write_line(
            f"hs2p origin [{check['checkpoint']}] {status}: resolves to {check['resolves_to']}, imported {check['imported']}"
        )
        if not check["ok"]:
            terminalreporter.write_line(_fail_message(check))
