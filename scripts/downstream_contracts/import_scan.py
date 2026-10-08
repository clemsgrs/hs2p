"""AST inventory of a downstream project's hs2p imports, resolved against the installed hs2p.

Run with the interpreter of the contract environment, where the candidate hs2p is
installed; the downstream production modules themselves are only parsed, never imported.

    python -I import_scan.py --repo SRC --project NAME --revision SHA \
        --production-path PKG [--exclusions JSON] --out inventory.json

What this proves: every ``import hs2p...`` / ``from hs2p... import name`` (aliases and
function-local imports included) and every attribute read on a name bound to an hs2p
module exists in the candidate. What it does not prove: signatures, returned fields, or
behaviour; those are left to the contract suites and probes.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def tracked_python_files(repo: Path, production_paths: list[str]) -> list[str]:
    listing = _git(repo, "ls-files", "-z", "--", *production_paths)
    return sorted(path for path in listing.split("\0") if path.endswith(".py"))


def _is_target(module: str | None, package: str) -> bool:
    return module is not None and (module == package or module.startswith(f"{package}."))


def _guarded_lines(tree: ast.AST) -> set[int]:
    """Lines inside a ``try`` body whose handlers would swallow an ImportError."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches = False
        for handler in node.handlers:
            names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
            if handler.type is None or any(isinstance(n, ast.Name) and n.id in _IMPORT_ERRORS for n in names):
                catches = True
        if catches:
            for stmt in node.body:
                lines.update(range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1))
    return lines


def _attribute_chain(node: ast.Attribute) -> tuple[str, list[str]] | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    return current.id, list(reversed(parts))


def scan_file(path: Path, rel: str, *, package: str = "hs2p") -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    guarded = _guarded_lines(tree)
    records: list[dict[str, Any]] = []
    bindings: dict[str, str] = {}  # local name -> dotted hs2p object it is bound to

    def record(line: int, kind: str, module: str, name: str | None, alias: str | None) -> None:
        records.append(
            {"file": rel, "line": line, "kind": kind, "module": module, "name": name, "alias": alias, "guarded": line in guarded}
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if not _is_target(item.name, package):
                    continue
                record(node.lineno, "import", item.name, None, item.asname)
                if item.asname:
                    bindings[item.asname] = item.name
                else:
                    top = item.name.split(".")[0]
                    bindings[top] = top
        elif isinstance(node, ast.ImportFrom):
            if node.level or not _is_target(node.module, package):
                continue
            for item in node.names:
                record(node.lineno, "from", node.module, item.name, item.asname)
                if item.name != "*":
                    bindings[item.asname or item.name] = f"{node.module}.{item.name}"

    # Attribute reads on hs2p bindings (``hs2p_reader.open_slide``); outermost chain only.
    seen: set[tuple[int, str, str]] = set()
    inner: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in inner:
            continue
        current = node.value
        while isinstance(current, ast.Attribute):
            inner.add(id(current))
            current = current.value
        chain = _attribute_chain(node)
        if chain is None or chain[0] not in bindings:
            continue
        base, parts = bindings[chain[0]], ".".join(chain[1])
        key = (node.lineno, base, parts)
        if key not in seen:
            seen.add(key)
            record(node.lineno, "attribute", base, parts, chain[0])
    records.sort(key=lambda r: (r["line"], r["kind"] != "import", r["kind"] == "attribute"))
    return records


def _import(module: str) -> types.ModuleType:
    return importlib.import_module(module)


def _resolve_from(module: str, name: str) -> Any:
    """``from module import name`` semantics: attribute first, then submodule."""
    parent = _import(module)
    if name == "*" or hasattr(parent, name):
        return getattr(parent, name, parent)
    return _import(f"{module}.{name}")


def _resolve_dotted(dotted: str) -> Any:
    """Object a binding refers to: a module, or an attribute/submodule of one."""
    try:
        return _import(dotted)
    except ModuleNotFoundError:
        module, _, name = dotted.rpartition(".")
        return _resolve_from(module, name)


def resolve(record: dict[str, Any]) -> tuple[bool, str | None, bool]:
    """Return (resolved, error, checkable). Attribute reads are only checked on modules."""
    try:
        if record["kind"] == "import":
            _import(record["module"])
        elif record["kind"] == "from":
            _resolve_from(record["module"], record["name"])
        else:
            current = _resolve_dotted(record["module"])
            if not isinstance(current, types.ModuleType):
                return True, None, False
            qualified = current.__name__
            for part in record["name"].split("."):
                if not isinstance(current, types.ModuleType):
                    break  # only the module surface is an import contract
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    current = _import(f"{qualified}.{part}")
                qualified = f"{qualified}.{part}"
    except Exception as exc:  # noqa: BLE001 - any failure to resolve is the finding
        return False, f"{type(exc).__name__}: {exc}", True
    return True, None, True


def _exclusion_for(record: dict[str, Any], exclusions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for exclusion in exclusions:
        if exclusion["module"] != record["module"]:
            continue
        if exclusion.get("name") not in (None, record["name"]):
            continue
        if "file" in exclusion and exclusion["file"] != record["file"]:
            continue
        return exclusion
    return None


def build_inventory(
    repo: Path,
    *,
    project: str,
    expected_revision: str,
    production_paths: list[str],
    exclusions: list[dict[str, Any]],
    package: str = "hs2p",
) -> dict[str, Any]:
    repo = Path(repo)
    revision = _git(repo, "rev-parse", "HEAD").strip()
    if revision != expected_revision:
        raise RuntimeError(f"{project} checkout is at {revision}, not the expected revision {expected_revision}")
    files = tracked_python_files(repo, production_paths)
    records: list[dict[str, Any]] = []
    for rel in files:
        for record in scan_file(repo / rel, rel, package=package):
            resolved, error, checkable = resolve(record)
            if not checkable:
                continue  # attribute read on a class/function binding, not a module contract
            exclusion = _exclusion_for(record, exclusions)
            if resolved:
                status = "resolved"
            elif exclusion is not None:
                status = "excluded"
            else:
                status = "unresolved"
            record.update(revision=revision, status=status, error=error)
            if exclusion is not None:
                record["exclusion_reason"] = exclusion["reason"]
            records.append(record)
    imported = sys.modules.get(package)
    unresolved = [r for r in records if r["status"] == "unresolved"]
    return {
        "project": project,
        "revision": revision,
        "package": package,
        "package_origin": getattr(imported, "__file__", None),
        "production_paths": production_paths,
        "scanned_files": files,
        "records": records,
        "unresolved": unresolved,
        "excluded": [r for r in records if r["status"] == "excluded"],
        "ok": not unresolved,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--production-path", action="append", required=True)
    parser.add_argument("--exclusions", default="[]", help="JSON list of {module, name?, file?, reason}")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    inventory = build_inventory(
        args.repo,
        project=args.project,
        expected_revision=args.revision,
        production_paths=args.production_path,
        exclusions=json.loads(args.exclusions),
    )
    args.out.write_text(json.dumps(inventory, indent=2) + "\n")
    print(
        f"{args.project}@{inventory['revision'][:12]}: {len(inventory['records'])} hs2p import contracts in "
        f"{len(inventory['scanned_files'])} files, {len(inventory['unresolved'])} unresolved, "
        f"{len(inventory['excluded'])} excluded (hs2p from {inventory['package_origin']})"
    )
    for record in inventory["unresolved"]:
        symbol = f" -> {record['name']}" if record["name"] else ""
        print(f"  UNRESOLVED {record['file']}:{record['line']} {record['kind']} {record['module']}{symbol}: {record['error']}")
    return 0 if inventory["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
