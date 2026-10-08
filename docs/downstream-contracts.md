# Downstream contract checks

[slide2vec](https://github.com/clemsgrs/slide2vec) and [soma](https://github.com/clemsgrs/soma) build on hs2p. Every hs2p pull request is checked against pinned revisions of both. The check runs on every pull request and can be started by hand. Locally and in CI it goes through one runner:

```shell
python scripts/downstream_contracts/run.py --hs2p .
```

| File | Role |
| --- | --- |
| `scripts/downstream_contracts/contracts.toml` | The single source of truth: pinned revisions, the CPU dependency set, the selected suites and probes, and the permitted skip and import exclusions. |
| `scripts/downstream_contracts/run.py` | Provisions the environment, fetches the downstream sources, runs every check and writes evidence. |
| `scripts/downstream_contracts/import_scan.py` | Builds the inventory of hs2p imports and resolves each one. |
| `scripts/downstream_contracts/plugin/hs2p_contract_guard.py` | Runs inside each downstream pytest process. It verifies hs2p's import origin and records each test's outcome. |
| `scripts/downstream_contracts/probes/` | hs2p-owned tests of dynamic contracts. They call downstream code; they never copy it. |
| `.github/workflows/downstream-contracts.yml` | Calls the runner on every pull request and on manual dispatch. |

## Running it locally

You need Python 3.11 or newer, `git` and network access. You do not need root, CUDA or any system packages.

```shell
python scripts/downstream_contracts/run.py --hs2p /path/to/hs2p-checkout
python scripts/downstream_contracts/run.py --hs2p . --soma-rev <full 40-char SHA>   # try another revision
python scripts/downstream_contracts/run.py --hs2p . --recreate-env                 # rebuild the environment
```

- The runner creates a virtualenv under `--work-dir` (default `~/.cache/hs2p-downstream-contracts`) and reuses it while the `[environment]` table and the Python version stay the same. Into it go the CPU torch wheels from `torch_index_url`, the listed `requirements` (including `openslide-bin`, the native OpenSlide library that `openslide-python` ≥ 1.4 loads) and the downstream packages, installed without their GPU/model dependencies. The candidate hs2p is installed **editable, last**, so nothing installed later can replace it. No model weights are downloaded.
- Each downstream checkout is a clean, detached checkout of the exact SHA. Each one sits in its own parent directory, because soma's `conftest.py` puts any sibling `slide2vec`/`hs2p` checkout on `sys.path`.
- Each project runs in its own pytest process.
- The exit status is `0` only if every check passes.

## What is checked

The runner does three things for each project:

1. **Import surface.** It finds the tracked production sources (`production_paths`; tests, docs and scripts are excluded) at the pinned SHA. It parses every `import hs2p…` and `from hs2p… import name`, including aliases and imports inside functions, and every attribute read on a name bound to an hs2p module (`hs2p_reader.open_slide`). Each one is resolved against the candidate the way Python resolves imports: attribute first, then submodule. Downstream modules are parsed, never imported. The result is `<project>/import-inventory.json`, which lists the SHA, file, line, module and name of each record, plus its status.
2. **Behaviour.** The selected suites run, followed by the probes:
   - **slide2vec suite** (`tests/test_hs2p5_integration.py`): exact-pixel grouped and single-tile reads, spacing-less masks, annotation grouping, previews and persisted artifacts.
   - **soma suites** (`tests/test_hs2p_4_tiling_cache.py`, `tests/test_supplied_coordinates.py`): the historical hs2p 4 tiling cache and supplied-coordinate artifact validation.
   - **Probes**: dynamic contracts that import resolution cannot see. They check that progress activation forwards hs2p events with their payloads, that slide2vec discovers backend suffixes from `*_SUPPORTED_SUFFIXES`, that spacing read plans produce the right hierarchical geometry, and that soma's dense reads at a given spacing are exact-pixel.
3. **Provenance.** The guard plugin checks where a fresh `import hs2p` resolves and where every loaded `hs2p` and `tests` module came from. It does this at session start, after collection (once every downstream `conftest.py` has edited `sys.path`) and at session end. The runner also confirms that each downstream checkout is still clean and at the expected SHA after its tests run.

Import resolution proves that names exist. It says nothing about signatures, returned fields or semantics; those are left to the suites and probes. The inventory is evidence about what downstream uses today. It does not promise a public API, and a name that hs2p never calls itself is not unused because of that.

## Evidence

Everything is written to `--out` (default `<work-dir>/evidence`). In CI it is uploaded as the `downstream-contracts-evidence` artifact, and `summary.md` becomes the job summary.

| File | Content |
| --- | --- |
| `summary.md`, `summary.json` | The verdict, the revisions, the environment and the problems found for each project. |
| `revisions.json` | The candidate path, SHA and dirty-file count, the import origin, and the downstream repositories and SHAs. |
| `setup-commands.log`, `constraints.txt`, `pip-freeze.txt` | Every setup command that ran, with its output, and the resolved environment. |
| `<project>/import-inventory.json` | The import inventory described above. |
| `<project>/pytest.log`, `junit.xml`, `session.json` | The pytest output, the JUnit report, and the guard's origin checks and outcome for each test. |
| `<project>/probes/`, `<project>/generated/` | The probe sources that ran, and the files the tests wrote (`--basetemp`). |

## Interpreting failures

| Problem in the summary | Meaning | Usual action |
| --- | --- | --- |
| `unresolved from hs2p.x -> name at file:line` | The candidate no longer provides a name that downstream production code imports. | Keep the name (or a compatible alias), or coordinate the downstream change and then bump the pin. |
| `<test>: failed` | A contract suite or probe assertion failed. | Read `<project>/pytest.log`. If the change is intended, adopt it downstream first. |
| `unexpected skip` | A required case did not run. This is usually a missing dependency. | Fix the environment in `contracts.toml`. A skip is never compatibility evidence. |
| `zero collected required cases` / `collection error` | A suite or probe file did not import, or was renamed. | Check the pin, and the probe if the downstream code was refactored. |
| `hs2p import origin mismatch` | The tests ran against an hs2p other than the candidate, or another checkout's `tests` package shadowed the downstream one. | Look at `sys_path` in `session.json` to see which entry won. |

## Updating pins

hs2p maintainers own the pins and change them in reviewed pull requests. Typical reasons are adopting a changed contract, or following a supported downstream release.

1. Preview the new revision. Run the runner with `--slide2vec-rev`/`--soma-rev <full SHA>`, or start the **Downstream contracts** workflow by hand with those inputs.
2. Update `revision` in `contracts.toml`. If the downstream suites moved, update `suites` and the probes in the same PR.
3. Review the change in the import inventory: new hs2p names that downstream now relies on are new contracts.
4. Exclusions need a rationale. Use `allowed_skips = [{ nodeid = "...", reason = "..." }]` for skips, and `import_exclusions = [{ module = "...", name = "...", file = "...", reason = "..." }]` for imports, where `name` and `file` are optional. Imports guarded by `try/except ImportError` are marked `guarded` in the inventory, but they still fail unless excluded.

## Making it a required check

The workflow's job is named `downstream-contracts`. It runs on every pull request event with no path filters or opt-in conditions, so the check is always reported. Making it required is a repository setting. A maintainer adds the `downstream-contracts` status check to the default-branch ruleset, next to `docker-test`.
