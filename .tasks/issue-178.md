# Issue 178: shared automatic backend policy

## Scope

- [x] Resolve slide and mask sources independently with the shared automatic order
      `cucim -> vips -> openslide -> asap`.
- [x] Keep explicit backend choices authoritative and preserve requested/resolved provenance.
- [x] Pass the level-0 spacing override into slide probes, suppress only probe-time spacing
      discrepancy warnings, and let the selected reader emit the contextual warning.
- [x] Keep selection openability-only, with no decoded-label inspection or decode retry.
- [x] Update documentation and release notes, including the artifact-recomputation impact.

## Pre-agreed public seams

The issue acceptance criteria define these behavior seams:

- `resolve_backend(...)`: deterministic automatic fallback and explicit-backend authority.
- `resolve_backends(...)`: independent role resolution and requested/resolved provenance.
- `open_slide(..., backend="auto", spacing_override=...)`: spacing metadata rescue in default
  automatic mode and a single selected-reader warning.
- High-level tiling backend resolution: propagation of `SlideSpec.spacing_at_level_0`.

## TDD / verification

- [x] Red: add one deterministic failing example for each missing behavior.
- [x] Green: implement the minimum shared policy needed by each example.
- [x] Refactor: remove duplicate selection logic and keep compatibility where practical.
- [x] Run targeted backend-selection tests.
- [x] Run the full test suite.
- [x] Rebase onto current `origin/main` and re-run tests.
- [x] Run Standards and Spec reviews against `origin/main`; fix and re-green.
- [ ] Commit, push `agent/issue-178`, and open a draft PR with `Closes #178`.
