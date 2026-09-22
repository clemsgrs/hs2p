# ADR 0003: Keyword-only constructors and multi-argument signatures

- Status: Accepted
- Date: 2026-09-22

## Context

Inserting `mask_backend` into the middle of `TilingConfig`'s field list (#163) silently
rebound positional arguments: `TilingConfig(..., "asap", True)` bound `True` to
`mask_backend` instead of the field it used to occupy. Nothing failed; the value landed in
the wrong same-typed slot. #165 fixed that one class by making it keyword-only. The hazard
is general: most hs2p constructors and several methods take runs of same-typed
parameters (`get_tile(x, y, width, height, level)` is five ints), so any field insertion or
reordering can rebind a caller's positional arguments without an error.

## Decision

Keyword-only is the repo-wide convention, adopted in hs2p 5.0 as a deliberate break:

1. **Every public constructor is keyword-only.** Config and value dataclasses use
   `@dataclass(kw_only=True)`; hand-written `__init__` methods put `*` before their first
   parameter. There is no single-argument exemption for constructors: `Mask(*, path, ...)`
   already follows it. Internal and result-only dataclasses with two or more fields follow
   the same rule so a field insertion is safe everywhere, not only at the public boundary.
2. **Multi-argument functions and methods** (three or more parameters, or any two adjacent
   same-typed parameters) take keyword-only arguments. This part lands in #212.

Exemptions:

- Trivial one-argument helpers stay positional.
- Signatures that deliberately mirror an established external idiom stay positional, such
  as `read_region(location, level, size)`, which matches OpenSlide.

This is a reviewed convention, not a lint gate. No standard Ruff or flake8 rule requires
keyword-only parameters, so reviewers check new signatures against this ADR, and
`tests/test_keyword_only_constructors.py` pins the converted constructors.

## Consequences

- Adding or reordering a field can no longer silently rebind a caller's argument; a stale
  positional call fails with `TypeError` at the call site.
- Positional construction of `SegmentationConfig`, `FilterConfig`, `PreviewConfig`,
  `SlideSpec`, `TilingArtifacts`, `CompatibilitySpec`, `SamplingSpec`, `TilingResult`,
  `TileGeometry`, `ContourResult`, `ResolvedTissueMask`, `ResolvedAnnotationMasks`,
  `Sam2Thumbnail`, `LevelSelection`, `SpacingReadPlan`, and `WSI` breaks in 5.0. Callers
  pass keywords.
- Persisted artifacts are unaffected: coordinate metadata and `.npz` files are keyed by
  field name and carry no positional information.
- `dataclasses.replace`, `asdict`, and pickling are unaffected; only direct positional
  construction changes.
