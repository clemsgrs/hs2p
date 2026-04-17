# Lessons

## 2026-03-20

- When a user points to newly added planning files, re-check the workspace before
  relying on an earlier absence and then update the recommendation against the new
  source of truth.
- When proposing new metadata, prefer the smallest schema that matches the
  current invariants; if a single scalar step is enough, store that and add an
  assertion or inference path instead of introducing separate x/y fields by
  default.
- When tightening an internal contract, update the tests and stubs to the new
  shape instead of weakening the implementation with backward-compatibility
  fallbacks that hide missing required fields.
- When a regression stays green on CI unexpectedly, inspect the workflow and
  pytest marker defaults before assuming the checked-in fixture is still being
  exercised there.
- Do not infer grid stride from the minimum global coordinate gap after merging
  tiles from multiple contours; contour-local origin offsets can create tiny
  cross-contour gaps that collapse the inferred stride and disable grouped-read
  optimizations.
- When backend selection becomes part of the persisted artifact identity, resolve
  the effective backend before hashing or saving metadata; otherwise `auto` leaks
  into resume/dedup logic and breaks reproducibility.
- When suppressing native library startup noise, capture and filter file-descriptor
  stderr, not just Python-level `sys.stderr`, because C/C++ initialization messages
  may bypass `redirect_stderr()`.
- When moving work from `fork` to `spawn`, reinitialize logging inside the child
  process if you still need INFO-level library logs; spawned workers do not inherit
  the parent’s configured handlers the same way forked workers do.
- When a dependency probes CUDA during SAM2 startup, treat the whole SAM2 path as
  CUDA-sensitive even if the config selects `cpu`; library startup may still touch
  CUDA and fail under `fork`.
- When a library already owns resize/postprocess geometry, do not pre-resize the
  input to a fixed square in a wrapper; preserve the original aspect ratio and
  let the library's own transform stack handle normalization.
- If a model backend expects square inputs but the source geometry is
  rectangular, pad to square before inference and crop back after prediction;
  do not stretch the source image into a square.
- When a tar artifact needs stable tile identity but flexible write order, store
  per-tile identity in a manifest and derive archive member names from that
  identity; do not rely on tar member position to encode semantics.
- When a user asks for a production behavior change, do not stop at the
  benchmark harness; thread the option through the actual code path and then
  reuse the same config contract in the benchmark so the comparison stays
  representative.
- When the goal is the fastest CI benchmark, keep the checked-in benchmark
  fixture on the fastest supported encoder/backend path and push slower
  alternatives into explicit opt-in tests instead of the default benchmark.
- When trimming a commit after review, remove helper scripts and their
  dedicated tests unless they are part of the deliverable; do not leave
  exploratory tooling in the tree just because it was useful while iterating.
- When a user asks to keep files out of a commit, preserve them in the local
  worktree as untracked or unstaged files; do not delete the files themselves
  unless the user explicitly asks to remove them locally too.
- When tests exercise standalone or local-only scripts rather than committed
  package functionality, mark them separately and keep them out of the default
  CI run; otherwise CI will fail on files that are intentionally not part of
  the pushed branch.
- When rendering contour previews from level-0 geometry, scale the contour
  coordinates by the selected `vis_level` downsample before drawing; canvas
  padding changes the output size but does not change the contour coordinate
  system.
- When contour previews need a visual stroke width, scale it from the requested
  preview downsample and the resolved preview level instead of hardcoding a
  single pixel width; higher-resolution previews should render with a thicker
  outline.
- When normalizing OmegaConf nodes, treat them as `collections.abc.Mapping`
  rather than plain `dict`; `DictConfig` passes mapping semantics but fails
  `isinstance(x, dict)` checks, which can send config data down the wrong branch.
- When the user says no backward compatibility is needed, do not keep a legacy
  alias "just in case"; remove the alias, update the docs/tests, and make the
  new API the only supported surface.
- When removing a public config field, sweep all internal call sites in the same
  change; leaving one old attribute access behind will compile but fail at
  runtime in code paths the tests may not cover.
