# Release Notes

## Unreleased

### Shared automatic slide and mask backend priority

Automatic selection now probes both slide and source-mask paths independently with the same
openability-only priority:

`cucim -> vips -> openslide -> asap`

Explicit backend choices remain authoritative. Automatic selection still stops at the first
reader that opens a source; it does not inspect mask labels or retry after a later decode
failure.

Slide probes now receive `spacing_at_level_0`, allowing the override to rescue missing native
spacing metadata during automatic selection. Probe-time spacing-discrepancy warnings are
suppressed, while the selected reader continues to emit the single contextual warning.

This priority change can select a different resolved backend for an existing configuration
that uses `auto`. Because resume compatibility records the resolved slide and mask backends,
previous backend-dependent artifacts may be rejected and recomputed. Pin `backend` and
`mask_backend` explicitly when preserving the previous decoder is required.
