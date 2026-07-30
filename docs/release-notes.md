# Release Notes

## Unreleased

### Format-aware automatic slide and mask backends

Flat `.png`, `.jpg`, and `.jpeg` slide or source-mask inputs now select the new
PIL reader directly under `auto` (case-insensitive). Pillow is a core dependency,
so ordinary flat benchmark images no longer require libvips or another native
WSI backend. Flat rasters have one level and require
`spacing_at_level_0`; PNG/JPEG density metadata is not interpreted as pathology
spacing. The input-reader setting `tiling.backend: pil` is separate from
`speed.jpeg_backend: pil`, which selects Pillow as the saved-tile JPEG encoder.

This routing is authoritative: corrupt, unsupported, or oversized flat rasters
fail through PIL without another backend probe or recommendation. PIL enforces
the project-owned `PIL_MAX_IMAGE_PIXELS` ceiling before pixel decoding,
independent of Pillow's mutable global ceiling. Explicit backend choices remain
authoritative.

Other inputs continue to probe slide and source-mask paths independently with
the same openability-only priority:

`cucim -> vips -> openslide -> asap`

PIL is never considered in that multi-resolution chain. Automatic selection
still stops at the first reader that opens a source; it does not inspect mask
labels or retry after a later decode failure.

Slide probes now receive `spacing_at_level_0`, allowing the override to rescue missing native
spacing metadata during automatic selection. Probe-time spacing-discrepancy warnings are
suppressed, while the selected reader continues to emit the single contextual warning.

This priority change can select a different resolved backend for an existing configuration
that uses `auto`. Because resume compatibility records the resolved slide and mask backends,
previous backend-dependent artifacts may be rejected and recomputed. Pin `backend` and
`mask_backend` explicitly when preserving the previous decoder is required.
