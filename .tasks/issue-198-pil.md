# Issue #198 — PIL flat-raster backend

- [x] Add focused tests proving `.png`, `.jpg`, and `.jpeg` auto-routing is
      case-insensitive, selects only PIL, and does not alter multi-resolution
      backend order.
- [x] Add focused tests for corrupt and oversized flat-raster failures, including
      the inclusive project-owned pixel ceiling and no fallback/recommendation.
- [x] Add focused tests for the complete one-level reader protocol, spacing,
      region padding, thumbnails, cleanup/context management, RGB/RGBA handling,
      and grayscale/palette label preservation.
- [x] Implement the smallest PIL reader and format-policy routing needed to make
      each behavior pass.
- [x] Register `pil` in runtime/config backend sets and verify requested/resolved
      provenance.
- [x] Update README, API, CLI, and release notes to explain flat-raster `auto`
      selection and distinguish it from `speed.jpeg_backend: pil`.
- [x] Run targeted tests and the full suite.
- [x] Review `main...HEAD` along standards and issue-spec axes, fix findings, and
      re-run the full suite.
- [x] Commit, push `agent/issue-198-pil`, and open a draft PR containing
      `Closes #198`.
