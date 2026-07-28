# Issue #177 — Preserve recorded output provenance on resume

- [x] Confirm the public seam: `hs2p.api.tile_slides`.
- [x] Trace successful-resume artifact reconstruction and process-list rewriting.
- [x] Red: add a public batch API regression test for preserved deleted TAR/preview paths, untouched downstream outputs, unrelated metadata, and coordinate validation.
- [x] Green: preserve recorded downstream provenance without filesystem validation or regeneration.
- [x] Document that downstream files are user-managed after successful completion.
- [x] Run the focused regression test and full test suite.
- [x] Rebase onto `origin/main` and re-run tests.
- [x] Run the two-axis `/code-review origin/main` against issue #177, fix findings, and re-green.
- [ ] Commit, push, and open a draft PR containing `Closes #177`.
