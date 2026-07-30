# Issue #197 — Content-aware spacing-read upsampling policy

## Scope

- [x] Centralize the image-versus-label upsampling decision in the spacing-read
      planner and reuse that decision for full reads.
- [x] Make public region/full spacing reads image-safe by default.
- [x] Keep finer label reads available only through explicit label semantics and
      nearest-neighbour interpolation.
- [x] Remove the duplicate tiling pre-check and identify tiling reads as image
      content.
- [x] Document the image/label asymmetry and safe public defaults.

## Pre-agreed public seams

The issue acceptance criteria explicitly define these behavior seams:

- `plan_spacing_read(...)`: centralized content-kind validation and finer-than-level-0
  policy.
- `WSI.read_region_at_spacing(...)`: image-safe default before backend access.
- `WSI.read_full_at_spacing(...)`: the same policy before loading a full level.
- `read_label_region_at_spacing(...)` and `read_label_at_spacing(...)`: explicit
  label semantics with exact nearest-neighbour output.
- `generate_tiles(...)`: centralized planner policy after duplicate-check removal.

## TDD / verification

- [x] Red → green: image planner rejects forbidden upsampling and validates content
      kinds while tolerance/exact/downsample behavior stays intact.
- [x] Red → green: region/full public defaults reject before backend access.
- [x] Red → green: label helpers permit finer reads and return exact replicated labels.
- [x] Red → green: `generate_tiles` reaches the shared image policy.
- [x] Run focused spacing-read and tiling regressions.
- [x] Run the full test suite.
- [x] Run Standards and Spec reviews against `main`; fix findings and re-green.
- [ ] Commit, push, and open a PR containing `Closes #197`.
