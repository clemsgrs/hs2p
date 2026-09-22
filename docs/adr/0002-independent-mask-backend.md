# ADR 0002: Resolve the mask backend independently from the slide backend

- Status: Accepted
- Date: 2026-07-21
- Amended: 2026-07-31 by the flat-raster format policy (#198)
- Amended: 2026-09-22 by the first-class `Mask` contract ([ADR 0004](0004-first-class-mask.md),
  #167, #195): the `open_mask_reader` helper and the `WSI` attached-mask path this ADR
  originally described no longer exist. The independent mask-backend decision stands; every
  source mask is now opened as an `hs2p.mask.Mask`.

## Context

ADR 0001 made the backend selected for a run authoritative for mask decoding, with no retry or
fallback. That backend, however, was the slide's backend: a mask was always decoded with
whatever backend opened the slide. Some masks need a different decoder than their slide (for
example a deflate-compressed label TIFF that OpenSlide decodes but the slide's cuCIM backend
cannot), and there was no way to express that without changing the slide backend too.

## Decision

`TilingConfig` gains a second field, `mask_backend`, alongside `backend`. The slide backend is
resolved only from the slide path and the mask backend only from the source-mask path — the two
roles are independent, and neither role's backend resolution influences the
other. Both share one format-aware selection policy: `auto` selects only PIL for
`.png`, `.jpg`, and `.jpeg` suffixes, and uses the cuCIM, VIPS, OpenSlide, then
ASAP openability chain for other inputs. PIL is never included in that chain.
Selection never inspects decoded label semantics or retries after selection. A
selected decoder is authoritative for that read (ADR 0001 still holds). A
native-backend slide probe receives the user's level-0 spacing override so it
can open a slide with missing native spacing metadata; probe-time
spacing-conflict warnings are suppressed, and only the selected reader emits
the contextual warning.

Both fields accept only `auto`, `cucim`, `asap`, `openslide`, `pil`, `vips`; null and unknown values
fail configuration validation, including when a `TilingConfig` is constructed directly in
Python (which is keyword-only, and validates the `requested_*` provenance fields on the same
allowlist). A slide with no source mask never resolves or validates mask-backend availability,
and its mask provenance is null. An explicit mask backend applies to every source-mask read —
precomputed tissue masks, annotation masks, overlays, and deferred preview reads.

Every source mask is opened as an `hs2p.mask.Mask` (#167), which is decoupled from the slide
backend: its default `backend="auto"` resolves the mask backend independently from the mask
path, never inheriting the slide's backend, and the mask exposes only the concrete backend it
opened. The high-level pipeline always passes the explicit resolved `mask_backend` and records
the requested value alongside it.

Opening a source mask is centralized in `Mask` so an open failure — from backend resolution or
the open itself — is reraised with actionable context naming the mask path and the backend,
rather than a raw codec error. The format policy has no fallback: a selected decoder that
cannot open the mask fails there. Preprocessing, overlays, and previews all route through it.

Requested and resolved values are kept separate for both roles (`requested_backend` /
`backend`, `requested_mask_backend` / `mask_backend`) and persisted in the tiling metadata,
`TilingArtifacts`, and `process_list.csv`. Requested values are provenance only. Resume compares
the resolved slide backend and, when a source mask exists, the resolved mask backend; it does
not reject artifacts merely because a requested value differs. When `mask_backend: auto` selects
a backend, a distinct `mask_backend.selected` progress event names the sample, mask path,
resolved backend, and reason.

## Consequences

- A mask can use a different decoder than its slide without changing the slide backend.
- For non-flat inputs, a native backend that opens but cannot decode a mask can
  be selected and then fail at read time; the fix is to set `mask_backend`
  explicitly.
- Flat raster suffixes fail through PIL without a native-backend fallback.
- Existing `auto` configurations can resolve to a different backend under the shared priority,
  so backend-dependent artifacts may need to be recomputed.
- Pre-#163 metadata and `process_list.csv` schemas (without the mask backend fields/columns)
  are rejected clearly rather than loaded.
