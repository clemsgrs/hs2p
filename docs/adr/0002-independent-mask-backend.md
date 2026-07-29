# ADR 0002: Resolve the mask backend independently from the slide backend

- Status: Accepted
- Date: 2026-07-21

## Context

ADR 0001 made the backend selected for a run authoritative for mask decoding, with no retry or
fallback. That backend, however, was the slide's backend: a mask was always decoded with
whatever backend opened the slide. Some masks need a different decoder than their slide (for
example a deflate-compressed label TIFF that OpenSlide decodes but the slide's cuCIM backend
cannot), and there was no way to express that without changing the slide backend too.

## Decision

`TilingConfig` gains a second field, `mask_backend`, alongside `backend`. The slide backend is
resolved only from the slide path and the mask backend only from the source-mask path — the two
roles are independent, and neither role's openability probe influences the other. Both share
one selection policy: `auto` checks openability only (open then close to pick the first backend
that opens the file) and never inspects decoded label semantics or retries after selection. A
selected decoder is authoritative for that read (ADR 0001 still holds).
The shared automatic priority is cuCIM, VIPS, OpenSlide, then ASAP. A slide probe receives the
user's level-0 spacing override so it can open a slide with missing native spacing metadata;
probe-time spacing-conflict warnings are suppressed, and only the selected reader emits the
contextual warning.

Both fields accept only `auto`, `cucim`, `asap`, `openslide`, `vips`; null and unknown values
fail configuration validation, including when a `TilingConfig` is constructed directly in
Python (which is keyword-only, and validates the `requested_*` provenance fields on the same
allowlist). A slide with no source mask never resolves or validates mask-backend availability,
and its mask provenance is null. An explicit mask backend applies to every source-mask read —
precomputed tissue masks, annotation masks, the public low-level readers, overlays, and
deferred preview reads.

The public low-level mask readers are decoupled from the slide backend: called **without** a
`mask_backend` (omitted or `None`) they resolve the mask backend independently from the mask
path via `auto`, never inheriting the slide's backend, and record `requested_mask_backend ==
"auto"`. The high-level pipeline is unaffected because it always passes an explicit resolved
`mask_backend`.

Opening a source mask is centralized through one helper (`open_mask_reader`) so an open failure
— from backend resolution or the open itself — is reraised with actionable context naming the
mask path and the requested backend (and the resolved backend when known), rather than a raw
codec error; the remedy is to set `mask_backend` explicitly. Overlays and the `WSI` attached-mask
path route through this helper.

Requested and resolved values are kept separate for both roles (`requested_backend` /
`backend`, `requested_mask_backend` / `mask_backend`) and persisted in the tiling metadata,
`TilingArtifacts`, and `process_list.csv`. Requested values are provenance only. Resume compares
the resolved slide backend and, when a source mask exists, the resolved mask backend; it does
not reject artifacts merely because a requested value differs. When `mask_backend: auto` selects
a backend, a distinct `mask_backend.selected` progress event names the sample, mask path,
resolved backend, and reason.

## Consequences

- A mask can use a different decoder than its slide without changing the slide backend.
- Because `auto` is openability-only, a backend that opens but cannot decode a mask can be
  selected and then fail at read time; the fix is to set `mask_backend` explicitly.
- Existing `auto` configurations can resolve to a different backend under the shared priority,
  so backend-dependent artifacts may need to be recomputed.
- Pre-#163 metadata and `process_list.csv` schemas (without the mask backend fields/columns)
  are rejected clearly rather than loaded.
