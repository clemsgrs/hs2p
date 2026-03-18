# Clean Contract Simplification

## Goal

Remove the remaining duplicate runtime/config contracts so the codebase uses:

- resolver-owned config normalization
- `ResolvedSamplingSpec` as the only runtime sampling model
- `preview` vocabulary everywhere
- one owner for seeding
- one geometry object for tissue coverage checks

## Checklist

- [x] Add `hs2p.configs.resolvers` and route CLI entrypoints through it
- [x] Remove `SamplingParameters` and dual `sampling_params` execution paths
- [x] Rename preview/QC vocabulary to `PreviewConfig` / `preview` / `save_previews`
- [x] Make `setup()` the sole random-seed owner
- [x] Introduce `ResolvedTileGeometry` and use it in `HasEnoughTissue`
- [x] Update docs and targeted regression tests to the new contracts
