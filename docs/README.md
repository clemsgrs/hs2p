# Documentation

- [API guide](api.md)
  - Python entrypoints, public dataclasses, and when to use `tile_slide()` vs `tile_slides()`
- [CLI guide](cli.md)
  - Input CSV schemas, config areas, progress reporting, and resume/precomputed workflows
- [Artifact reference](artifacts.md)
  - `process_list.csv` manifests
- [Release notes](release-notes.md)
  - User-visible behavior changes and artifact compatibility impact
- [Benchmark notes](benchmark.md)
  - Throughput findings and the benchmark entrypoints in `scripts/`
- [Tissue mask generation](tissue-mask-generation.md)
  - Standalone pyramidal tissue-mask generation outside the main tiling pipeline
- Architecture decisions
  - [ADR 0001: Use one authoritative backend for mask decoding](adr/0001-authoritative-mask-decoding.md)
  - [ADR 0002: Resolve the mask backend independently from the slide backend](adr/0002-independent-mask-backend.md)
