# Documentation

Start with the [installation and quick starts](../README.md) to tile your first slide, then choose the guide for your workflow.

| Guide | Use it to |
| --- | --- |
| [Python API](api.md) | Tile one slide or a batch, inspect results, and save or load artifacts. |
| [CLI](cli.md) | Configure batch tiling or annotation sampling, choose backends, and resume runs. |
| [Artifacts](artifacts.md) | Read coordinates, metadata, tile exports, and `process_list.csv`. |
| [Tissue-mask generation](tissue-mask-generation.md) | Create reusable pyramidal tissue masks with the standalone script. |
| [Benchmarks](benchmark.md) | Run throughput benchmarks and interpret the recorded results. |
| [Release notes](https://github.com/clemsgrs/hs2p/releases) | Check behavior changes and artifact compatibility before upgrading. |
| [Downstream contracts](downstream-contracts.md) | Check an hs2p change against the pinned slide2vec and soma contracts, and update the pins. |

Architecture decisions explain the mask-reading and API contracts:

- [One authoritative backend for mask decoding](adr/0001-authoritative-mask-decoding.md)
- [Independent slide and mask backends](adr/0002-independent-mask-backend.md)
- [Keyword-only constructors](adr/0003-keyword-only-constructors.md)
- [First-class `Mask` domain object](adr/0004-first-class-mask.md)
