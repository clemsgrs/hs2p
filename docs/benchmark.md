# Tile Store Benchmark

This note tells the story behind how I optimized the tile throughput when creating the tile store.

<img src="../output/tile-store-benchmark-path-breakdown/throughput.png" alt="Tile-store benchmark progression" width="900">

## Baseline

The starting point was the simplest possible path:

- ASAP / `wholeslidedata` reads
- one read per tile
- PIL JPEG encoding

On the benchmark slide, that baseline reached `62.64 tiles/s` at `4` workers.
The plot makes the main problem obvious: throughput rises a little with worker
count, but the curve flattens quickly because the pipeline is still doing one
decode per tile.

## WSD + Supertiles

The first read-path change was to stop thinking about the slide as a sequence
of isolated tiles and instead read larger regions that cover multiple output
tiles.

By “supertiles” I mean a larger read window, such as an `8x8` or `4x4` block of
output tiles, that is read once and then sliced back into the individual tiles.
That reduces repeated decode work because neighboring tiles often overlap the
same underlying image data.

Using WSD with supertiles raised throughput to `82.23 tiles/s` at `4` workers.
That is a real gain, but it is still a modest one. The important part is that
the curve improved even before we changed readers, which confirmed that the
larger-region idea itself was sound.

## CuCIM Batch Reads

In parallel, we tried a different path: keep the tile structure, but let CuCIM
batch multiple reads more efficiently.

Using CuCIM batch reads raised throughput to `85.60 tiles/s` at `4` workers.
That is close to the WSD supertile result, which is useful evidence: the win is
not tied to one particular reader. The common factor is that both approaches
reduce how often the pipeline has to decode a tiny tile-sized region from
scratch.

## Combining the Read Improvements

Both directions improved the baseline, so the next obvious step was to combine
them.

With `cucim + supertiles + PIL`, throughput rises to `193.23 tiles/s` at
`4` workers. That is more than 3x the original baseline.

At that point, the encoder finally matters again. Switching from PIL to
TurboJPEG on the same combined path lifts throughput further to
`217.56 tiles/s` at `4` workers.
