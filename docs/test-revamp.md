# Test Revamp Overview

A new test suite was added under `test-revamped/`.

Highlights:
- Uses deterministic in-memory mocked multires slide/mask pyramids for exact, geometry-driven assertions.
- Uses optional real-world smoke regression with TIFF fixtures only (`test/input/test-wsi.tif` and `test/input/test-mask.tif`).
- Does not depend on legacy `test/input/config.yaml` or `test/gt/test-wsi.npy`.

The legacy files remain in place for compatibility but are intentionally unused by revamped tests.
