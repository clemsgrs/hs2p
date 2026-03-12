# Test Suite Structure

## Default fast suite

The default `pytest` run is the fast, backend-free regression lane for the Python-first codebase. It is centered on:

- `tests/test_tiling_api.py` for the public tiling API contract
- `tests/test_cli_smoke.py` for current-schema CLI smoke coverage
- mocked/internal algorithm tests that exercise tiling math and mask semantics without real WSI backends

`pyproject.toml` marks the default run with `-m "not integration"`, so routine local runs and CI focus on stable API and CLI coverage.

## Optional integration suite

Backend-dependent fixture tests are marked `@pytest.mark.integration` and excluded from the default run. These cover:

- `tests/test_legacy_coordinates_regression.py`
- `tests/test_real_fixture_smoke_regression.py`

They should run only in environments that provide the required WSI backend plus the real fixture assets.

## Fixture refresh script

`scripts/generate_test_fixture_tiling_artifacts.py` generates current-format tiling artifacts for `tests/fixtures/input/test-wsi.tif` and `tests/fixtures/input/test-mask.tif` using the public tiling API and the same fixture config used by the integration regression.

The default output directory is `tests/fixtures/gt-current/`, and the script writes:

- `{sample_id}.tiles.npz`
- `{sample_id}.tiles.meta.json`

This is intended to support the eventual transition away from the legacy `.npy` golden by giving the integration suite a readable, current-schema artifact source.

## Current guidance

- Prefer testing documented public APIs and supported CLI schemas over deprecated internal calling patterns.
- Keep low-level `hs2p.wsi` tests only when they protect unique algorithmic behavior that the API-level tests do not cover.
- Avoid adding new tests that depend on tuple-unpacking `CoordinateExtractionResult`, legacy CSV schemas, or legacy `.npy` output shapes as public contracts.
