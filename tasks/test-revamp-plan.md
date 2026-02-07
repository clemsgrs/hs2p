# Test Revamp Plan

- [x] Create fresh test root under `test-revamped/`
- [x] Add helper modules for fake multires WSI/mask backend and parameter factories
- [x] Add deterministic unit tests for sorting and mask coverage utilities
- [x] Add level-selection and spacing-guard tests
- [x] Add deterministic multires tiling tests with exact coordinates/counts
- [x] Add sampling/filter tests with controlled multi-label masks
- [x] Add optional real-fixture smoke regression using only WSI+mask TIFF
- [x] Keep legacy `test/input/config.yaml` and `test/gt/test-wsi.npy` untouched but unused
- [x] Update pytest discovery and CI workflow to run `test-revamped`
- [x] Run revamped tests locally and fix failures
- [x] Document results in `docs/`
