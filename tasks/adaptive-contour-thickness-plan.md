# Adaptive Contour Thickness Plan

- [x] Add failing tests for adaptive contour thickness behavior in `tests/test_visualize_mask_thickness.py`
- [x] Implement automatic contour thickness inference in `hs2p/wsi/wsi.py`
- [x] Remove manual `line_thickness` from `WholeSlideImage.visualize_mask` public API
- [x] Update visualization documentation in `README.md`
- [x] Run focused and relevant regression tests
- [x] Mark plan items complete with outcomes

## Outcomes

- Added comprehensive thickness tests, including API breakage test for `line_thickness`.
- Contour thickness is now always inferred from whole-slide size and visualization level.
- Verified fixture calibration target remains at thickness `15` for `test-wsi.tif` at the current visualization scale.
- Verified visualization output generation path with fixture files (`/tmp/hs2p-mask-visu-auto-thickness.jpg`).
