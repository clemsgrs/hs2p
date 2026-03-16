from types import SimpleNamespace

import numpy as np

import hs2p.wsi.wsi as wsimod


def test_process_contours_concatenates_numpy_outputs_without_list_roundtrip():
    class DummyWSI:
        def _resolve_tile_read_metadata(self, tiling_params):
            del tiling_params
            return 1, 1.0, 2.0

        def process_contour(
            self, contour, contour_holes, tiling_params, filter_params, annotation
        ):
            del contour_holes, tiling_params, filter_params, annotation
            if contour == "empty":
                return (
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.float32),
                    1,
                    2.0,
                )
            if contour == "first":
                return (
                    np.array([10, 30], dtype=np.int64),
                    np.array([20, 40], dtype=np.int64),
                    np.array([0.25, 0.75], dtype=np.float32),
                    1,
                    2.0,
                )
            return (
                np.array([50], dtype=np.int64),
                np.array([60], dtype=np.int64),
                np.array([0.5], dtype=np.float32),
                1,
                2.0,
            )

    x_coords, y_coords, tissue_pct, contour_indices, tile_level, resize_factor = (
        wsimod.WholeSlideImage.process_contours(
            DummyWSI(),
            contours=["first", "empty", "second"],
            holes=[[], [], []],
            tiling_params=SimpleNamespace(),
            filter_params=SimpleNamespace(),
            annotation=None,
            disable_tqdm=True,
            num_workers=1,
        )
    )

    assert isinstance(x_coords, np.ndarray)
    assert isinstance(y_coords, np.ndarray)
    assert isinstance(tissue_pct, np.ndarray)
    assert isinstance(contour_indices, np.ndarray)
    np.testing.assert_array_equal(x_coords, np.array([10, 30, 50], dtype=np.int64))
    np.testing.assert_array_equal(y_coords, np.array([20, 40, 60], dtype=np.int64))
    np.testing.assert_allclose(
        tissue_pct, np.array([0.25, 0.75, 0.5], dtype=np.float32)
    )
    np.testing.assert_array_equal(contour_indices, np.array([0, 0, 2], dtype=np.int32))
    assert tile_level == 1
    assert resize_factor == 2.0
