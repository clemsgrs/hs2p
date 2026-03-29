from types import SimpleNamespace

import numpy as np

import hs2p.wsi.tiling as tiling_mod


def test_process_contours_concatenates_numpy_outputs_without_list_roundtrip(monkeypatch):
    def _fake_process_contour(
        *,
        reader,
        annotation_mask,
        annotation_pct,
        level_spacings,
        level_dimensions,
        level_downsamples,
        seg_level,
        contour,
        contour_holes,
        tiling_params,
        filter_params,
        annotation=None,
    ):
        del (
            reader,
            annotation_mask,
            annotation_pct,
            level_spacings,
            level_dimensions,
            level_downsamples,
            seg_level,
            contour_holes,
            tiling_params,
            filter_params,
            annotation,
        )
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

    monkeypatch.setattr(tiling_mod, "process_contour", _fake_process_contour)

    x_coords, y_coords, tissue_pct, contour_indices, tile_level, resize_factor = (
        tiling_mod.process_contours(
            reader=object(),
            annotation_mask={"tissue": np.ones((1, 1), dtype=np.uint8)},
            annotation_pct=None,
            level_spacings=[0.5],
            level_dimensions=[(1, 1)],
            level_downsamples=[(1.0, 1.0)],
            seg_level=0,
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
