import numpy as np

from hs2p.wsi.geometry import project_discrete_grid_origins


def test_project_discrete_grid_origins_truncates_top_left_anchors():
    coordinates = np.array(
        [
            [5, 5],
            [15, 15],
            [25, 25],
            [35, 45],
        ],
        dtype=np.int64,
    )

    projected = project_discrete_grid_origins(
        coordinates,
        scale_x=0.1,
        scale_y=0.1,
    )

    np.testing.assert_array_equal(
        projected,
        np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 4],
            ],
            dtype=np.int64,
        ),
    )


def test_project_discrete_grid_origins_supports_anisotropic_scales():
    coordinates = np.array([[10, 10], [31, 49]], dtype=np.int64)

    projected = project_discrete_grid_origins(
        coordinates,
        scale_x=0.25,
        scale_y=0.1,
    )

    np.testing.assert_array_equal(
        projected,
        np.array([[2, 1], [7, 4]], dtype=np.int64),
    )
