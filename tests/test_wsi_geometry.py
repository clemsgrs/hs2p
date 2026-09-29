import numpy as np

from hs2p.wsi.geometry import project_discrete_grid_origins


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
