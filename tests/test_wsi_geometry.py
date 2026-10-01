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


def test_resolve_tile_stride_defines_the_stride_once_at_the_read_level():
    from hs2p.wsi.geometry import resolve_tile_stride

    # 10% overlap on a 16 px read at a 2x level: 14.4 -> 14 at the read level, and the
    # level-0 stride follows (28), never an independent 28.8 -> 29
    stride = resolve_tile_stride(read_tile_size_px=16, tile_size_lv0=32, overlap=0.1)
    assert (stride.read_step_px, stride.step_px_lv0) == (14, 28)

    stride = resolve_tile_stride(read_tile_size_px=256, tile_size_lv0=512, overlap=0.1)
    assert (stride.read_step_px, stride.step_px_lv0) == (230, 460)

    # no overlap and level-0 reads are unchanged
    stride = resolve_tile_stride(read_tile_size_px=16, tile_size_lv0=32, overlap=0.0)
    assert (stride.read_step_px, stride.step_px_lv0) == (16, 32)
    stride = resolve_tile_stride(read_tile_size_px=224, tile_size_lv0=224, overlap=0.1)
    assert (stride.read_step_px, stride.step_px_lv0) == (202, 202)

    # a stride never collapses to zero
    stride = resolve_tile_stride(read_tile_size_px=4, tile_size_lv0=8, overlap=0.99)
    assert (stride.read_step_px, stride.step_px_lv0) == (1, 2)
