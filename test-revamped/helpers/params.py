from __future__ import annotations

from types import SimpleNamespace


def make_segment_params(
    downsample: int = 2,
    sthresh: int = 8,
    sthresh_up: int = 255,
    mthresh: int = 3,
    close: int = 0,
    use_otsu: bool = False,
    use_hsv: bool = False,
):
    return SimpleNamespace(
        downsample=downsample,
        sthresh=sthresh,
        sthresh_up=sthresh_up,
        mthresh=mthresh,
        close=close,
        use_otsu=use_otsu,
        use_hsv=use_hsv,
    )


def make_filter_params(
    ref_tile_size: int = 4,
    a_t: int = 0,
    a_h: int = 0,
    max_n_holes: int = 8,
    filter_white: bool = False,
    filter_black: bool = False,
    white_threshold: int = 220,
    black_threshold: int = 25,
    fraction_threshold: float = 0.9,
):
    return SimpleNamespace(
        ref_tile_size=ref_tile_size,
        a_t=a_t,
        a_h=a_h,
        max_n_holes=max_n_holes,
        filter_white=filter_white,
        filter_black=filter_black,
        white_threshold=white_threshold,
        black_threshold=black_threshold,
        fraction_threshold=fraction_threshold,
    )


def make_tiling_params(
    spacing: float = 1.0,
    tolerance: float = 0.01,
    tile_size: int = 8,
    overlap: float = 0.0,
    min_tissue_percentage: float = 0.0,
    drop_holes: bool = False,
    use_padding: bool = False,
):
    return SimpleNamespace(
        spacing=spacing,
        tolerance=tolerance,
        tile_size=tile_size,
        overlap=overlap,
        min_tissue_percentage=min_tissue_percentage,
        drop_holes=drop_holes,
        use_padding=use_padding,
    )


def make_sampling_params(pixel_mapping: dict[str, int], tissue_percentage: dict[str, float | None]):
    color_mapping = {k: None for k in pixel_mapping}
    return SimpleNamespace(
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        tissue_percentage=tissue_percentage,
    )
