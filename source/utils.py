import cv2
import time
import h5py
import pyvips
import numpy as np
import pandas as pd

from PIL import Image, ImageOps
from pathlib import Path


def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_mode(bands):
    bands_to_mode = {1: "L", 3: "RGB", 4: "RGBA"}
    return bands_to_mode[bands]


def initialize_df(
    slides,
    masks,
    spacings,
    seg_params,
    filter_params,
    vis_params,
    patch_params,
    use_heatmap_args=False,
):
    """
    initiate a pandas df describing a list of slides to process
    args:
            slides (df or list): list of slide filepath
                    if df, these paths assumed to be stored under the 'slide_path' column
            masks (list): list of slides' segmentation masks filepath
            spacings (list): list of slides' spacing at level 0
            seg_params (dict): segmentation paramters
            filter_params (dict): filter parameters
            vis_params (dict): visualization paramters
            patch_params (dict): patching paramters
            use_heatmap_args (bool): whether to include heatmap arguments such as ROI coordinates
    """
    total = len(slides)
    if isinstance(slides, pd.DataFrame):
        slide_ids = list(slides.slide_id.values)
        slide_paths = list(slides.slide_path.values)
        if "mask_path" in slides.columns:
            mask_paths = list(slides.mask_path.values)
        if "spacing" in slides.columns:
            slide_spacings = list(slides.spacing.values)
    else:
        slide_ids = [Path(s).stem for s in slides]
        slide_paths = slides.copy()
        mask_paths = masks.copy()
        slide_spacings = spacings.copy()
    default_df_dict = {
        "slide_id": slide_ids,
        "slide_path": slide_paths,
    }
    if len(mask_paths) > 0:
        default_df_dict.update({"mask_path": mask_paths})
    if len(slide_spacings) > 0:
        default_df_dict.update({"spacing": slide_spacings})

    # initiate empty labels in case not provided
    if use_heatmap_args:
        default_df_dict.update({"label": np.full((total), -1)})

    default_df_dict.update(
        {
            "process": np.full((total), 1, dtype=np.uint8),
            "status": np.full((total), "tbp"),
            "has_patches": np.full((total), "tbd"),
            # seg params
            "seg_level": np.full((total), int(seg_params["seg_level"]), dtype=np.int8),
            "sthresh": np.full((total), int(seg_params["sthresh"]), dtype=np.uint8),
            "mthresh": np.full((total), int(seg_params["mthresh"]), dtype=np.uint8),
            "close": np.full((total), int(seg_params["close"]), dtype=np.uint32),
            "use_otsu": np.full((total), bool(seg_params["use_otsu"]), dtype=bool),
            # filter params
            "a_t": np.full((total), int(filter_params["a_t"]), dtype=np.float32),
            "a_h": np.full((total), int(filter_params["a_h"]), dtype=np.float32),
            "max_n_holes": np.full(
                (total), int(filter_params["max_n_holes"]), dtype=np.uint32
            ),
            # vis params
            "vis_level": np.full((total), int(vis_params["vis_level"]), dtype=np.int8),
            "line_thickness": np.full(
                (total), int(vis_params["line_thickness"]), dtype=np.uint32
            ),
            # patching params
            "use_padding": np.full(
                (total), bool(patch_params["use_padding"]), dtype=bool
            ),
            "contour_fn": np.full((total), patch_params["contour_fn"]),
            "tissue_thresh": np.full((total), patch_params["tissue_thresh"]),
        }
    )

    if use_heatmap_args:
        # initiate empty x,y coordinates in case not provided
        default_df_dict.update(
            {
                "x1": np.empty((total)).fill(np.NaN),
                "x2": np.empty((total)).fill(np.NaN),
                "y1": np.empty((total)).fill(np.NaN),
                "y2": np.empty((total)).fill(np.NaN),
            }
        )

    if isinstance(slides, pd.DataFrame):
        temp_copy = pd.DataFrame(
            default_df_dict
        )  # temporary dataframe w/ default params
        # find key in provided df
        # if exist, fill empty fields w/ default values, else, insert the default values as a new column
        for key in default_df_dict.keys():
            if key in slides.columns:
                mask = slides[key].isna()
                slides.loc[mask, key] = temp_copy.loc[mask, key]
            else:
                slides.insert(len(slides.columns), key, default_df_dict[key])
    else:
        slides = pd.DataFrame(default_df_dict)

    return slides


def save_hdf5(output_path, asset_dict, attr_dict=None, mode="a"):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(
                key,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type,
            )
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0] :] = val
    file.close()
    return output_path


def save_patch(
    wsi,
    scale,
    save_dir,
    asset_dict,
    attr_dict=None,
    fmt="png",
):
    coords = asset_dict["coords"]
    patch_size = attr_dict["coords"]["patch_size"]

    npatch = len(coords)
    region = pyvips.Region.new(wsi)

    # given we use .fetch, we need to move coordinates from slide level 0 to patch_level
    downscaled_coords = np.ceil((np.array(coords) / np.array(scale))).astype(np.int32)

    start_time = time.time()

    for i, coord in enumerate(coords):

        ps_x, ps_y = min(patch_size, wsi.width - downscaled_coords[i][0]), min(
            patch_size, wsi.height - downscaled_coords[i][1]
        )
        r = region.fetch(downscaled_coords[i][0], downscaled_coords[i][1], ps_x, ps_y)
        mode = get_mode(wsi.bands)
        patch = Image.frombuffer(mode=mode, size=(ps_x, ps_y), data=r).convert("RGB")
        if ps_x != patch_size or ps_y != patch_size:
            patch = ImageOps.pad(
                patch, (patch_size, patch_size), color=None, centering=(0, 0)
            )
        # when saving patch, keep level 0 as (x,y) reference
        save_path = Path(save_dir, f"{coord[0]}_{coord[1]}.{fmt}")
        patch.save(save_path)

    end_time = time.time()
    patch_saving_mins, patch_saving_secs = compute_time(start_time, end_time)
    return npatch, patch_saving_mins, patch_saving_secs


def initialize_hdf5_bag(first_patch, save_coord=False):
    (
        x,
        y,
        cont_idx,
        patch_size,
        patch_level,
        downsample,
        downsampled_level_dim,
        level_dim,
        img_patch,
        name,
        save_path,
    ) = tuple(first_patch.values())
    file_path = Path(save_path, f"{name}.h5")
    file = h5py.File(file_path, "w")
    img_patch = np.array(img_patch)[np.newaxis, ...]
    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[
        1:
    ]  # maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset(
        "imgs", shape=img_shape, maxshape=maxshape, chunks=img_shape, dtype=dtype
    )

    dset[:] = img_patch
    dset.attrs["patch_size"] = patch_size
    dset.attrs["patch_level"] = patch_level
    dset.attrs["wsi_name"] = name
    dset.attrs["downsample"] = downsample
    dset.attrs["level_dim"] = level_dim
    dset.attrs["downsampled_level_dim"] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset(
            "coords", shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32
        )
        coord_dset[:] = (x, y)

    file.close()
    return file_path


def DrawGrid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(
        img,
        tuple(np.maximum([0, 0], coord - thickness // 2)),
        tuple(coord - thickness // 2 + np.array(shape)),
        (0, 0, 0, 255),
        thickness=thickness,
    )
    return img


def DrawMapFromCoords(
    canvas,
    wsi_object,
    coords,
    patch_size,
    vis_level,
    indices=None,
    draw_grid=True,
    verbose=False,
):

    downsamples = wsi_object.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)

    downscaled_patch_size = tuple(
        np.ceil((np.array(patch_size) / np.array(downsamples))).astype(np.int32)
    )
    if verbose:
        print(f"downscaled patch size: {downscaled_patch_size}")

    # if we use pyvips Region fetching, we need to scale coordinates from level 0 to to vis_level
    downscaled_coords = np.ceil((coords / np.array(downsamples))).astype(np.int32)

    for idx in range(total):

        patch_id = indices[idx]
        coord = downscaled_coords[patch_id]
        wsi = wsi_object.open_page(vis_level)
        ps_x, ps_y = min(downscaled_patch_size[0], wsi.width - coord[0]), min(
            downscaled_patch_size[1], wsi.height - coord[1]
        )
        region = pyvips.Region.new(wsi).fetch(coord[0], coord[1], ps_x, ps_y)
        mode = get_mode(wsi.bands)
        patch = Image.frombuffer(mode=mode, size=(ps_x, ps_y), data=region).convert(
            "RGB"
        )
        if ps_x != downscaled_patch_size[0] or ps_y != downscaled_patch_size[1]:
            patch = ImageOps.pad(
                patch, downscaled_patch_size, color=None, centering=(0, 0)
            )
        patch = np.array(patch)
        canvas_crop_shape = canvas[
            coord[1] : coord[1] + downscaled_patch_size[1],
            coord[0] : coord[0] + downscaled_patch_size[0],
            :3,
        ].shape[:2]
        canvas[
            coord[1] : coord[1] + downscaled_patch_size[1],
            coord[0] : coord[0] + downscaled_patch_size[0],
            :3,
        ] = patch[: canvas_crop_shape[0], : canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, downscaled_patch_size)

    return Image.fromarray(canvas)


def VisualizeCoords(
    hdf5_file_path,
    wsi_object,
    downscale=16,
    draw_grid=False,
    bg_color=(0, 0, 0),
    alpha=-1,
    verbose=False,
):
    vis_level = wsi_object.get_best_level_for_downsample(downscale)
    file = h5py.File(hdf5_file_path, "r")
    dset = file["coords"]
    coords = dset[:]
    w, h = wsi_object.level_dimensions[0]

    if verbose:
        print(f"original size: {w} x {h}")

    w, h = wsi_object.level_dimensions[vis_level]

    patch_size = dset.attrs["patch_size"]
    patch_level = dset.attrs["patch_level"]
    if verbose:
        print(f"downscaled size for stiching: {w} x {h}")
        print(f"number of patches: {len(coords)}")
        print(f"patch size: {patch_size}")
        print(f"patch level: {patch_level}")

    patch_size = tuple(
        (
            np.array((patch_size, patch_size))
            * wsi_object.level_downsamples[patch_level]
        ).astype(np.int32)
    )
    if verbose:
        print(f"ref patch size: {patch_size}")

    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError(
            "Visualization Downscale %d is too large" % downscale
        )

    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(
            size=(w, h), mode="RGBA", color=bg_color + (int(255 * alpha),)
        )

    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(
        heatmap,
        wsi_object,
        coords,
        patch_size,
        vis_level,
        indices=None,
        draw_grid=draw_grid,
        verbose=verbose,
    )

    file.close()
    return heatmap
