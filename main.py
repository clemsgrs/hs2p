import hydra
from pathlib import Path
from omegaconf import DictConfig

from utils import seg_and_patch


@hydra.main(version_base='1.2.0', config_path='config', config_name='default')
def main(cfg: DictConfig):

    output_dir = Path(cfg.output_dir, cfg.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_save_dir = Path(output_dir, 'patches')
    mask_save_dir = Path(output_dir, 'masks')
    stitch_save_dir = Path(output_dir, 'stitches')

    directories = {
        'data_dir': Path(cfg.data_dir, cfg.dataset_name, 'slides'),
        'output_dir': output_dir,
        'patch_save_dir': patch_save_dir,
        'mask_save_dir' : mask_save_dir,
        'stitch_save_dir': stitch_save_dir,
    }

    for dirname, dirpath in directories.items():
        if dirname not in ['data_dir']:
            dirpath = Path(dirpath)
            dirpath.mkdir(parents=False, exist_ok=True)

    slide_list = Path(cfg.slide_list)

    print()

    seg_times, patch_times = seg_and_patch(
        **directories,
        slide_list=slide_list,
        seg=cfg.flags.seg,
        patch=cfg.flags.patch,
        stitch=cfg.flags.stitch,
        auto_skip=cfg.flags.auto_skip,
        seg_params=cfg.seg_params,
        filter_params=cfg.filter_params,
        vis_params=cfg.vis_params,
        patch_params=cfg.patch_params,
        verbose=cfg.flags.verbose,
    )


if __name__ == '__main__':

    # python3 main.py --config-name 'panda'
    main()
