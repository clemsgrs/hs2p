import os
import hydra
import shutil
from pathlib import Path
from omegaconf import DictConfig

from utils import initialize_wandb, seg_and_patch


@hydra.main(version_base="1.2.0", config_path="config", config_name="default")
def main(cfg: DictConfig):

    # set up wandb
    if cfg.wandb.username:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")

    output_dir = Path(cfg.output_dir, cfg.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_save_dir = Path(output_dir, "patches")
    mask_save_dir = Path(output_dir, "masks")
    visu_save_dir = Path(output_dir, "visualization")

    directories = {
        "output_dir": output_dir,
        "patch_save_dir": patch_save_dir,
        "mask_save_dir": mask_save_dir,
        "visu_save_dir": visu_save_dir,
    }

    for dirpath in directories.values():
        if not cfg.resume:
            if dirpath.exists():
                shutil.rmtree(dirpath)
            dirpath.mkdir(parents=False)
        else:
            dirpath.mkdir(parents=False, exist_ok=True)

    slide_list = Path(cfg.slide_list)

    process_list_fp = None
    if Path(output_dir, "process_list.csv").is_file() and cfg.resume:
        process_list_fp = Path(output_dir, "process_list.csv")

    print()

    seg_times, patch_times = seg_and_patch(
        output_dir,
        patch_save_dir,
        mask_save_dir,
        visu_save_dir,
        slide_list=slide_list,
        seg=cfg.flags.seg,
        patch=cfg.flags.patch,
        visu=cfg.flags.visu,
        process_list=process_list_fp,
        seg_params=cfg.seg_params,
        filter_params=cfg.filter_params,
        vis_params=cfg.vis_params,
        patch_params=cfg.patch_params,
        verbose=cfg.flags.verbose,
    )


if __name__ == "__main__":

    main()
