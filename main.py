import os
import hydra
import shutil
import datetime
import subprocess
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from omegaconf import DictConfig

from source.utils import initialize_df
from utils import initialize_wandb, seg_and_patch, seg_and_patch_slide


@hydra.main(version_base="1.2.0", config_path="config", config_name="default")
def main(cfg: DictConfig):

    run_id = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_save_dir = Path(output_dir, "masks")
    patch_save_dir = Path(output_dir, "patches", f"{cfg.patch_params.patch_size}", f"{cfg.patch_params.format}")
    visu_save_dir = Path(output_dir, "visualization", f"{cfg.patch_params.patch_size}")

    directories = {
        "mask_save_dir": mask_save_dir,
        "patch_save_dir": patch_save_dir,
        "visu_save_dir": visu_save_dir,
    }

    for dirname, dirpath in directories.items():
        if not cfg.resume:
            if dirpath.exists():
                shutil.rmtree(dirpath)
            dirpath.mkdir(parents=True)
        else:
            dirpath.mkdir(parents=True, exist_ok=True)

    if cfg.slide_csv.endswith('.txt'):
        with open(cfg.slide_csv, 'r') as f:
            slide_paths = [x.strip() for x in f.readlines()]
        slide_df = pd.DataFrame.from_dict({'slide_path': slide_paths})
    else:
        slide_df = pd.read_csv(cfg.slide_csv)

    process_list_fp = None
    if Path(output_dir, "process_list.csv").is_file() and cfg.resume:
        process_list_fp = Path(output_dir, "process_list.csv")

    print()

    if cfg.speed.multiprocessing:

        slide_paths = slide_df.slide_path.values.tolist()
        mask_paths = []
        if "mask_path" in slide_df.columns:
            mask_paths = slide_df.mask_path.values.tolist()

        if process_list_fp is None:
            df = initialize_df(
                slide_paths, mask_paths, cfg.seg_params, cfg.filter_params, cfg.vis_params, cfg.patch_params
            )
        else:
            df = pd.read_csv(process_list_fp)
            df = initialize_df(df, cfg.seg_params, cfg.filter_params, cfg.vis_params, cfg.patch_params)

        mask = df["process"] == 1
        process_stack = df[mask]
        slide_paths_to_process = process_stack.slide_path
        slide_ids_to_process = process_stack.slide_id
        args = [
            (
                df,
                output_dir,
                patch_save_dir,
                mask_save_dir,
                visu_save_dir,
                cfg.seg_params,
                cfg.filter_params,
                cfg.vis_params,
                cfg.patch_params,
                Path(fp),
                sid,
                cfg.flags.seg,
                cfg.flags.patch,
                cfg.flags.visu,
                cfg.flags.verbose,
            ) for fp, sid in zip(slide_paths_to_process, slide_ids_to_process)
        ]

        if cfg.wandb.enable:
            subprocess.Popen(["python3", "log_nproc.py", "--id", f"{run_id}", "--output_dir", f"{visu_save_dir}", "--fmt", "jpg", "--total", f"{len(process_stack)}"])

        num_workers = mp.cpu_count()
        if num_workers > 10:
            num_workers = 10
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(seg_and_patch_slide, args)

    else:

        seg_times, patch_times = seg_and_patch(
            output_dir,
            patch_save_dir,
            mask_save_dir,
            visu_save_dir,
            slide_df=slide_df,
            patch=cfg.flags.patch,
            visu=cfg.flags.visu,
            process_list=process_list_fp,
            seg_params=cfg.seg_params,
            filter_params=cfg.filter_params,
            vis_params=cfg.vis_params,
            patch_params=cfg.patch_params,
            verbose=cfg.flags.verbose,
            log_to_wandb=cfg.wandb.enable,
        )


if __name__ == "__main__":

    main()
