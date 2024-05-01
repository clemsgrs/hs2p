import os
import tqdm
import wandb
import hydra
import datetime
import threading
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from omegaconf import DictConfig

from source.utils import initialize_df
from utils import initialize_wandb, log_progress, seg_and_patch, seg_and_patch_slide_mp


@hydra.main(
    version_base="1.2.0", config_path="config/extraction", config_name="default"
)
def main(cfg: DictConfig):

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_save_dir = Path(output_dir, "masks")
    patch_save_dir = Path(
        output_dir,
        "patches",
        f"{cfg.patch_params.patch_size}",
        f"{cfg.patch_params.format}",
    )
    visu_save_dir = Path(output_dir, "visualization", f"{cfg.patch_params.patch_size}")

    directories = {
        "mask_save_dir": mask_save_dir,
        "patch_save_dir": patch_save_dir,
        "visu_save_dir": visu_save_dir,
    }

    for dirname, dirpath in directories.items():
        if not cfg.resume:
            dirpath.mkdir(parents=True)
        else:
            dirpath.mkdir(parents=True, exist_ok=True)

    if cfg.slide_csv.endswith(".txt"):
        with open(cfg.slide_csv, "r") as f:
            slide_paths = [x.strip() for x in f.readlines()]
        slide_df = pd.DataFrame.from_dict({"slide_path": slide_paths})
    else:
        slide_df = pd.read_csv(cfg.slide_csv)

    process_list_fp = None
    if Path(output_dir, "process_list.csv").is_file() and cfg.resume:
        process_list_fp = Path(output_dir, "process_list.csv")

    print()

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    if cfg.speed.multiprocessing:

        slide_paths = slide_df.slide_path.values.tolist()
        mask_paths = []
        if "segmentation_mask_path" in slide_df.columns:
            mask_paths = slide_df.segmentation_mask_path.values.tolist()
        spacings = []
        if "spacing" in slide_df.columns:
            spacings = slide_df.spacing.values.tolist()

        if process_list_fp is None:
            df = initialize_df(
                slide_paths,
                mask_paths,
                spacings,
                cfg.seg_params,
                cfg.filter_params,
                cfg.vis_params,
                cfg.patch_params,
            )
        else:
            df = pd.read_csv(process_list_fp)
            df = initialize_df(
                df, cfg.seg_params, cfg.filter_params, cfg.vis_params, cfg.patch_params
            )

        mask = df["process"] == 1
        process_stack = df[mask]

        slide_ids_to_process = process_stack.slide_id
        slide_paths_to_process = process_stack.slide_path

        mask_paths_to_process = [None] * len(slide_paths_to_process)
        if "segmentation_mask_path" in process_stack.columns:
            mask_paths_to_process = process_stack.segmentation_mask_path

        spacings_to_process = [None] * len(slide_paths_to_process)
        if "spacing" in process_stack.columns:
            spacings_to_process = process_stack.spacing

        args = [
            (
                patch_save_dir,
                mask_save_dir,
                visu_save_dir,
                cfg.seg_params,
                cfg.filter_params,
                cfg.vis_params,
                cfg.patch_params,
                sid,
                slide_fp,
                mask_fp,
                spacing,
                cfg.flags.patch,
                cfg.flags.visu,
                cfg.flags.verbose,
                cfg.backend,
            )
            for sid, slide_fp, mask_fp, spacing in zip(
                slide_ids_to_process,
                slide_paths_to_process,
                mask_paths_to_process,
                spacings_to_process,
            )
        ]

        total = len(args)
        processed_count = mp.Value("i", 0)

        # start the logging thread
        if cfg.wandb.enable:
            stop_logging = threading.Event()
            logging_thread = threading.Thread(
                target=log_progress, args=(processed_count, stop_logging, total)
            )
            logging_thread.start()

        results = []
        with mp.Pool(num_workers) as pool:
            for r in tqdm.tqdm(
                pool.imap_unordered(seg_and_patch_slide_mp, args),
                desc="Patch extraction",
                unit=" slide",
                total=total,
            ):
                results.append(r)
                if r[0] is not None:
                    with processed_count.get_lock():
                        processed_count.value += 1

        avg_process_time = round(
            np.mean(
                [r[-1] for r in results if (r[0] is not None and r[-1] is not None)]
            ),
            2,
        )
        min_process_time = round(
            np.min(
                [r[-1] for r in results if (r[0] is not None and r[-1] is not None)]
            ),
            2,
        )
        max_process_time = round(
            np.max(
                [r[-1] for r in results if (r[0] is not None and r[-1] is not None)]
            ),
            2,
        )
        if cfg.wandb.enable:
            stop_logging.set()
            logging_thread.join()
            wandb.log({"processed": processed_count.value})
            wandb.log({"avg_time_sec": avg_process_time})
            wandb.log({"min_time_sec": min_process_time})
            wandb.log({"max_time_sec": max_process_time})

        dfs = []
        for t_df, sid, s, vl, sl, pt in results:

            mask = df["slide_id"] == sid
            df.loc[mask, "status"] = s
            df.loc[mask, "process"] = 0
            df.loc[mask, "vis_level"] = vl
            df.loc[mask, "seg_level"] = sl
            df.loc[mask, "process_time"] = pt

            dfs.append(t_df)

        df.to_csv(Path(output_dir, "process_list.csv"), index=False)

        tile_df = pd.concat(dfs, ignore_index=True)
        tile_df.to_csv(Path(output_dir, "tiles.csv"), index=False)

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
            num_workers=num_workers,
            verbose=cfg.flags.verbose,
            log_to_wandb=cfg.wandb.enable,
            backend=cfg.backend,
        )


if __name__ == "__main__":

    main()
