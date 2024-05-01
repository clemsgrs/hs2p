import os
import tqdm
import wandb
import hydra
import datetime
import threading
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from omegaconf import DictConfig

from utils import initialize_wandb, log_progress, sample_patches, sample_patches_mp


@hydra.main(
    version_base="1.2.0", config_path="config/sampling", config_name="witali_liver"
)
def main(cfg: DictConfig):

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")
        run_id = wandb_run.id

    pixel_mapping = {k: v for e in cfg.pixel_mapping for k, v in e.items()}
    if cfg.color_mapping is not None:
        color_mapping = {k: v for e in cfg.color_mapping for k, v in e.items()}
    else:
        color_mapping = None

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(exist_ok=True, parents=True)

    seg_mask_save_dir = Path(output_dir, "segmentation_mask")
    overlay_mask_save_dir = Path(output_dir, "annotation_mask")
    seg_mask_save_dir.mkdir(exist_ok=True)
    overlay_mask_save_dir.mkdir(exist_ok=True)

    df = pd.read_csv(cfg.csv)

    slide_ids = df["slide_id"].values.tolist()
    slide_fps = df["slide_path"].values.tolist()
    seg_mask_fps = [None] * len(slide_fps)
    if "segmentation_mask_path" in df.columns:
        seg_mask_fps = [Path(f) for f in df["segmentation_mask_path"].values.tolist()]
    annot_mask_fps = df["annotation_mask_path"].values.tolist()

    spacings = [None] * len(slide_ids)
    if "spacing" in df.columns:
        spacings = df.spacing.values.tolist()

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

    if cfg.speed.multiprocessing:

        args = [
            (
                sid,
                Path(slide_fp),
                Path(annot_mask_fp),
                output_dir,
                pixel_mapping,
                cfg.visu,
                cfg.seg_params,
                cfg.vis_params,
                cfg.filter_params,
                cfg.patch_params,
                spacing,
                seg_mask_fp,
                1,
                color_mapping,
                cfg.filtering_threshold,
                cfg.skip_category,
                cfg.sort,
                cfg.topk,
                cfg.alpha,
                seg_mask_save_dir,
                overlay_mask_save_dir,
                cfg.backend,
            )
            for sid, slide_fp, seg_mask_fp, annot_mask_fp, spacing in zip(
                slide_ids, slide_fps, seg_mask_fps, annot_mask_fps, spacings
            )
        ]

        total = len(args)
        processed_count = mp.Value('i', 0)

        # start the logging thread
        if cfg.wandb.enable:
            stop_logging = threading.Event()
            logging_thread = threading.Thread(target=log_progress, args=(processed_count, stop_logging, total))
            logging_thread.start()

        dfs = []
        with mp.Pool(num_workers) as pool:
            for r in tqdm.tqdm(pool.imap_unordered(sample_patches_mp, args), desc="Patch sampling", unit=" slide", total=total):
                dfs.append(r)
                with processed_count.get_lock():
                    processed_count.value += 1

        if cfg.wandb.enable:
            stop_logging.set()
            logging_thread.join()
            wandb.log({"processed": processed_count.value})

        tile_df = pd.concat(dfs, ignore_index=True)
        tiles_fp = Path(output_dir, f"sampled_patches.csv")
        tile_df.to_csv(tiles_fp, index=False)

    else:

        dfs = []

        with tqdm.tqdm(
            zip(slide_ids, slide_fps, seg_mask_fps, annot_mask_fps, spacings),
            desc=f"Patche sampling",
            unit=" slide",
            initial=0,
            total=len(slide_ids),
            leave=True,
        ) as t:

            for i, (sid, slide_fp, seg_mask_fp, annot_mask_fp, spacing) in enumerate(t):

                t_df = sample_patches(
                    sid,
                    Path(slide_fp),
                    Path(annot_mask_fp),
                    output_dir,
                    pixel_mapping,
                    cfg.visu,
                    cfg.seg_params,
                    cfg.vis_params,
                    cfg.filter_params,
                    cfg.patch_params,
                    spacing=spacing,
                    seg_mask_fp=seg_mask_fp,
                    num_workers=num_workers,
                    color_mapping=color_mapping,
                    filtering_threshold=cfg.filtering_threshold,
                    skip=cfg.skip_category,
                    sort=cfg.sort,
                    topk=cfg.topk,
                    alpha=cfg.alpha,
                    seg_mask_save_dir=seg_mask_save_dir,
                    overlay_mask_save_dir=overlay_mask_save_dir,
                    backend=cfg.backend,
                )
                if t_df is not None:
                    dfs.append(t_df)

                if cfg.wandb.enable:
                    wandb.log({"processed": i + 1})

            df = pd.concat(dfs, ignore_index=True)
            tiles_fp = Path(output_dir, f"sampled_patches.csv")
            df.to_csv(tiles_fp, index=False)


if __name__ == "__main__":

    main()
