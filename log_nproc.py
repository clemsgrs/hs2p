import time
import wandb
from pathlib import Path

from source.utils import compute_time


def main(output_dir: Path, fmt: str = "jpg"):

    processed = [fp for fp in Path(output_dir).glob(f"*.{fmt}")]
    nproc = len(processed)
    return nproc


if __name__ == "__main__":

    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="whether logging to wandb should be enabled",
    )
    parser.add_argument(
        "--id",
        help="id of the corresponding main experiment",
    )
    parser.add_argument(
        "--project",
        help="project name",
    )
    parser.add_argument(
        "--username",
        help="user name",
    )
    parser.add_argument(
        "--output_dir",
        help="directory where main experiment output is saved",
        required=True,
    )
    parser.add_argument(
        "--fmt", help="file format of main experiment saved output", required=True
    )
    parser.add_argument(
        "--total", type=int, help="total number of output files expected", required=True
    )
    parser.add_argument(
        "--freq",
        type=int,
        help="time between two consecutive logging calls (in seconds)",
        default=5,
    )
    parser.add_argument(
        "--tol",
        type=int,
        help="if number of processed slides stops moving for longer than tol (in minutes), the process is killed",
        default=30,
    )
    args = vars(parser.parse_args())

    if args["log_to_wandb"]:
        run = wandb.init(id=args["id"], project=args["project"], entity=args["username"])
        run.define_metric("processed", summary="max")
    print()
    stop = False
    previous_nproc = 0
    start_time = time.time()
    while not stop:
        nproc = main(args["output_dir"], args["fmt"])
        if nproc > previous_nproc:
            start_time = time.time()
            if args["log_to_wandb"]:
                wandb.log({"processed": nproc})
            print(f'nslide processed: {nproc}/{args["total"]}', end="\r")
        else:
            end_time = time.time()
            mins, secs = compute_time(start_time, end_time)
            if mins > args["tol"]:
                sys.exit()
        time.sleep(args["freq"])
        stop = nproc == args["total"]
        previous_nproc = nproc

    sys.exit()
