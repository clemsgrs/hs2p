import time
import wandb
from pathlib import Path


def main(output_dir: Path, fmt: str = 'jpg'):

    processed = [fp for fp in Path(output_dir).glob(f"*.{fmt}")]
    nproc = len(processed)
    return nproc


if __name__ == "__main__":

    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help='id of the corresponding main experiment', required=True)
    parser.add_argument('--output_dir', help='directory where main experiment output is saved', required=True)
    parser.add_argument('--fmt', help='file format of main experiment saved output', required=True)
    parser.add_argument('--total', type=int, help='total number of output files expected', required=True)
    parser.add_argument('--freq', type=int, help='time between two consecutive logging calls (in seconds)', default=5)
    args = vars(parser.parse_args())

    run = wandb.init(id=args["id"])
    run.define_metric("processed", summary="max")
    print()
    stop = False
    previous_nproc = 0
    while not stop:
        nproc = main(args["output_dir"], args["fmt"])
        if nproc > previous_nproc:
            wandb.log({"processed": nproc+1})
            print(f'nslide processed: {nproc}/{args["total"]}', end="\r")
        time.sleep(args["freq"])
        stop = (nproc == args["total"])
        previous_nproc = nproc

    sys.exit()