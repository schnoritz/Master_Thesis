import numpy as np
import os
import nibabel as nib
import torch
import argparse
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Choose your directory with patient data")

    parser.add_argument(
        "-dir",
        type=str,
    )

    parser.add_argument(
        "-segs",
        nargs='+',
        action='store',
        dest='segment_list'
    )

    parser.add_argument(
        "-num",
        type=int
    )

    args = parser.parse_args()

    if args.dir[-1] != "/":
        args.dir += "/"

    if args.num:
        segments = [x for x in os.listdir(args.dir) if not x.startswith(".")]

        segments = random.sample(segments, args.num)
    else:
        segments = args.segment_list

    for seg in segments:
        print(f"{args.dir}{seg}/training_data.pt")

        if os.path.isfile(f"{args.dir}{seg}/training_data.pt") and os.path.isfile(f"{args.dir}{seg}/target_data.pt"):
            masks = torch.load(f"{args.dir}{seg}/training_data.pt")
            target = torch.load(f"{args.dir}{seg}/target_data.pt")
        else:
            print(seg, "missing data!")
            continue

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))

        ax[0].imshow(masks[0, 256, :, :], cmap="bone")
        ax[0].imshow(target[0, 256, :, :], alpha=0.4, cmap="jet")

        ax[1].imshow(masks[0, :, 256, :], cmap="bone")
        ax[1].imshow(target[0, :, 256, :], alpha=0.4, cmap="jet")

        plt.savefig(
            f"/mnt/qb/baumgartner/sgutwein84/test/{seg}.png")
        print(f"{seg} created")
        plt.close(fig)
