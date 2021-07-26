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
        "dir",
        type=str,
        metavar="",
        help="",
    )

    parser.add_argument(
        "num",
        type=int,
        metavar="",
        help="",
    )

    args = parser.parse_args()

    if args.dir[-1] != "/":
        args.dir += "/"

    segments = [x for x in os.listdir(args.dir) if not x.startswith(".")]
    segments = random.sample(segments, args.num)

    for seg in segments:
        masks = torch.load(f"{args.dir}{seg}/training_data.pt")
        target = torch.load(f"{args.dir}{seg}/target_data.pt")

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        ax[0].imshow(masks[0, 256, :, :])
        ax[0].imshow(target[0, 256, :, :], alpha=0.4, cmap="jet")

        ax[1].imshow(masks[0, :, 256, :])
        ax[1].imshow(target[0, :, 256, :], alpha=0.4, cmap="jet")

        plt.savefig(
            f"/home/baumgartner/sgutwein84/container/test/{seg}.png")
        print(f"{seg} created")
