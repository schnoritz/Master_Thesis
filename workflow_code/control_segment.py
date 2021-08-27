import argparse
import torch
import nibabel as nib
import numpy as np
import os
import random
import matplotlib.pyplot as plt


def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dir",
        required=True,
        dest="segment_path"
    )

    parser.add_argument(
        "-segs",
        nargs='+',
        action='store',
        dest='segment_list'
    )

    parser.add_argument(
        "-r",
        type=int,
        action='store',
        dest='random'
    )

    parser.add_argument(
        "-save",
        action="store",
        required=True,
        dest='save_path'
    )

    return parser.parse_args()


def get_segment(segment_path, seg, save_path):

    if os.path.isfile(f"{segment_path}/{seg}/training_data.pt"):
        masks = np.array(torch.load(f"{segment_path}/{seg}/training_data.pt"))

        target = np.array(torch.load(f"{segment_path}/{seg}/target_data.pt"))
        target = target.squeeze()

        mask_names = ["binary", "ct", "radio_depth", "center", "source"]

        for num, name in enumerate(mask_names):

            dat = nib.Nifti1Image(masks[num], np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{seg}_{name}.nii.gz")

        dat = nib.Nifti1Image(target, np.eye(4))
        dat.header.get_xyzt_units()
        dat.to_filename(f"{save_path}/{seg}_target.nii.gz")

    else:
        print("Missing data for", seg)


if __name__ == "__main__":

    args = parse()

    if not args.random:

        for seg in args.segment_list:

            get_segment(args.segment_path, seg, args.save_path)
            print("Finished ", seg)

    else:

        segments = [x for x in os.listdir(args.segment_path) if not x.startswith(".")]
        segments = random.sample(segments, args.random)

        for seg in segments:

            get_segment(args.segment_path, seg, args.save_path)
            print("Finished ", seg)
