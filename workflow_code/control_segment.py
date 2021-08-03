import argparse
import torch
import nibabel as nib
import numpy as np


def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-path",
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
        "-save",
        action="store",
        dest='save_path'
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    for seg in args.segment_list:

        masks = np.array(torch.load(f"{args.segment_path}/{seg}/training_data.pt"))
        target = np.array(torch.load(f"{args.segment_path}/{seg}/target_data.pt"))
        target = target.squeeze()

        mask_names = ["binary", "ct", "radio_depth", "center", "source"]

        for num, name in enumerate(mask_names):
            dat = nib.Nifti1Image(masks[num], np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{args.save_path}/{seg}_{name}.nii.gz")

        dat = nib.Nifti1Image(target, np.eye(4))
        dat.header.get_xyzt_units()
        dat.to_filename(f"{args.save_path}/{seg}_target.nii.gz")
