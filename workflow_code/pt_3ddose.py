import numpy as np
import cv2
from cv2 import resize
from pydicom import dcmread, uid
import torch
import os

import matplotlib.pyplot as plt
import matplotlib
from pt_ct import convert_ct_array


def dose_to_pt(dose_path, ct_path, tensor=False):
    """creates a 3D PyTorch Tensor of shape (512, 512, num_slices)

    Args:
        dose_path (int, pathlile_object): path to the .3ddose file that needs to be converted

    Returns:
        torch.tensor: tensor of shape (512, 512, num_slices)
    """
    if ct_path[-1] != "/":
        ct_path += "/"

    cts = [ct_path + x for x in os.listdir(ct_path)
           if not x.startswith(".") and "dcm" in x.lower()]

    dose, dose_limits = read_dose(dose_path)
    ct_limits = read_ct(cts)

    diff = ct_limits-dose_limits
    dose = upscale(dose, target_size=(
        512, 512), limits=diff)

    if tensor:
        return torch.tensor(dose)
    else:
        return dose


def read_ct(path):

    with dcmread(sorted(path)[0], force=True) as fin:

        slice_spacing = int(fin.SliceThickness)
        pixel_spacing = fin.PixelSpacing

        px_sp = np.array([pixel_spacing[0], pixel_spacing[1], slice_spacing]).astype(float)

        start = np.array(fin.ImagePositionPatient).astype(float)
        end = np.array(fin.ImagePositionPatient).astype(float)+px_sp[0]*512
        end[-1] = np.array(fin.ImagePositionPatient).astype(float)[-1] + \
            px_sp[2]*len(path)

        limits = np.zeros((6,))
        limits[::2] = start
        limits[1::2] = end

        return limits/10


def read_dose(filepath):
    """reads in a .3ddose file

    Args:
        filepath (str or pathlike object): filepath to the .3ddose file

    Returns:
        np.array: numpy array with the dose
    """

    with open(filepath, 'r') as fout:

        dim = np.array(fout.readline().split()).astype("int")

        dose_limits = []
        for _ in range(3):
            dat = fout.readline().split()
            dose_limits.extend([float(dat[0]), float(dat[-1])])

        dose_volume = np.array(fout.readline().split()).astype("float")
        dose_volume = dose_volume.reshape(dim[0], dim[1], dim[2], order='F')
        dose_volume = dose_volume.transpose((1, 0, 2))

    return dose_volume, dose_limits


def upscale(dose, target_size, limits):
    """upscales a dose distribution to a specified targetsize using nearest neighbor interpolation

    Args:
        dose (np.array): dose numpy array
        target_size (tuple or list): dimensions of the 3d array

    Returns:
        np.array: upscaled numpy array
    """

    limits = abs(np.round(limits/0.3)).astype(int)
    top = np.zeros((limits[2], dose.shape[1], dose.shape[2]))
    bottom = np.zeros((limits[3], dose.shape[1], dose.shape[2]))

    dose = np.concatenate((top, dose, bottom))

    left = np.zeros((dose.shape[0], limits[0], dose.shape[2]))
    right = np.zeros((dose.shape[0], limits[1], dose.shape[2]))

    dose = np.concatenate((left, dose, right), axis=1)

    final_dose = resize(
        dose, target_size, interpolation=cv2.INTER_LINEAR_EXACT)

    return final_dose


if __name__ == "__main__":

    files = ["p0", "p4", "p12", "p36", "p21", "p23", "p17"]
    segments = []
    for i in files:
        segments.append({
            "patient": i,
            "segment": i + "_8"

        })
        segments.append({
            "patient": i,
            "segment": i + "_12"

        })
        segments.append({
            "patient": i,
            "segment": i + "_19"

        })

    for segment in segments:

        ct_path = f"/home/baumgartner/sgutwein84/container/output_prostate/ct/{segment['patient']}"
        ct = convert_ct_array(ct_path, target_size=None, tensor=False)

        dose = dose_to_pt(
            f"/home/baumgartner/sgutwein84/container/output_prostate/{segment['segment']}/{segment['segment']}_1E07.3ddose",
            f"/home/baumgartner/sgutwein84/container/output_prostate/ct/{segment['patient']}")

        plt.imshow(dose[:, :, 37])
        plt.imshow(ct[:, :, 37], alpha=0.5)
        plt.savefig(
            f"/home/baumgartner/sgutwein84/container/test/{segment['segment']}.png")
