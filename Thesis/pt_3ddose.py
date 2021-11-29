import numpy as np
import cv2
from cv2 import resize
from pydicom import dcmread, uid
import torch
import os
import scipy.ndimage as spy

import matplotlib.pyplot as plt
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

    dose, dose_limits, spacing = read_dose(dose_path)
    ct_limits = get_ct_lmits(cts)

    diff = ct_limits-dose_limits
    dose = upscale(dose, target_size=(512, 512), limits=diff, spacing=spacing)

    if tensor:
        return torch.tensor(dose)
    else:
        return dose


def get_ct_lmits(path):

    with dcmread(sorted(path)[0], force=True) as fin:

        slice_spacing = int(fin.SliceThickness)
        pixel_spacing = fin.PixelSpacing

        px_sp = np.array([pixel_spacing[0], pixel_spacing[1], slice_spacing]).astype(float)
        print(px_sp)
        start = np.array(fin.ImagePositionPatient).astype(float)
        end = np.array(fin.ImagePositionPatient).astype(float)+px_sp[0]*512
        end[-1] = np.array(fin.ImagePositionPatient).astype(float)[-1] + px_sp[2]*len(path)

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

        dim = np.array(fout.readline().split()).astype(int)

        dose_limits = []
        spacing = []
        for _ in range(3):
            dat = fout.readline().split()
            dose_limits.extend([float(dat[0]), float(dat[-1])])
            spacing.append(abs(float(dat[1]) - float(dat[0])))

        dose_volume = np.array(fout.readline().split()).astype("float")
        dose_volume = dose_volume.reshape(dim[0], dim[1], dim[2], order='F')
        dose_volume = dose_volume.transpose((1, 0, 2))

        spacing[0], spacing[1] = spacing[1], spacing[0]

        dose_volume = spy.zoom(dose_volume, (1, 1, spacing[2]/0.3))
        spacing[-1] = 0.3

        print(dose_volume.shape)

    return dose_volume, dose_limits, spacing


def upscale(dose, target_size, limits, spacing):
    """upscales a dose distribution to a specified targetsize using nearest neighbor interpolation

    Args:
        dose (np.array): dose numpy array
        target_size (tuple or list): dimensions of the 3d array

    Returns:
        np.array: upscaled numpy array
    """

    limits = [abs(int(limits[i]/spacing[int(i/2)])) for i in range(6)]
    print(limits, spacing)

    top = np.zeros((limits[2], dose.shape[1], dose.shape[2]))
    bottom = np.zeros((limits[3], dose.shape[1], dose.shape[2]))

    dose = np.concatenate((top, dose, bottom))

    left = np.zeros((dose.shape[0], limits[0], dose.shape[2]))
    right = np.zeros((dose.shape[0], limits[1], dose.shape[2]))

    dose = np.concatenate((left, dose, right), axis=1)
    print(dose.shape)

    final_dose = resize(
        dose, target_size, interpolation=cv2.INTER_LINEAR)

    return final_dose


if __name__ == "__main__":

    import nibabel as nib

    segments = ["h1_0"]
    patient = "h1"
    entity = "head"
    print(segments)

    for segment in segments:

        dose = dose_to_pt(
            f"/mnt/qb/baumgartner/sgutwein84/output_{entity}/{segment}/{segment}_1E07.3ddose",
            f"/mnt/qb/baumgartner/sgutwein84//output_{entity}/ct/{patient}")

        ct_path = f"/mnt/qb/baumgartner/sgutwein84/output_{entity}/ct/{patient}"
        ct = convert_ct_array(ct_path, target_size=dose.shape, tensor=False)

        dose = dose/dose.max()

        print(ct.shape)
        print(dose.shape)

        plt.imshow(ct[:, :, 40])
        plt.imshow(dose[:, :, 40], alpha=0.9)
        plt.savefig(f"/home/baumgartner/sgutwein84/container/test/test.png")

        dat = nib.Nifti1Image(np.array(dose), np.eye(4))
        dat.header.get_xyzt_units()
        dat.to_filename(f"/home/baumgartner/sgutwein84/container/test/dose.nii.gz")
        dat = nib.Nifti1Image(np.array(ct), np.eye(4))
        dat.header.get_xyzt_units()
        dat.to_filename(f"/home/baumgartner/sgutwein84/container/test/ct.nii.gz")
