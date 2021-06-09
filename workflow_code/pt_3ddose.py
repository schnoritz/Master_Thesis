from typing import final
import numpy as np
import cv2
from cv2 import resize
import torch


def dose_to_pt(dose_path, tensor=False):
    """creates a 3D PyTorch Tensor of shape (512, 512, num_slices)

    Args:
        dose_path (int, pathlile_object): path to the .3ddose file that needs to be converted

    Returns:
        torch.tensor: tensor of shape (512, 512, num_slices)
    """

    dose = read_dose(dose_path)
    dose = upscale(dose, target_size=(512, 512, dose.shape[2]))

    if tensor:
        return torch.tensor(dose)
    else:
        return dose


def read_dose(filepath):
    """reads in a .3ddose file

    Args:
        filepath (str or pathlike object): filepath to the .3ddose file

    Returns:
        np.array: numpy array with the dose
    """

    with open(filepath, 'r') as fout:

        dim = np.array(fout.readline().split()).astype("int")
        for i in range(3):
            fout.readline()
        dose_volume = np.array(fout.readline().split()).astype("float")
        dose_volume = dose_volume.reshape(dim[0], dim[1], dim[2], order='F')
        dose_volume = dose_volume.transpose((1, 0, 2))

    return dose_volume


def upscale(dose, target_size):
    """upscales a dose distribution to a specified targetsize using nearest neighbor interpolation

    Args:
        dose (np.array): dose numpy array
        target_size (tuple or list): dimensions of the 3d array

    Returns:
        np.array: upscaled numpy array
    """

    target_height = int(target_size[0] / dose.shape[1] * dose.shape[0])
    add = np.zeros((target_size[0] - target_height,
                   target_size[0], target_size[2]))

    resized = resize(
        dose, (target_size[0], target_height), interpolation=cv2.INTER_NEAREST)
    final_dose = np.concatenate((add, resized), axis=0)

    return final_dose

# if __name__ == "__main__":

#     dose = dose_to_pt(
#         "/home/baumgartner/sgutwein84/container/output/p_0_2x2/p_0_2x2_1E08.3ddose"
#     )
#     print(dose.dtype)
