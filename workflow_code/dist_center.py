from utils import define_iso_center, define_origin_position
from numba import njit, prange
import numpy as np
from tqdm import tqdm
import torch
import sys

import matplotlib.pyplot as plt


def distance_center(egsinp_path, ct_path, target_size, tensor=False):

    egsinp_lines = open(egsinp_path).readlines()

    iso_center, px_sp = define_iso_center(egsinp_lines[5], ct_path)
    origin_position = define_origin_position(egsinp_lines[5], iso_center, px_sp)

    if tensor:
        return torch.tensor(calculate_distance(iso_center=iso_center, origin_position=origin_position, target_size=target_size, px_sp=px_sp), dtype=torch.float32)
    else:
        return calculate_distance(iso_center=iso_center, origin_position=origin_position, target_size=target_size, px_sp=px_sp)


def calculate_distance(iso_center, origin_position, target_size, px_sp):

    distance_vol = np.zeros(target_size).astype(float)

    for x in tqdm(range(target_size[0]), miniters=20, file=sys.stdout, postfix="\n"):
        for y in range(target_size[1]):
            for z in range(target_size[2]):
                distance_vol[x, y, z] = distance(
                    x, y, z, iso_center, origin_position, px_sp)

    print("Center Mask created!")
    return distance_vol


@njit
def distance(x, y, z, iso_center, origin_position, px_sp):

    vox = np.array([x, y, z])

    vox_iso = np.array([
        (vox[0] - iso_center[0])*px_sp[0],
        (vox[1] - iso_center[1])*px_sp[1],
        (vox[2] - iso_center[2])*px_sp[2],
    ])

    iso_origin = np.array([
        (iso_center[0] - origin_position[0])*px_sp[0],
        (iso_center[1] - origin_position[1])*px_sp[1],
        (iso_center[2] - origin_position[2])*px_sp[2],
    ])

    dist = np.linalg.norm(np.cross((vox_iso), iso_origin)) / np.linalg.norm(iso_origin)

    return dist


if __name__ == "__main__":

    egsinp = "/Users/simongutwein/Studium/Masterarbeit/m0/m5_14/beam_config_m5_14.egsinp"
    ct_path = "/Users/simongutwein/Studium/Masterarbeit/m0/ct/m5"
    distance_array = distance_center(egsinp, ct_path, target_size=(512, 512, 108))

    for i in range(distance_array.shape[2]):
        plt.imshow(distance_array[:, :, i])
        plt.show()
