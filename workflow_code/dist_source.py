from utils import define_iso_center, define_origin_position
from numba import njit, prange
import numpy as np
from tqdm import tqdm
import torch
import sys

import matplotlib.pyplot as plt


def distance_source(egsinp_path, ct_path, target_size, tensor=False):

    egsinp_lines = open(egsinp_path).readlines()
    iso_center, px_sp = define_iso_center(egsinp_lines[5], ct_path)
    origin_position = define_origin_position(egsinp_lines[5], iso_center, px_sp)

    if tensor:
        return torch.tensor(calculate_distance(origin_position=origin_position, target_size=target_size, px_sp=px_sp), dtype=torch.float32)
    else:
        return calculate_distance(origin_position=origin_position, target_size=target_size, px_sp=px_sp)


def calculate_distance(origin_position, target_size, px_sp):

    distance_vol = np.zeros(target_size)

    for x in tqdm(range(target_size[0]), miniters=20, file=sys.stdout, postfix="\n"):
        for y in range(target_size[1]):
            for z in range(target_size[2]):
                distance_vol[x, y, z] = distance(x, y, z, origin_position, px_sp)

    print("Source Mask created!")
    return distance_vol


@njit
def distance(x, y, z, origin_position, px_sp):

    vox = np.array([x, y, z])
    vox_origin = np.array([
        (vox[0] - origin_position[0])*px_sp[0],
        (vox[1] - origin_position[1])*px_sp[1],
        (vox[2] - origin_position[2])*px_sp[2],
    ])
    dist = np.abs(np.linalg.norm(vox_origin))

    return dist


if __name__ == "__main__":

    egsinp = "/Users/simongutwein/Studium/Masterarbeit/m0/m5_14/beam_config_m5_14.egsinp"
    ct_path = "/Users/simongutwein/Studium/Masterarbeit/m0/ct/m5"
    distance_array = distance_source(egsinp, ct_path, target_size=(512, 512, 108))

    print(distance_array.max())

    for i in range(distance_array.shape[2]):
        plt.imshow(distance_array[:, :, i])
        plt.show()
