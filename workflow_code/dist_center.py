from utils import define_iso_center, define_origin_position
from numba import njit, prange
import numpy as np
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt


def distance_center(egsinp_path, egsphant_path, shape, tensor=False):

    egsinp_lines = open(egsinp_path).readlines()

    iso_center = define_iso_center(egsinp_lines[5], egsphant_path)
    origin_position = define_origin_position(egsinp_lines[5], iso_center)

    if tensor:
        return torch.tensor(calculate_distance(iso_center=iso_center, origin_position=origin_position, shape=shape), dtype=torch.float16)
    else:
        calculate_distance(iso_center=iso_center,
                           origin_position=origin_position, shape=shape)


def calculate_distance(iso_center, origin_position, shape):

    distance_vol = np.zeros(shape)

    for x in tqdm(range(shape[0]), miniters=50):
        for y in range(shape[1]):
            for z in range(shape[2]):
                distance_vol[x, y, z] = distance(
                    x, y, z, iso_center, origin_position)

    print("Center Mask created!")
    return distance_vol


@njit
def distance(x, y, z, iso_center, origin_position):

    vox = np.array([x, y, z])
    dist = np.linalg.norm(np.cross((vox - iso_center), iso_center -
                          origin_position)) / np.linalg.norm(iso_center - origin_position)

    return dist

# if __name__ == "__main__":

#     path = "/home/baumgartner/sgutwein84/container/output/p_0_2x2/p_0_2x2.egsinp"
#     phant = "/home/baumgartner/sgutwein84/container/output/p.egsphant"
#     distance_array = distance_center(path, phant, shape=(512, 512, 110))
