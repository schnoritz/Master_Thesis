from utils import define_iso_center, define_origin_position
from numba import njit, prange
import numpy as np
from tqdm import tqdm
import torch
import sys

import matplotlib.pyplot as plt


def distance_source(egsinp_path, egsphant_path, target_size, tensor=False):

    egsinp_lines = open(egsinp_path).readlines()
    origin_position = define_origin_position(
        egsinp_lines[5], define_iso_center(egsinp_lines[5], egsphant_path))

    if tensor:
        return torch.tensor(calculate_distance(origin_position=origin_position, target_size=target_size), dtype=torch.float16)
    else:
        return calculate_distance(origin_position=origin_position, target_size=target_size)


def calculate_distance(origin_position, target_size):

    distance_vol = np.zeros(target_size)

    for x in tqdm(range(target_size[0]), miniters=50, file=sys.stdout, postfix="\n"):
        for y in range(target_size[1]):
            for z in range(target_size[2]):
                distance_vol[x, y, z] = distance(x, y, z, origin_position)

    print("Source Mask created!")
    return distance_vol


@njit
def distance(x, y, z, origin_position):

    vox = np.array([x, y, z])
    dist = np.abs(np.linalg.norm(origin_position - vox))

    return dist


# if __name__ == "__main__":

#     path = "/home/baumgartner/sgutwein84/container/output/p_0_2x2/p_0_2x2.egsinp"
#     phant = "/home/baumgartner/sgutwein84/container/output/p.egsphant"

#     distance_array = distance_source
# (path, phant, target_size=(512, 512, 110))
#     print(distance_array.target_size)

#     for i in range(distance_array.target_size[2]):
#         plt.imshow(distance_array[:,:,i])
#         plt.show()
#         plt.close()
