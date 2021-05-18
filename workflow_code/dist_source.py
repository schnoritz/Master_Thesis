from utils import define_iso_center, define_origin_position
from numba import njit, prange
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


def distance_center(egsinp_path, egsphant_path, shape):

    egsinp_lines = open(egsinp_path).readlines()
    origin_position = define_origin_position(egsinp_lines[5], define_iso_center(egsinp_lines[5], egsphant_path))

    return calculate_distance(origin_position=origin_position, shape=shape)


def calculate_distance(origin_position, shape):

    distance_vol = np.zeros(shape)

    for x in tqdm(range(shape[0])):
        for y in range(shape[1]):
            for z in range(shape[2]):
                distance_vol[x, y, z] = distance(x, y, z, origin_position)

    return distance_vol


@njit
def distance(x, y, z, origin_position):

    vox = np.array([x, y, z])
    dist = np.abs(np.linalg.norm(origin_position - vox))

    return dist


# if __name__ == "__main__":

#     path = "/home/baumgartner/sgutwein84/container/output/p_0_2x2/p_0_2x2.egsinp"
#     phant = "/home/baumgartner/sgutwein84/container/output/p.egsphant"

#     distance_array = distance_center(path, phant, shape=(512, 512, 110))
#     print(distance_array.shape)

#     for i in range(distance_array.shape[2]):
#         plt.imshow(distance_array[:,:,i])
#         plt.show()
#         plt.close()