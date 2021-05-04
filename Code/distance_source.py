import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from FUNCTIONS import *

@timeit
@njit
def distance_matrix(shape, origin):
    dist = np.empty(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                point = np.array([x, y, z])
                dist[x, y, z] = np.abs(np.linalg.norm(origin - point))

    return dist

if __name__ == "__main__":

    shape = (512, 512, 110)
    origin=np.array([-1100., 200., 70.]).astype("float")
    distance = distance_matrix(shape, origin)
    print(distance.shape)

    # for i in range(shape[2]):
    #     plt.imshow(distance[:, :, i], cmap="gist_ncar")
    #     plt.show()
 
