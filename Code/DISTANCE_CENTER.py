import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from FUNCTIONS import *

@timeit
@njit
def distance_matrix(shape, origin, iso):
    dist = np.empty(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                point = np.array([x, y, z])
                dist[x, y, z] = np.linalg.norm(np.cross((point - iso), iso-origin))/np.linalg.norm(iso-origin)

    return dist

if __name__ == "__main__":

    shape = (512, 512, 110)
    origin=np.array([-110., 700., 70.]).astype("float")
    iso_center=np.array([256., 256., 55.]).astype("float")
    vec = iso_center - origin
    distance = distance_matrix(shape, origin, iso_center)
    print(distance.shape)

    # for i in range(shape[2]):
    #     plt.imshow(distance[:, :, i], cmap="gist_ncar")
    #     plt.show()
 

