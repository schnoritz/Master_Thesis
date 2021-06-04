import numpy as np
from numba import njit, prange
import numba


class Ray:
    def __init__(self, origin_position, voxel_position, target_volume):

        self.origin_pos = np.array(origin_position).astype(np.float64)
        # np.round(np.array(origin_position), 4)
        self.voxel_pos = np.array(voxel_position)
        self.target_volume = target_volume
        self.ray_vector = np.array(voxel_position - origin_position)
        self.radiological_depth = self.calculate_path()

    def calculate_path(self):

        a_min = calc_a_min(self.target_volume, self.origin_pos, self.ray_vector)
        a_max = 1

        assert a_min <= a_max, "a_min is bigger than a_max"

        outer_inter = np.empty((3, 2))
        get_min_max_index(
            self.target_volume, self.origin_pos, self.ray_vector, a_min, outer_inter
        )

        a = calc_alpha(self.origin_pos, self.voxel_pos, outer_inter, a_min, a_max)
        depth = calc_depth(self.origin_pos, self.ray_vector, a, self.target_volume)

        return depth


@njit
def check_alpha(r, eps, v, origin, idx):

    if abs(r[idx]) <= eps:
        a = np.array([np.inf, np.inf])
    else:
        a = np.array([(0 - origin[idx]) / (r[idx]), (v[idx] - origin[idx]) / (r[idx])])

    a = a[a > 0].min()

    return a


@njit
def calc_a_min(target, origin, ray_vec):

    a_min = 0
    volume_dim = np.array(target.shape) - 1
    epsilon = 1e-5

    a_x = check_alpha(ray_vec, epsilon, volume_dim, origin, 0)
    a_y = check_alpha(ray_vec, epsilon, volume_dim, origin, 1)

    if origin[0] + a_y * ray_vec[0] > volume_dim[0] or origin[0] + a_y * ray_vec[0] < 0:
        a_min = a_x
    elif (
        origin[1] + a_x * ray_vec[1] > volume_dim[1] or origin[1] + a_x * ray_vec[1] < 0
    ):
        a_min = a_y
    else:
        a_min = np.array([a_x, a_y]).min()

    return a_min


@njit
def check_direction(r, mm, idx, origin, a_min):

    if r[idx] < 0:
        mm[idx][0] = int(np.ceil(abs(origin[idx] + r[idx])))
        mm[idx][1] = int(np.floor(abs(origin[idx] + a_min * r[idx])))

    elif r[idx] > 0:
        mm[idx][0] = int(np.floor(abs(origin[idx] + a_min * r[idx])))
        mm[idx][1] = int(np.ceil(abs(origin[idx] + r[idx])))

    else:
        mm[idx][0], mm[idx][1] = -1, -1


@njit
def get_min_max_index(target, origin, ray_vec, a_min, min_max):

    for direction in range(2):
        check_direction(ray_vec, min_max, direction, origin, a_min)


@njit
def calc_alpha(origin, voxel, min_max, a_min, a_max):

    a_x = []
    if not int(min_max[0][0]) == -1 and origin[0] - voxel[0] != 0:
        for i in prange(int(min_max[0][0]), int(min_max[0][1]) + 1):
            a_x.append((origin[0] - i) / (origin[0] - voxel[0]))

    a_y = []
    if not int(min_max[1][0]) == -1 and origin[1] - voxel[1] != 0:
        for i in prange(int(min_max[1][0]), int(min_max[1][1]) + 1):
            a_y.append((origin[1] - i) / (origin[1] - voxel[1]))

    a_z = []
    if not int(min_max[2][0]) == -1 and origin[2] - voxel[2] != 0:
        for i in prange(int(min_max[2][0]), int(min_max[2][1]) + 1):
            a_z.append((origin[2] - i) / (origin[2] - voxel[2]))

    a_x, a_y, a_z = np.array(a_x), np.array(a_y), np.array(a_z)
    a = np.concatenate((a_x, a_y, a_z))
    a = a[a <= a_max]
    a = a[a >= a_min]
    a = np.unique(np.sort(a))

    return a


@njit
def calc_depth(origin, ray_vector, a, target_volume):

    intersections = np.empty((len(a), 3))
    diff = np.empty((intersections.shape[0] - 1, intersections.shape[1]))
    lengths = np.empty(diff.shape[0])
    depth = 0.0

    for i in prange(len(a)):
        intersections[i, :] = origin + a[i] * ray_vector

    # inter = np.ceil(intersections[:-1])#-1
    inter = np.empty_like(intersections)
    np.round_(intersections, 8, inter)
    inter = inter.astype(numba.int32)

    for i in prange(len(intersections) - 1):
        diff[i] = intersections[i] - intersections[i + 1]

    for i in prange(len(diff)):
        lengths[i] = np.sqrt(
            np.square(diff[i][0]) + np.square(diff[i][1]) + np.square(diff[i][2])
        )

    for i in prange(len(lengths)):
        depth += (
            lengths[i]
            * target_volume[int(inter[i][0]), int(inter[i][1]), int(inter[i][2])]
        )

    return depth
