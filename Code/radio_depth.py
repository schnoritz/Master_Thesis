#code to calculate radiological depth on a volume

import numpy as np
import math
from numba import njit
import matplotlib.pyplot as plt
from time import time


class ray():
    def __init__(self, origin_position, voxel_position, target_volume):

        self.origin_pos = np.array(origin_position)
        self.voxel_pos = np.array(voxel_position)
        self.target_volume = target_volume
        self.ray_vector = np.array(voxel_position - origin_position)
        self.path = self.calculate_path()

    def calculate_path(self):

        x, y, z = self.voxel_pos[0], self.voxel_pos[1], self.voxel_pos[2]
        if self.ray_vector[0] == 0:
            if self.ray_vector[1] > 1:
                depth = np.sum(self.target_volume[x, 0:y,z])
            else:
                depth = np.sum(self.target_volume[x, y:self.target_volume.shape[1], z])

        elif self.ray_vector[1] == 0:
            if self.ray_vector[0] > 1:
                depth = np.sum(self.target_volume[0:x, y, z])
            else:
                depth = np.sum(self.target_volume[x:self.target_volume.shape[0], y, z])

        else:

            a_min, plane= calculate_a(self.target_volume, self.origin_pos, self.voxel_pos)

            a_max = 1
            i_x_min, i_x_max, i_y_min, i_y_max = get_min_max_index(self.target_volume, self.origin_pos, self.ray_vector, plane, a_min)
            a = calc_alpha(self.origin_pos, self.voxel_pos, i_x_min, i_x_max, i_y_min, i_y_max, a_min, a_max)
            depth = calc_depth(self.origin_pos, self.ray_vector, a, self.target_volume)
            
        return depth


    # def calculate_a(self):

    #     volume_dim = self.target_volume.shape

    #     a_x = np.array([(self.origin_pos[0]-0)/(self.origin_pos[0]-self.voxel_pos[0]),
    #               (self.origin_pos[0]-volume_dim[0])/(self.origin_pos[0]-self.voxel_pos[0])])
        
    #     a_x = a_x[a_x > 0].min()

    #     a_y = np.array([(self.origin_pos[1]-0)/(self.origin_pos[1]-self.voxel_pos[1]),
    #                     (self.origin_pos[1]-volume_dim[1])/(self.origin_pos[1]-self.voxel_pos[1])])

    #     a_y = a_y[a_y > 0].min()

    #     a_min = np.array([a_x, a_y]).min()

    #     if a_y < a_x:
    #         plane = 'y'
    #     else:
    #         plane = 'x'
        
    #     return a_min, plane

    # def get_min_max_index(self,plane, a_min):

    #     if plane == 'x':
    #             i_x_min = 1
    #             i_x_max = self.target_volume.shape[0]
    #             i_y_min = np.floor(
    #                 self.origin_pos[1] + self.ray_vector[1]).astype('int32')
    #             i_y_max = np.ceil(
    #                 self.origin_pos[1] + a_min*self.ray_vector[1]).astype('int32')

    #     else:
    #         i_y_min = 1
    #         i_y_max = self.target_volume.shape[0]
    #         i_x_min = np.floor(self.origin_pos[0] + self.ray_vector[0]).astype('int32')
    #         i_x_max = np.ceil(
    #             self.origin_pos[0] + a_min*self.ray_vector[0]).astype('int32')

    #     return i_x_min, i_x_max, i_y_min, i_y_max

    # def calc_alpha(self, i_x_min, i_x_max, i_y_min, i_y_max, a_min, a_max):

    #     a_x = []
    #     for i in range(i_x_min, i_x_max+1):
    #         a_x.append((self.origin_pos[0]-i)/(self.origin_pos[0]-self.voxel_pos[0]))

    #     a_y = []
    #     for i in range(i_y_min, i_y_max+1):
    #         a_y.append((self.origin_pos[1]-i)/(self.origin_pos[1]-self.voxel_pos[1]))

    #     a_x, a_y = np.array(a_x), np.array(a_y)
    #     a = np.concatenate([a_x, a_y])
    #     a = a[a <= a_max]
    #     a = a[a >= a_min]
    #     a = np.unique(np.sort(a))

    #     return a

    # def calc_depth(self, a):
    #     intersections = np.array([self.origin_pos + alpha * self.ray_vector for alpha in a])
    #     diff = np.array([intersections[i] - intersections[i+1]for i in range(len(intersections)-1)])
    #     inter = [zip(np.ceil((intersections[i]+intersections[i+1]) /2).astype('int32')-1) for i in range(len(intersections)-1)]

    #     lengths = np.array(
    #         [np.sqrt(np.square(var[0]) + np.square(var[1]) + np.square(var[2])) for var in diff])

    #     depth = np.sum(np.array(
    #         [self.target_volume[x, y, z] * lengths[i] for i, (x, y, z) in enumerate(inter)]))

    #     return depth

@njit
def get_min_max_index(target, origin, ray_vec, plane, a_min):

    if plane == 'x':
        i_x_min = 1
        i_x_max = target.shape[0]
        i_y_min = int(np.floor(origin[1] + ray_vec[1]))
        i_y_max = int(np.ceil(origin[1] + a_min*ray_vec[1]))

    else:
        i_y_min = 1
        i_y_max = target.shape[0]
        i_x_min = int(np.floor(origin[0] + ray_vec[0]))
        i_x_max = int(np.ceil(origin[0] + a_min*ray_vec[0]))

    return i_x_min, i_x_max, i_y_min, i_y_max

@njit
def calc_alpha(origin, voxel, i_x_min, i_x_max, i_y_min, i_y_max, a_min, a_max):

    a_x = []
    for i in range(i_x_min, i_x_max+1):
        a_x.append((origin[0]-i)/(origin[0]-voxel[0]))

    a_y = []
    for i in range(i_y_min, i_y_max+1):
        a_y.append((origin[1]-i)/(origin[1]-voxel[1]))

    a_x, a_y = np.array(a_x), np.array(a_y)
    a = np.concatenate((a_x, a_y))
    a = a[a <= a_max]
    a = a[a >= a_min]
    a = np.unique(np.sort(a))

    return a

@njit #code noch debuggen für parallelisierung
def calculate_a(target, origin, voxel):

    volume_dim = target.shape

    a_x = np.array([(origin[0]-0)/(origin[0]-voxel[0]),
                    (origin[0]-volume_dim[0])/(origin[0]-voxel[0])])

    a_x = a_x[a_x > 0].min()

    a_y = np.array([(origin[1]-0)/(origin[1]-voxel[1]),
                     (origin[1]-volume_dim[1])/(origin[1]-voxel[1])])

    a_y = a_y[a_y > 0].min()

    a_min = np.array([a_x, a_y]).min()

    if a_y < a_x:
        plane = 'y'
    else:
        plane = 'x'

    return a_min, plane

@njit
def calc_depth(origin, ray_vector, a, target_volume):
    intersections = np.empty((len(a),3))
    diff = np.empty((intersections.shape[0]-1,intersections.shape[1]))
    lengths = np.empty(diff.shape[0])
    depth = 0.0

    for i in range(len(a)):
        intersections[i,:] = origin + a[i]*ray_vector
    
    inter = np.ceil(intersections[:-1])-1

    for i in range(len(intersections)-1):
        diff[i] = intersections[i] - intersections[i+1]

    for i in range(len(diff)):
        lengths[i] = np.sqrt(np.square(diff[i][0]) + np.square(diff[i][1]) + np.square(diff[i][2]))

    for i in range(len(lengths)):
        depth += lengths[i] * target_volume[int(inter[i][0]), int(inter[i][1]), int(inter[i][2])]   

    return depth

def rgb2gray(rgb):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

if __name__ == "__main__":
    
    origin = np.array([-10,510,25])
    volume = plt.imread("/Users/simongutwein/Studium/Masterarbeit/CT_512.jpg")
    volume = np.resize(volume, (32, volume.shape[0], volume.shape[1]))
    volume = volume.transpose(1, 2, 0)

   
    radio_depth_volume = np.empty((volume.shape[0], volume.shape[1]))
    needed_time = []
    start = time()
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            voxel = np.array([i, j, 20])
            curr_ray = ray(origin, voxel, volume)
            radio_depth_volume[i, j] = curr_ray.path
            
            
    needed_time = time()-start
    time_per_voxel = needed_time/(volume.shape[0]*volume.shape[1])
    print("Needed Time (all): ", needed_time, "\nTime per Voxel: ", time_per_voxel)

    plt.imshow(radio_depth_volume, cmap="magma")
    plt.imshow(volume[:, :, 0], cmap="gray", alpha=0.3)
    plt.show()

               

    
    
