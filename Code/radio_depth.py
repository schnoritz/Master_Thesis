#code to calculate radiological depth on a volume


import numpy as np
import math
from numba import njit, prange
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

        a_min= calc_a_min(self.target_volume, self.origin_pos, self.ray_vector)
        a_max = 1

        i_x_min, i_x_max, i_y_min, i_y_max, i_z_min, i_z_max = get_min_max_index(self.target_volume, self.origin_pos, self.ray_vector, a_min)
        
        a = calc_alpha(self.origin_pos, self.voxel_pos, i_x_min, i_x_max, i_y_min, i_y_max, i_z_min, i_z_max, a_min, a_max)
        depth = calc_depth(self.origin_pos, self.ray_vector, a, self.target_volume)
            
        return depth


@njit
def calc_a_min(target, origin, ray_vec):

    a_min = 0
    plane = 'N'
    volume_dim = np.array(target.shape)-1

    if ray_vec[0] == 0:
        a_x = np.array([np.inf, np.inf])
    else:
        a_x = np.array([(0-origin[0])/(ray_vec[0]),
                        (volume_dim[0]-origin[0])/(ray_vec[0])])

    a_x = a_x[a_x > 0].min()

    if ray_vec[1] == 0:
        a_y = np.array([np.inf, np.inf])
    else:
        a_y = np.array([(0-origin[1])/(ray_vec[1]),
                        (volume_dim[1]-origin[1])/(ray_vec[1])])

    a_y = a_y[a_y > 0].min()

    if origin[0] + a_y * ray_vec[0] > volume_dim[0] or origin[0] + a_y * ray_vec[0] < 0:
        a_min = a_x
    elif origin[1] + a_x * ray_vec[1] > volume_dim[1] or origin[1] + a_x * ray_vec[1] < 0:
        a_min = a_y
    else:
        a_min = np.array([a_x, a_y]).min()

    return a_min


@njit
def get_min_max_index(target, origin, ray_vec, a_min):

    if ray_vec[0] < 0:
        i_x_min = int(np.ceil(origin[0] + ray_vec[0]))
        i_x_max = int(np.floor(origin[0] + a_min*ray_vec[0]))
    elif ray_vec[0] > 0:
        i_x_min = int(np.floor(origin[0] + a_min*ray_vec[0]))
        i_x_max = int(np.ceil(origin[0] + ray_vec[0]))
    else:
        i_x_min, i_x_max = -1, -1


    if ray_vec[1] < 0:
        i_y_min = int(np.ceil(origin[1] + ray_vec[1]))
        i_y_max = int(np.floor(origin[1] + a_min*ray_vec[1]))
    elif ray_vec[1] > 0:
        i_y_min = int(np.floor(origin[1] + a_min*ray_vec[1]))
        i_y_max = int(np.ceil(origin[1] + ray_vec[1]))
    else:
        i_y_min, i_y_max = -1, -1
    
    if ray_vec[2] < 0:
        i_z_min = int(np.ceil(origin[2] + ray_vec[2]))
        i_z_max = int(np.floor(origin[2] + a_min*ray_vec[2]))
    elif ray_vec[2] > 0:
        i_z_min = int(np.floor(origin[2] + a_min*ray_vec[2]))
        i_z_max = int(np.ceil(origin[2] + ray_vec[2]))
    else:
        i_z_min, i_z_max = -1, -1


    return i_x_min, i_x_max, i_y_min, i_y_max, i_z_min, i_z_max


@njit
def calc_alpha(origin, voxel, i_x_min, i_x_max, i_y_min, i_y_max, i_z_min, i_z_max, a_min, a_max):

    a_x = []
    if not i_x_min == -1:
        for i in range(i_x_min, i_x_max+1):
            a_x.append((origin[0]-i)/(origin[0]-voxel[0]))

    a_y = []
    if not i_y_min == -1:
        for i in range(i_y_min, i_y_max+1):
            a_y.append((origin[1]-i)/(origin[1]-voxel[1]))

    a_z = []
    if not i_z_min == -1:
        for i in range(i_z_min, i_z_max+1):
            a_z.append((origin[2]-i)/(origin[2]-voxel[2]))

    a_x, a_y, a_z = np.array(a_x), np.array(a_y), np.array(a_z)
    a = np.concatenate((a_x, a_y, a_z))
    a = a[a <= a_max]
    a = a[a >= a_min]
    a = np.unique(np.sort(a))
    
    return a

@njit
def calc_depth(origin, ray_vector, a, target_volume):
    intersections = np.empty((len(a),3))
    diff = np.empty((intersections.shape[0]-1,intersections.shape[1]))
    lengths = np.empty(diff.shape[0])
    depth = 0.0

    for i in prange(len(a)):
        intersections[i,:] = origin + a[i]*ray_vector
    
    inter = np.ceil(intersections[:-1])#-1
    # with np.printoptions(threshold=np.inf):
    #     print(np.round(intersections,1))
   
    for i in prange(len(intersections)-1):
        diff[i] = intersections[i] - intersections[i+1]

    for i in prange(len(diff)):
        lengths[i] = np.sqrt(np.square(diff[i][0]) + np.square(diff[i][1]) + np.square(diff[i][2]))

    for i in prange(len(lengths)):
        depth += lengths[i] * target_volume[int(inter[i][0]), int(inter[i][1]), int(inter[i][2])]   

    return depth

def rgb2gray(rgb):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def debug(dat=None, rotate=True):

    if rotate==True:
        angle = np.linspace(0, 1, 40, endpoint=False)*2*np.pi
        x = np.round(800*np.cos(angle),0)
        y = np.round(800*np.sin(angle),0)
        origin = []
        for i,j in zip(x,y):
            origin.append([i,j,35])

        origins = np.array(origin)
        #print(origins)
    else:
        origins = [np.array(dat)]
        #print(origins)
    return origins


if __name__ == "__main__":
    
    volume = np.array(rgb2gray(plt.imread("/Users/simongutwein/Studium/Masterarbeit/CT_2.jpg")))
    volume = np.resize(volume, (64, volume.shape[0], volume.shape[1]))
    volume = volume.transpose(1, 2, 0)
    radio_depth_volume = np.empty((volume.shape[0], volume.shape[1]))
    voxel = np.array([95,15,20])
    origin = np.array([-800., 0., 35.])
    curr_ray = ray(origin, voxel, volume)
    print(curr_ray.path)


    for origin in debug(rotate=True):
        print(origin)
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

        plt.imshow(radio_depth_volume, cmap="jet")
        plt.imshow(volume[:, :, 20], cmap="gray", alpha=0.5)
        plt.show()
