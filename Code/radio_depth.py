#code to calculate radiological depth on a volume

import numpy as np
import math
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ray():
    def __init__(self, origin_position, voxel_position, target_volume):

        self.origin_pos = np.array(origin_position)
        self.voxel_position = np.array(voxel_position)
        self.ray_vector = np.array(voxel_position - origin_position)
        self.path= self.calculate_path()
        self.path_lengths = self.calculate_path_length()
        self.radiological_depth = self.calculate_radiological_depth(target_volume)


    def calculate_path(self):

        var = np.array([(64-self.origin_pos[0])/self.ray_vector[0], (64-self.origin_pos[1])/self.ray_vector[1]])

        var = var[np.isfinite(var)].min()
        
        border_intersection = np.array([ self.origin_pos[0]+var*self.ray_vector[0],  self.origin_pos[1]+var*self.ray_vector[1],  self.origin_pos[2]+var*self.ray_vector[2]])
        directions = np.array([self.ray_vector/abs(self.ray_vector[0]), self.ray_vector/abs(self.ray_vector[1]), self.ray_vector/abs(self.ray_vector[2])])

        idxs = []
        if self.ray_vector[0] > 0:
            idxs.append(np.ceil(border_intersection[0]))
        else:
            idxs.append(np.floor(border_intersection[0]))
        if self.ray_vector[1] > 0:
            idxs.append(np.ceil(border_intersection[1]))
        else:
            idxs.append(np.floor(border_intersection[1]))
        if self.ray_vector[2] > 0:
            idxs.append(np.ceil(border_intersection[2]))
        else:
            idxs.append(np.floor(border_intersection[2]))

        x, y, z = self.calculate_intersection(idxs)

        if not np.isnan(x):
            x_plane = np.array([ self.origin_pos[0] + x*self.ray_vector[0],  self.origin_pos[1] + x*self.ray_vector[1],  self.origin_pos[2] + x*self.ray_vector[2]])
        
        if not np.isnan(y):
            y_plane = np.array([ self.origin_pos[0] + y*self.ray_vector[0],  self.origin_pos[1] + y*self.ray_vector[1],  self.origin_pos[2] + y*self.ray_vector[2]])
        
        if not np.isnan(z):
            z_plane = np.array([ self.origin_pos[0] + z*self.ray_vector[0],  self.origin_pos[1] + z*self.ray_vector[1],  self.origin_pos[2] + z*self.ray_vector[2]])

        path = np.empty((0,3), dtype=int)

        if 'x_plane' in locals():
            path = np.append(path, self.calculate_plane_intersections(
                x_plane, directions, 0), axis=0)

        if 'y_plane' in locals():
            path = np.append(path, self.calculate_plane_intersections(
                y_plane, directions, 1), axis=0)

        if 'z_plane' in locals():
            path = np.append(path, self.calculate_plane_intersections(
                z_plane, directions, 2), axis=0)
        
        idx = np.nonzero(self.ray_vector)
        path = path[path[:, idx[0][0]].argsort()].round(5) 
        path =  np.unique(path, axis=0)
        
        with np.printoptions(threshold=np.inf):
            print(path)

        return np.array(path)


    def calculate_intersection(self, idxs): 
        
        
        x = (idxs[0]-self.origin_pos[0])/self.ray_vector[0]

        y = (idxs[1]-self.origin_pos[1])/self.ray_vector[1]

        z = (idxs[2]-self.origin_pos[2])/self.ray_vector[2]

        return x, y, z


    def calculate_plane_intersections(self, point, dir, num):

        path = []
        for i in range(abs(int(point[num])-self.voxel_position[num])+1):
            path.append(point+i*dir[num])

        return np.array(path)


    def calculate_path_length(self):

        lengths = np.empty((len(self.path)-1,1))

        for i in range(len(self.path)-1):
            lengths[i] = np.sqrt(np.sum(np.square(self.path[i]-self.path[i+1])))
    
        return lengths


    def calculate_radiological_depth(self, target_volume):

        voxels = self.path[1:].astype('int32')
        
        radiological_depth = 0
        for i in range(len(voxels)):
            radiological_depth += target_volume[voxels[i,0]-1, voxels[i,1]-1, voxels[i,2]-1]*self.path_lengths[i]

        return radiological_depth


def function():
    pass


if __name__ == "__main__":
    

    origin = np.array([51,128,16])
    volume = np.zeros((64, 64, 32))
    volume[20:40, 20:40, 15:25] = 1
    voxel = np.array([51, 0, 20])
    curr_ray = ray(origin, voxel, volume)
    #print(curr_ray.radiological_depth)

    radio_depth_volume = np.empty((64,64))
    for i in range(64):
        for j in range(64):
            print(i,j)
            voxel = np.array([i, j, 20])
            curr_ray = ray(origin, voxel, volume)
            radio_depth_volume[i,j] = curr_ray.radiological_depth

    plt.imshow(radio_depth_volume)
    plt.show()

               

    
    
