import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from FUNCTIONS import *

@njit
def binary_mask(mask, origin, iso, bord, n):

    for x in range(mask.shape[0]):
        if x % 10 == 0:
            print(x)
        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                point = np.array([x, y, z])
                inter = np.empty_like(point)
                np.round_(intersection(point, n, iso, point - origin), 6, inter)
                mask[x, y, z] = check(bord, inter)



@njit
def borders(fs, iso, n):

    top_down = np.array([-1., 0., 0.])
    alpha = np.arccos(np.dot(top_down, n))
    x_1, x_2 = iso[0] - np.sin(alpha) * fs/2, iso[0] + np.sin(alpha) * fs/2
    y_1, y_2 = iso[1] - np.cos(alpha) * fs/2, iso[1] + np.cos(alpha) * fs/2
    z_1, z_2 = iso[2] - fs/2, iso[2] + fs/2

    return np.array([[x_1, x_2],[y_1, y_2],[z_1, z_2]])


@njit
def intersection(p, n, iso, r_dir):
	ndotu = n.dot(r_dir)
	w = p - iso
	si = -n.dot(w) / ndotu
	inter = w + si * r_dir + iso
	return inter


@njit
def check(b, inter):

    if within(b[0], inter[0]) and within(b[1], inter[1]) and within(b[2], inter[2]):
        return 1
    else:
        return 0


@njit
def within(nums, num_checked):

    upper = nums.max()
    lower = nums.min()
    if num_checked >= lower and num_checked <= upper:
        return True
    else: 
        return False


def get_origin_fz(SID, phi, iso, fs, px_sp=1.171875):

    phi = phi*np.pi/180

    pixel_SID = int(SID/px_sp)
    origin = np.array([iso[0] - np.cos(phi)*pixel_SID, iso[1] +
                       np.sin(phi)*pixel_SID, iso[2]]).astype("float")
    
    pixel_fz = int(fs/px_sp)
    
    return origin, pixel_fz

def get_binary_mask(mask, angle, fieldsize):

    iso = np.array([mask.shape[0]/2, mask.shape[1]/2,
                   mask.shape[2]/2]).astype("float")
    origin, fs = get_origin_fz(SID=1435, phi=angle, iso=iso, fs=100)
    norm_vec = (iso - origin)/np.linalg.norm(iso - origin)
    bord = borders(fs, iso, norm_vec)
    binary_mask(mask, origin, iso, bord, norm_vec)



if __name__ == "__main__":
    
    nums = 8
    path = "/work/ws/nemo/tu_zxoys08-egs_dat-0/binary/"
    fs = 100 #fieldsize in mm
    for i in range(nums):
        target_filename = "binary_" + str(i*45) + "_" + str(int(fs/10))
        mask = np.empty((512, 512, 110))
        get_binary_mask(mask, i*45, fs)
        with open(path + target_filename + ".npy", 'wb+') as fout:
            #can be read with np.load(fin)
            np.save(fout, mask)
            print(path + target_filename, "created.")

     
