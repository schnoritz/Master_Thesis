import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from FUNCTIONS import *

@njit
def binary_mask(mask, origin, iso, bord, n):

    progress = -5
    for x in range(mask.shape[0]):
        if x % int(mask.shape[0]/20) == 0:
            progress += 5
            print("Binary Mask || Progress: " + str(progress) + "%")

        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                point = np.array([x, y, z])
                inter = np.empty_like(point)
                np.round_(intersection(point, n, iso, point - origin), 6, inter)
                mask[x, y, z] = check(bord, inter)


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

def get_binary_mask(mask, angle, fieldsize):

    fieldsize = fieldsize/1.171875
    iso = np.array([mask.shape[0]/2, mask.shape[1]/2,
                   mask.shape[2]/2]).astype("float")
    origin, fs = get_origin_fz(SID=1435, phi=angle, iso=iso, fs=fieldsize)
    norm_vec = (iso - origin)/np.linalg.norm(iso - origin)
    bord = borders(fs, iso, norm_vec)
    binary_mask(mask, origin, iso, bord, norm_vec)


if __name__ == "__main__":
    
    nums = 8
    path = "/work/ws/nemo/tu_zxoys08-egs_dat-0/binary/"
    fs = 50 #fieldsize in mm
    for i in range(nums):
        target_filename = str(i*45) + "_" + str(int(fs/10)) + "x" + str(int(fs/10)) + "_0x0"
        mask = np.empty((512, 512, 110))
        get_binary_mask(mask, i*45, fs)
