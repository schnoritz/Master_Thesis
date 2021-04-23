import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import progressbar
from FUNCTIONS import timeit

#creates distance to center, distance to source and binary 


def get_origin(SID, phi, iso, px_sp=1.171875):
    """calculates origin position on 3d volume dimensions with voxel size of 1.171875 as standard

    Args:
        SID (int): source iso center distance in mm
        phi (int): angle from top down view in degree ([0-360[)
        iso (np.array((3,1))): [description]
        px_sp (float, optional): [voxel size in x and y dimension]. Defaults to 1.171875.

    Returns:
        origin (np.array((3,1)): origin position in 3d space with x, y, z 
    """
    phi = phi*np.pi/180

    pixel_SID = int(SID/px_sp)
    origin = np.array([iso[0] - np.cos(phi)*pixel_SID, iso[1] +
                       np.sin(phi)*pixel_SID, iso[2]]).astype("float")

    return origin


def scale_fieldsize(fs, px_sp=1.171875):
    """scale fieldsize accordingly to pixel spacing in x and y dimension and 3 mm slice thicknes

    Args:
        fs (int): fieldsize in mm
        px_sp (float, optional): pixel spacing in x and y direction. Defaults to 1.171875.

    Returns:
        [type]: [description]
    """ 
    #wie hier rundsen? problem: vielleicht immer aufrunden?   
    return np.array([np.round(fs/px_sp, 0), np.round(fs/px_sp,0), np.round(fs/3,0)]).astype("float")


@njit
def calc_distance_source(dist, origin):
    """calculates 3d mask of source distance of easch pixel

    Args:
        dist (np.array(target_size)): volume to be filled in this function
        origin (np.array(3,1)): position of origin
    """
    progress = -5
    for x in range(dist.shape[0]):
        if x % int(dist.shape[0]/20) == 0:
            progress += 5
            print("Sorce Distance           || Progress: " + str(progress) + "%")

        for y in range(dist.shape[1]):
            for z in range(dist.shape[2]):
                point = np.array([x, y, z])
                dist[x, y, z] = np.abs(np.linalg.norm(origin - point))


@njit
def calc_distance_center(dist, origin, iso):
    """calculates 3d mask of distance to center of bean of easch pixel

    Args:
        dist (np.array(target_size)): volume to be filled in this function
        origin (np.array(3,1)): position of origin
        iso (np.array(3,1)): position of iso center
    """
    n = (iso-origin)/np.linalg.norm(iso-origin)
    progress = -5
    for x in range(dist.shape[0]):
        if x % int(dist.shape[0]/20) == 0:
            progress += 5
            print("Center Beam Distance     || Progress: " + str(progress) + "%")

        for y in range(dist.shape[1]):
            for z in range(dist.shape[2]):
                point = np.array([x, y, z])
                dist[x, y, z] = np.linalg.norm(np.cross((point - iso), n))


@timeit
def create_trainings_masks(SID, angle, fs):

    # define needed directories
    main_dir = "/work/ws/nemo/tu_zxoys08-egs_dat-0/"
    training = main_dir + "training/"
    binary = training + "binary/"
    center_distance = training + "center_distance/"
    source_distance = training + "source_distance/"

    #define target filenames for binary file with field information and with out for center and source distance
    target_filename = str(angle) + "_" + str(int(fs/10)) + \
        "x" + str(int(fs/10)) + "_0x0.npy"

    target_filename_cs = str(angle) + ".npy"


    print("Creating: " + target_filename)
    
    #define size of desired output array
    target_size = (512, 512, 110)

    #calculate iso center in middle of the 3d volume 
    iso_center = np.array([target_size[0]/2-1, target_size[1]/2-1, target_size[2]/2-1]).astype("float")
    origin = get_origin(SID, angle, iso_center).astype("float")
    fs = scale_fieldsize(fs)

    # calculate source distance map
    save_source = False
    if not os.path.isfile(source_distance + target_filename_cs):
        dist_source = np.empty(target_size).astype("float")
        calc_distance_source(dist_source, origin)
        print("------------------------------------------")
        save_source = True
    else: 
        print(center_distance + target_filename_cs + " already exists!")

    # calculate center beam distance map
    save_center = False
    if not os.path.isfile(center_distance + target_filename_cs):
        dist_center = np.empty(target_size).astype("float")
        calc_distance_center(dist_center, origin, iso_center)
        print("------------------------------------------")
        save_center = True
    else:
        print(center_distance + target_filename_cs + " already exists!")

    # calculate binary mask array
    save_binary = False
    if not os.path.isfile(binary + target_filename):
        binary_mask = np.empty(target_size).astype("float")
        get_binary_mask(binary_mask, angle, fs, origin, iso_center)
        print("------------------------------------------")
        save_binary = True
    else:
        print(binary + target_filename + " already exists!")

    # save generated 3d volumes to given path
    if save_source:
        save_to_path(source_distance + target_filename_cs, dist_source)
        print("Saved Source Distance")

    if save_center:
        save_to_path(center_distance + target_filename_cs, dist_center)
        print("Saved Center Distance")

    if save_binary:
        save_to_path(binary + target_filename, binary_mask)
        print("saved Binary Mask")


def save_to_path(filepath, arr):
    """saves array to given filepath

    Args:
        filepath (str): desired filepath
        arr (np.array): array to be saved
    """
    with open(filepath, "wb+") as fout:
        np.save(fout, arr)



def get_binary_mask(mask, angle, fs, origin, iso):
    """calculates the binary mask for given size, angle, and fieldsize

    Args:
        mask (np.array): 3d volume to be filled in this function
        angle (int): gantry angle in degrees
        fs (int): fieldsize in pixel dimensions 
        origin (np.array): origin position
        iso (np.array): iso center position
    """
    norm_vec = (iso - origin)/np.linalg.norm(iso - origin)
    bord = borders(fs, iso, norm_vec)
    binary_mask(mask, origin, iso, bord, norm_vec)


@njit
def borders(fs, iso, norm_vec):
    """calculates borders for binary mask check

    Args:
        fs (int): fielssize in pixel dimensions
        iso (np.array): iso center pisition 
        norm_vec (np.array): vector normalized to size one of central plane colinear
            to ray direction

    Returns:
        borders (np.array(3,2)): boarders of field in central plane 
    """
    top_down = np.array([-1., 0., 0.])
    alpha = np.arccos(np.dot(top_down, norm_vec))
    x_1, x_2 = iso[0] - np.sin(alpha) * fs[0]/2, iso[0] + np.sin(alpha) * fs[0]/2
    y_1, y_2 = iso[1] - np.cos(alpha) * fs[1]/2, iso[1] + np.cos(alpha) * fs[1]/2
    z_1, z_2 = iso[2] - (fs[2]/2), iso[2] + (fs[2]/2)

    return np.array([[x_1, x_2], [y_1, y_2], [z_1, z_2]])


@njit
def binary_mask(mask, origin, iso, borders, norm_vec):
    """calculates binary mask

    Args:
        mask (np.array): array to be filled in this function
        origin (np.array): origin position in 3d space
        iso (np.array): iso center position in 3d space
        borders (np.array(3,2)): borders of field in 3d space in central plane
        norm_vec (np.array): normalized normal vector to size one of central plane
    """
    progress = -5
    for x in range(mask.shape[0]):
        if x % int(mask.shape[0]/20) == 0:
            progress += 5
            print("Binary Mask              || Progress: " + str(progress) + "%")

        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                point = np.array([x, y, z])
                inter = np.empty_like(point)
                np.round_(intersection(point, norm_vec, iso,
                          point - origin), 6, inter)
                mask[x, y, z] = check(borders, inter)


@njit
def intersection(p, n, iso, r_dir):
    """calculates intersection of given point with central plane through iso center

    Args:
        p (np.array): point of which intersection has to be calculated
        n (np.array): normal vector of cental plane through iso center
        iso (np.array): iso center position in 3d space
        r_dir (np.array): ray direction from origin to point

    Returns:
        [type]: [description]
    """
    ndotu = n.dot(r_dir)
    w = p - iso
    si = -n.dot(w) / ndotu
    inter = w + si * r_dir + iso
    return inter


@njit
def check(b, inter):
    """cehcks if point lies withing field or outside field

    Args:
        b (np.array): borders of field
        inter (np.array): intersection of point to origin line with central plane

    Returns:
        bool : boolean value to be filled inside the binary field mask
    """
    if within(b[0], inter[0]) and within(b[1], inter[1]) and within(b[2], inter[2]):
        return 1
    else:
        return 0


@njit
def within(nums, num_checked):
    """checks if number lies within 2 other number

    Args:
        nums (np.array): 2 numbers unsorted
        num_checked (float): number to be checked

    Returns:
        bool : True is number lies within and false if lies not within the 2 numbers
    """
    upper = nums.max()
    lower = nums.min()
    if num_checked >= lower and num_checked <= upper:
        return True
    else:
        return False


def plot_volume(volume):

    for i in range(volume.shape[2]):
        plt.imshow(volume[:, :, i])
        plt.show()


def calc_trainings_masks(fs, num_angles=8, SID=1435):
    """creates masks in training folder

    Args:
        fs (int): fieldsize in cm
        num_angles (int, optional): number of stops which are calcualted between 0 and 360. Defaults to 8.
        SID (int, optional): Surface Iso Center distance . Defaults to 1435 for MR-Linac.
    """
    fs = int(fs*10)  
    angles = np.linspace(0,360,num_angles, endpoint=False).astype("int")
    angles_str = [str(angle) for angle in angles]
    print("Angles: " + ", ".join(angles_str) + "\nFieldsize: " + str(int(fs/10)))

    for angle in angles:
        create_trainings_masks(SID, angle, fs)

if __name__ == "__main__":

    calc_trainings_masks(2)

