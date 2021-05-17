from pydicom import dcmread
import numpy as np
from radio_depth import *
import os
import sys
import warnings

def create_radio_depth(radio_depth_volume, ct, origin, angle, files_out, target_name):

    log_file = "/home/baumgartner/sgutwein84/training_data/log.txt"

    progress = -5
    for x in range(ct.shape[0]):
        if x % int(ct.shape[0]/20) == 0:
            progress += 5
            print("Radio Depth           || Progress: " + str(progress) + "%")

        for y in range(ct.shape[1]):
            for z in range(ct.shape[2]):
                voxel = np.array([x, y, z])

                try:
                    curr_ray = Ray(origin, voxel, ct)

                except Exception as ex:
                    if not type(ex).__name__ == "KeyboardInterrupt":
                        with open(log_file, "a") as fout:
                            fout.writelines(["Angle" + str(angle) +"(" + str(x) + "," + str(y) + "," + str(z) + ") , " + str(ex)])
                                 
                radio_depth_volume[x, y, z] = curr_ray.path

    with open(files_out + target_name + ".npy", "wb+") as fout:
        np.save(fout, radio_depth_volume)

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

if __name__ == "__main__":

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    patient = "p"
    in_folder = "/home/baumgartner/sgutwein84/training_data/training/ct/" + patient + ".npy"
    out_folder = "/home/baumgartner/sgutwein84/training_data/training/radio_depth/"

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    with open(in_folder, 'rb') as fin:
        ct_volume = np.load(fin)

    print(f"CT Volume Shape: {ct_volume.shape}")

    for angle in np.linspace(0,360,8, endpoint=False):

        iso = np.array([ct_volume.shape[0]/2-1,
                    ct_volume.shape[1]/2-1, ct_volume.shape[2]/2-1]).astype("float")

        origin = get_origin(1435, angle, iso)
        
        print(f"Origin Position at: {origin}")

        radio_depth = np.zeros(ct_volume.shape)

        target_name = patient + "_" + str(int(angle))

        if not os.path.isfile(out_folder + target_name + ".npy"):
            print(f"Creating {target_name}.npy")
            create_radio_depth(radio_depth, ct_volume, origin, angle, out_folder, target_name)
        else:
            print(out_folder + target_name + ".npy, already exists!")


    

