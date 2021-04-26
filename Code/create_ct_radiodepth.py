from pydicom import dcmread
import numpy as np
import os
import re
from natsort import natsorted, ns
from create_trainings_masks import plot_volume, get_origin
from RADIO_DEPTH import *

def create_ct_arr(files_in, files_out):

    files = [i for i in os.listdir(files_in) if not i.startswith(".")]
    ct = np.empty((512, 512, len(files)))
    i = -1
    for file_ in natsorted(files, key=lambda y: y.lower()):
        i += 1
        dat = dcmread(files_in + file_)
        ct[:, :, i] = dat.pixel_array

    with open(files_out + "p.npy", "wb+") as fout:
        np.save(fout, ct)

    return ct

def create_radio_depth(radio_depth_volume, ct, origin, angle, files_out):

    log_file = "/Users/simongutwein/Studium/Masterarbeit/log.txt"

    progress = -5
    for x in range(ct.shape[0]):
        if x % int(ct.shape[0]/20) == 0:
            progress += 5
            print("Radio Depth           || Progress: " + str(progress) + "%")

        for y in range(ct.shape[1]):
            for z in range(ct.shape[2]):
                voxel = np.array([x, y, z])
                voxel = np.array([0, 0, 0])
                try:
                    curr_ray = ray(origin, voxel, ct)
                except:
                    with open(log_file, "a") as fout:
                        fout.writelines(["(" + str(x) + "," + str(y) + "," + str(z) + ") "])
                radio_depth_volume[x, y, z] = curr_ray.path

    with open(files_out + str(angle) + ".npy", "wb+") as fout:
        np.save(fout, radio_depth_volume)

if __name__ == "__main__":

    dcm_files = "/work/ws/nemo/tu_zxoys08-egs_dat-0/training/dcm/p/"
    target_dir = "/work/ws/nemo/tu_zxoys08-egs_dat-0/training/ct/p/"

    with open("/Users/simongutwein/Studium/Masterarbeit/p.npy", 'rb') as fin:
        ct_volume = np.load(fin)
    #ct_volume = create_ct_arr(dcm_files, target_dir)

    #print(ct_volume.shape)

    angle = 180
    radio_depth_path = "/work/ws/nemo/tu_zxoys08-egs_dat-0/training/radiological_depth/p/"
    
    iso = np.array([ct_volume.shape[0]/2-1,
                   ct_volume.shape[1]/2-1, ct_volume.shape[2]/2-1]).astype("float")
    origin = get_origin(1435, angle, iso)
    print(origin)
    radio_depth = np.empty_like(ct_volume)
    create_radio_depth(radio_depth, ct_volume, origin, angle, radio_depth_path)

    #plot_volume(radio_depth)


    

