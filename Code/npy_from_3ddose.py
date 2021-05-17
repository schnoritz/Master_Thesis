import numpy as np
import os
from read_3ddose import read_3ddose_file, upscale
import matplotlib.pyplot as plt

def npy_from_3ddose(file_path, target_path=None, target_volume=(512, 512, 110)):

    if target_path:

        target_filename = "_".join(file_path.split("/")[-1].split(".")[0].split("_")[:-1]) + ".npy"

        if target_path[-1] != "/":
            target_path += "/"

        if not os.path.isfile(target_path + target_filename):

            dose_vol = read_3ddose_file(file_path)
            dose_vol = upscale(dose_vol, target_volume)

            with open(target_path + target_filename, 'wb+') as fout:
                np.save(fout, dose_vol)
                print(target_path + target_filename + " saved!")
        else:
            print(target_path + target_filename + " already exists!")

    else:

        dose_vol = read_3ddose_file(file_path)
        dose_vol = upscale(dose_vol, target_volume)

        return dose_vol

if __name__ == "__main__":

    dose_file = "/home/baumgartner/sgutwein84/container/training_data/3ddose/"
    output_path = "/home/baumgartner/sgutwein84/container/training_data/training/target"

    files = [x for x in os.listdir(dose_file) if not x.startswith(".") and x.split(".")[-1] == "3ddose"]

    for file_ in files:
        npy_from_3ddose(dose_file + file_, output_path)
