import numpy as np
import os
from READ_3DDOSE import read_3ddose_file, upscale
import matplotlib.pyplot as plt

def npy_from_3ddose(file_path, target_path=None, target_volume=(512, 512, 110), save=False):
   
    target_filename = file_path.split("/")[-1].split(".")[0] + ".npy"
    
    if target_path[-1] != "/":
        target_path += "/"

    if not os.path.isfile(target_path + target_filename) or save == False:
        dose_vol = read_3ddose_file(file_path)
        dose_vol = upscale(dose_vol, target_volume)

    if save:

        if not os.path.isfile(target_path + target_filename):
            with open(target_path + target_filename, 'wb+') as fout:
                np.save(fout, dose_vol)
                print(target_path + target_filename + " saved!")
        else:
            print(target_path + target_filename + " already exists!")

    else: 

        return dose_vol

if __name__ == "__main__":

    dose_file = "/home/baumgartner/sgutwein84/training_data/3ddose/"
    output_path = "/home/baumgartner/sgutwein84/training_data/training/target"
    files = [x for x in os.listdir(dose_file) if not x.startswith(".")]
    for file_ in files:
        npy_from_3ddose(dose_file + file_, output_path, save=True)
    
    
