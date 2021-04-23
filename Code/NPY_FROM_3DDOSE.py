import numpy as np
import os
from READ_3DDOSE import read_3ddose_file, upscale
import matplotlib.pyplot as plt

dir_path = "/work/ws/nemo/tu_zxoys08-egs_dat-0/"
dose = "test/"
binary = "binary/"
dose_files = os.listdir(dir_path + dose)


for file_ in dose_files:
   
    target_filename = "_".join(file_.split(".")[0].split("_")[1:]) + ".npy"
    dose_vol = read_3ddose_file(dir_path + dose + file_)
    dose_vol = upscale(dose_vol, (512, 512, 110))
    print(file_, dose_vol.shape)

    #    for i in range(target_size.shape[2]):
    #        plt.imshow(dose_vol[:, :, i])
    #        plt.show()
    
    with open(dir_path + dose + target_filename, 'wb+') as fout:
        np.save(fout, dose_vol)

    with open(dir_path + dose + target_filename, 'rb') as fin:
        dose_vol = np.load(fin)
    
    with open(dir_path + binary + target_filename, 'rb') as fin:
        binary_vol = np.load(fin)

    for slice_ in range(dose_vol.shape[2]):
        plt.imshow(dose_vol[:, :, slice_])
        plt.imshow(binary_vol[:, :, slice_])
        plt.show()
