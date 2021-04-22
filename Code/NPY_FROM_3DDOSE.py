import numpy as np
import os
from READ_3DDOSE import read_3ddose_file, upscale
import matplotlib.pyplot as plt

dir_path = "/work/ws/nemo/tu_zxoys08-egs_dat-0/"
dose = "test/"
binary = "binary/"
dose_files = os.listdir(dir_path + dose)


for file_ in dose_files:
   
    dose_vol = read_3ddose_file(dir_path + dose + file_)
    target_size = np.empty((512, 512, 110))
    dose_vol = upscale(dose_vol, target_size)
    print(file_, dose_vol.shape)
    target_filename = file_.split(".")[0] + ".npy"
    #    for i in range(target_size.shape[2]):
    #        plt.imshow(dose_vol[:, :, i])
    #        plt.show()
    
    open(dir_path + dose + target_filename, 'wb+').close()
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
