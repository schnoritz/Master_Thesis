
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_paths(root_path, files, idx):
    ct_path = root_path + files.iloc[idx, 2]
    depth_path = root_path + files.iloc[idx, 3]
    binary_path = root_path + files.iloc[idx, 4]
    center_path = root_path + files.iloc[idx, 5]
    source_path = root_path + files.iloc[idx, 6]
    target_path = root_path + files.iloc[idx, 7]

    return [ct_path, depth_path, binary_path, center_path, source_path], target_path


def read_arrays(tr_p, tar_p):

    arrays = []
    for array in tr_p:
        with open(array, 'rb') as fin:
            arrays.append(np.load(fin))

    with open(tar_p, 'rb') as fin:
        target_array = np.load(fin)

    return arrays, target_array

if __name__ == "__main__":

    root_dir = "/home/baumgartner/sgutwein84/container/training_data/training/"
    image_dir = "/home/baumgartner/sgutwein84/container/training_data/test/"
    csv_dir = root_dir + "csv_files.xls"
    files = pd.read_excel(csv_dir)

    for i in range(72):
        tr_paths, tar_path = get_paths(root_dir, files, i)
        training, target = read_arrays(tr_paths, tar_path)

        fig, axs = plt.subplots(1, 6, figsize=(24, 4))
        j = 0
        for mask in training:
            axs[j].imshow(mask[:, :, mask.shape[2]//2])
            j += 1

        axs[5].imshow(target[:, :, target.shape[2]//2])
            
        name = image_dir + "image" + str(i) + ".png"

        if os.path.isfile(name):
            os.remove(name)

        plt.close()

        plt.savefig(name)
        print("Image saved!")

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        idx = int(training[0].shape[2]//2)
        axs[0].imshow(training[0][:, :, idx]) #ct
        axs[0].imshow(training[1][:, :, idx],alpha=0.5)  # radio_depth
        
        axs[1].imshow(training[2][:, :, idx])  # binary
        axs[1].imshow(target[:, :, idx], alpha=0.5)  # dose

        axs[2].imshow(training[0][:, :, idx])  # ct
        axs[2].imshow(target[:, :, idx], alpha=0.5)  # dose

        name = image_dir + "image_overlay" + str(i) + ".png"

        if os.path.isfile(name):
            os.remove(name)

        plt.savefig(name)
        print("Overlay saved!")

        plt.close()


