import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import _DIMENSION_NAME
from numpy.random import gamma
import torch
import matplotlib.animation as ani
from pprint import pprint
import pymedphys
import os
import pydicom


def animation(train_pixels, target_pixels, gif_name):
    eps = 1e-120
    ntrain = train_pixels/(train_pixels.max()+eps)
    ntarget = target_pixels/(target_pixels.max()+eps)
    fig = plt.figure(figsize=(8, 8))
    anim = plt.imshow((ntrain[0]+ntarget[0])/2)
    plt.grid(False)

    def update(i):
        anim.set_array((ntrain[i]+ntarget[i])/2)
        return anim,

    a = ani.FuncAnimation(fig, update, frames=range(
        len(train_pixels)), interval=200, blit=True)
    a.save(gif_name, writer=ani.PillowWriter(fps=24))


if __name__ == "__main__":

    ct = os.listdir("/Users/simongutwein/Downloads/CT")
    pseudo = os.listdir("/Users/simongutwein/Downloads/1_test_copy_pCT")

    cts = []
    for image in ct:
        dat = pydicom.dcmread("/Users/simongutwein/Downloads/CT/" + image)
        cts.append(dat.pixel_array)

    pseudos = []
    for image in pseudo:
        dat = pydicom.dcmread(
            "/Users/simongutwein/Downloads/1_test_copy_pCT/" + image)
        pseudos.append(dat.pixel_array)

    cts = np.array(cts).astype("float")
    cts -= 1024*np.ones_like(cts)
    pseudos = np.array(pseudos).astype("float")
    pseudos[pseudos < -200] = np.nan
    pseudos[pseudos > 200] = np.nan
    cts[cts < -200] = np.nan
    cts[cts > 200] = np.nan

    # plt.hist(np.ndarray.flatten(cts), bins=200, alpha=0.5, density=True)
    # plt.hist(np.ndarray.flatten(pseudos), bins=200, alpha=0.5, density=True)
    plt.hist(np.ndarray.flatten(cts), bins=200,
             alpha=0.5, cumulative=True, density=True)
    plt.hist(np.ndarray.flatten(pseudos), bins=200,
             alpha=0.5, cumulative=True, density=True)
    plt.legend(["original ct", "pseudo ct"])
    plt.xlabel("Houndsfield units")
    plt.ylabel("occurrences")

    plt.show()

    # target = torch.load(
    #     "/Users/simongutwein/Studium/Masterarbeit/test_data/p0_0/target_data.pt")
    # target = target.squeeze()

    # prediction = torch.load(
    #     "/Users/simongutwein/Studium/Masterarbeit/test_data/p0_0/target_data.pt")
    # prediction = prediction.squeeze()

    # print(target.max(), target.min())

    # prediction += np.random.rand(target.shape[0],
    #                              target.shape[1], target.shape[2]).astype(np.float32)
    # print(prediction.max(), prediction.min())

    # gamma_options = {
    #     'dose_percent_threshold': 3,
    #     'distance_mm_threshold': 3,
    #     'lower_percent_dose_cutoff': 20,
    #     'interp_fraction': 10,  # Should be 10 or more for more accurate results
    #     'max_gamma': 2,
    #     'random_subset': None,
    #     'local_gamma': True,
    #     'quiet': True
    # }

    # coords = (np.arange(target.shape[0]), np.arange(
    #     target.shape[1]), np.arange(target.shape[2]))

    # gamma_val = pymedphys.gamma(
    #     coords, np.array(target),
    #     coords, np.array(prediction),
    #     **gamma_options)

    # for i in range(gamma_val.shape[2]):
    #     plt.imshow(gamma_val[:, :, i])
    #     plt.show()
    #     plt.close()
