import torch
from npy_from_3ddose import npy_from_3ddose
import matplotlib.pyplot as plt

def comapre(dose, binary):

    dose_vol = npy_from_3ddose(dose)
    binary_vol = torch.load(binary)

    for i in range(dose_vol.shape[2]):
        plt.imshow(dose_vol[:,:,i])
        plt.imshow(binary_vol[:,:,i], alpha=0.5)
        plt.show()
        plt.close()

    for i in range(dose_vol.shape[0]):
        plt.imshow(dose_vol[i, :, :])
        plt.imshow(binary_vol[i, :, :], alpha=0.9)
        plt.show()
        plt.close()
        if i == 100:
            break

if __name__ == "__main__":

    dose = "/home/baumgartner/sgutwein84/container/output/p_0_4x4/p_0_4x4_1E08.3ddose"
    binary = "/home/baumgartner/sgutwein84/container/output/p_0_4x4/binary_mask_0_4x4.pt"

    comapre(dose, binary)