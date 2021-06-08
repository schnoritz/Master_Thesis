import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from pprint import pprint
import os
import pickle
from model import Dose3DUNET
from resize_right import resize
from interp_methods import linear

if __name__ == "__main__":

    #dat= torch.ones((1,5,512,512,110))
    dat = torch.load(
        "/Users/simongutwein/home/baumgartner/sgutwein84/container/training_data/p_0/training_data.pt")
    dat = torch.unsqueeze(dat, 0)
    resized = resize(dat, out_shape=(1, 5, 512, 512, 111),
                     interp_method=linear,  support_sz=2)
    print(resized.shape)

    for i in range(111):
        plt.imshow(resized[0, 0, :, :, i])
        plt.show()
