import matplotlib.pyplot as plt
from numba.core.registry import TargetRegistry
import numpy as np
from numpy.core.defchararray import startswith
import torch
import random
from pprint import pprint
import os
import pickle
from pt_3ddose import dose_to_pt
from time import time
from pt_ct import convert_ct_array
from scipy import ndimage
import matplotlib.animation as ani
from pydicom import dcmread
from pprint import pprint


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

    training = torch.load(
        "/Users/simongutwein/Studium/Masterarbeit/p0_0/training_data.pt")

    target = torch.load(
        "/Users/simongutwein/Studium/Masterarbeit/p0_0/target_data.pt")

    print(target.max())
    print(training[0].max(), training[1].max(),
          training[2].max(), training[3].max(), training[4].max())

    training[0] = training[0]
    training[1] = training[1]/3000
    training[2] = training[2]/3000
    training[3] = training[3]/(1.171875)
    training[4] = training[4]/(1435/1.171875)
    target = target / 1E-17

    print(target.max())
    print(training[0].max(), training[1].max(),
          training[2].max(), training[3].max(), training[4].max())

    mosaic = """
    AABBCCDDEEFF
    .HH..II..JJ.
    """
    cmap = "jet"
    fig = plt.figure(constrained_layout=True, figsize=(60, 20))
    ax_dict = fig.subplot_mosaic(mosaic)
    ax_dict["A"].imshow(training[0, :, :, 37], cmap=cmap)
    ax_dict["B"].imshow(training[1, :, :, 37], cmap=cmap)
    ax_dict["C"].imshow(training[2, :, :, 37], cmap=cmap)
    ax_dict["D"].imshow(training[3, :, :, 37], cmap=cmap)
    ax_dict["E"].imshow(training[4, :, :, 37], cmap=cmap)
    ax_dict["F"].imshow(target[0, :, :, 37], cmap=cmap)
    ax_dict["H"].imshow(training[0, :, :, 37], cmap=cmap)
    ax_dict["H"].imshow(target[0, :, :, 37], cmap=cmap, alpha=0.7)
    ax_dict["I"].imshow(training[1, :, :, 37], cmap=cmap)
    ax_dict["I"].imshow(training[0, :, :, 37], cmap=cmap, alpha=0.8)
    ax_dict["J"].imshow(training[1, :, :, 37], cmap=cmap)
    ax_dict["J"].imshow(training[2, :, :, 37], cmap=cmap, alpha=0.8)

    plt.show()

    # slices = "/Users/simongutwein/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/p3"
    # dose_mask = torch.randn((512, 512, 131))
    # stack = convert_ct_array(slices, target_size=dose_mask.shape, tensor=True)
    # print(stack.shape, stack.dtype)

    # model = Dose3DUNET().float()
    # model.load_state_dict(torch.load(
    #     "/home/baumgartner/sgutwein84/container/pytorch-3DUNet/saved_models/UNET_epoch653.pth"))
    # mask = torch.load(
    #     "/home/baumgartner/sgutwein84/container/training_data20210522/p_22/training_data.pt")
    # target = torch.load(
    #     "/home/baumgartner/sgutwein84/container/training_data20210522/p_22/target_data.pt")

    # # an device senden
    # device = torch.device("cuda")

    # mask = torch.unsqueeze(mask, 0)
    # mask = mask.float()
    # mask.to(device)
    # print(f"Input shape: {mask.shape}")
    # start = time()
    # pred = model(mask)
    # print(f"Prediction took {time()-start} seconds")
    # print(f"Prediction shape: {pred.shape}")

    # pred = pred.detach().numpy()

    # for i in range(110):
    #     plt.imshow(pred[0, 0, :, :, i])
    #     plt.show()
    #     plt.savefig(
    #         f"/home/baumgartner/sgutwein84/container/logs/test0/pred_{i}")
    #     plt.close()
    #     plt.imshow(target[0, :, :, i])
    #     plt.show()
    #     plt.savefig(
    #         f"/home/baumgartner/sgutwein84/container/logs/test0/target_{i}")
    #     plt.close()

    # segments = [f"p_{i}" for i in range(40)]

    # training = torch.randn((5, 512, 512, 110))
    # target = torch.randn((1, 512, 512, 110))

    # for seg in segments:
    #     os.mkdir(
    #         f"/home/baumgartner/sgutwein84/container/trainings_data/{seg}"
    #     )

    #     torch.save(
    #         training,
    #         f'/home/baumgartner/sgutwein84/container/trainings_data/{seg}/training_data.pt'
    #     )

    #     torch.save(
    #         target,
    #         f'/home/baumgartner/sgutwein84/container/trainings_data/{seg}/target_data.pt'
    #     )

    # epochs = []
    # for epoch in range(100):

    #     train_loss = np.round(random.uniform(0.5, 10)*(1/(epoch+1)), 4)
    #     test_loss = np.round(train_loss + random.uniform(0, 0.5), 4)

    #     epochs.append({
    #         "epoch": epoch+1,
    #         "train_loss": train_loss,
    #         "test_loss": test_loss
    #     })

    #     if epoch > 5:
    #         top5 = [sorted(epochs, key=lambda k: k['test_loss'])[i]["epoch"]
    #                 for i in range(5)]
    #         print(top5)

    # test_losses = [epochs[i]["test_loss"] for i in range(len(epochs))]
    # train_losses = [epochs[i]["train_loss"] for i in range(len(epochs))]
    # plt.plot(test_losses, color="red")
    # plt.plot(train_losses,  color="blue")
    # plt.vlines(np.array(top5)-1, ymin=0, ymax=1, color="green")
    # plt.legend()
    # plt.show()

    # pprint(sorted(epochs, key=lambda k: k['test_loss'])[:5])

    # trainings_masks = "/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/p_17/training_data.pt"
    # target_masks = "/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/p_17/target_data.pt"

    # training = torch.load(trainings_masks)
    # target = torch.load(target_masks)

    # fig, ax = plt.subplots(1, 6, figsize=(48, 8))
    # ax[0].imshow(training[0, :, :, 37])
    # ax[1].imshow(training[1, :, :, 37])
    # ax[2].imshow(training[2, :, :, 37])
    # ax[3].imshow(training[3, :, :, 37])
    # ax[4].imshow(training[4, :, :, 37])
    # ax[5].imshow(target[:, :, 37])
    # plt.show()

    # fig, ax = plt.subplots(1, 4, figsize=(32, 8))
    # ax[0].imshow(training[0, :, :, 37])
    # ax[0].imshow(target[:, :, 37], alpha=0.5)
    # ax[1].imshow(training[1, :, :, 37])
    # ax[1].imshow(training[2, :, :, 37], alpha=0.5)
    # ax[2].imshow(training[1, :, :, 37])
    # ax[2].imshow(target[:, :, 37], alpha=0.5)
    # ax[3].imshow(training[0, :, :, 37])
    # ax[3].imshow(training[3, :, :, 37], alpha=0.1)
    # plt.show()
