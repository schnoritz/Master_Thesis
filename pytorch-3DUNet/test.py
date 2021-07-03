from torch.random import get_rng_state
from torch.utils.data import DataLoader
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn as nn
from utils import save_model
from model import Dose3DUNET
from dataset import setup_loaders
from pprint import pprint
import random
from time import time
import dataqueue
from pynvml import *


if __name__ == "__main__":

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    segment = "p0_0"

    device = torch.device("cuda:0")

    target_dose = torch.load(
        f"/home/baumgartner/sgutwein84/container/prostate_training_data/{segment}/target_data.pt")
    target_dose = target_dose.squeeze()

    masks = torch.load(
        f"/home/baumgartner/sgutwein84/container/prostate_training_data/{segment}/training_data.pt")
    masks = torch.unsqueeze(masks, 0)

    model = Dose3DUNET()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(torch.cuda.is_available(), torch.cuda.device_count)

    checkpoint = torch.load(
        "/home/baumgartner/sgutwein84/container/pytorch-3DUNet/experiments/bs4_ps64/UNET_48.pt", map_location="cuda")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = nn.DataParallel(model)
    model.float()

    model = model.to(device)
    model.eval()
    preds = []
    ps = 128
    for i in range(0, 512, ps):
        for j in range(0, 512, ps):

            mask = masks[0, :, j:j+ps, i:i+ps, :]
            mask = mask.unsqueeze(0)
            torch.cuda.empty_cache()
            mask = mask.to(device)
            print(mask.shape)
            preds.append(model(mask).cpu().detach().squeeze())
            del mask
            torch.cuda.empty_cache()

    print(len(preds))
    stacked = []
    for i in range(4):
        stacked.append(torch.cat(preds[i*4:(i+1)*4]))

    print(len(stacked), stacked[0].shape)
    end = torch.cat(stacked, 1)

    print(end.shape)

    for i in range(end.shape[2]):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(end[:, :, i])
        ax[1].imshow(target_dose[:, :, i])
        plt.show()
        plt.savefig(
            f"/home/baumgartner/sgutwein84/container/test/end_4_64_{i}.png")
        plt.close(fig)
    # pred = pred.cpu().detach().numpy()
    # pred = pred.squeeze()

    # masks = masks.cpu().detach().numpy()
    # masks = masks.squeeze()

    # torch.save(pred, "/home/baumgartner/sgutwein84/container/test/pred.pt")
    # torch.save(dose, "/home/baumgartner/sgutwein84/container/test/true.pt")

    # for i in range(8):
    #     fig, ax = plt.subplots(1, 4)
    #     ax[0].imshow(masks[0, :, :, i])
    #     ax[1].imshow(dose[:, :, i])
    #     ax[2].imshow(pred[:, :, i])
    #     ax[3].imshow(dose[:, :, i]-pred[:, :, i])
    #     plt.savefig(
    #         f"/home/baumgartner/sgutwein84/container/test/pred_{i}")
    #     plt.close()

    # test = Test()

    # for i in test:
    #     for j in test:
    #         print(i, j)

    # subject_list = ["/Users/simongutwein/Studium/Masterarbeit/test_data/" + x for x in os.listdir(
    #     "/Users/simongutwein/Studium/Masterarbeit/test_data") if not x.startswith(".")]

    # print(len(subject_list))

    # train_set, test_set = get_train_test_sets(subject_list, 0.8)
    # print("Train-Set")
    # pprint(train_set)
    # print()
    # print("Test-Set")
    # pprint(test_set)
    # print("\n")

    # # Number of total Samples: 469
    # # Training-Samples: 376
    # # Test-Samples: 93

    # batch_size = 32
    # segments_per_queue = 2

    # # 128 samples aus 128/32 segmenten laden -> shufflen -> in 4er batches yielden

    # overall_start = time()
    # total_number = 0
    # epochs = 3

    # train_queue = DataQueue(train_set, batch_size,
    #                         segments_per_queue, ps=32, sps=2000)
    # test_queue = DataQueue(test_set, batch_size,
    #                        segments_per_queue, ps=32, sps=2000)

    # for epoch in range(epochs):
    #     print("Epoch", epoch)
    #     for num_queue, (train_patches, target_patches) in enumerate(train_queue):
    #         print("train", num_queue)
    #         for num, (train, target) in enumerate(zip(train_patches, target_patches)):
    #             total_number += train.shape[0]
    #             # print(train.shape)

    #             #print("step: train")
    #         if (num_queue+1) % 5 == 0:
    #             print("Validate")
    #             for (test_patches, target_patches_test) in test_queue:
    #                 for num, (test, target) in enumerate(zip(test_patches, target_patches_test)):
    #                     pass
    #                     #print("step: validate")
    #                     # print(test.shape)
    # print(
    #     f"loading for all took: {time()-overall_start} seconds for {total_number} Patches\n")

    # # for i in train_loader:
    #     print(i['trainings_data']['data'].shape)
    #     print(i['target_data']['data'].shape)

    # model = Dose3DUNET().float()
    # optimizer = torch.optim.Adam(model.parameters())

    # save_dir = "/Users/simongutwein/Studium/Masterarbeit/save/"

    # save_model(
    #     model=model,
    #     optimizer=optimizer,
    #     train_loss=10,
    #     test_loss=15,
    #     save_dir=save_dir,
    #     epoch=100,
    #     save=True)

    # model_info = torch.load(
    #     "/Users/simongutwein/Studium/Masterarbeit/save/UNET_epoch100.pt")

    # model.load_state_dict(model_info['model_state_dict'])
    # optimizer.load_state_dict(model_info['optimizer_state_dict'])
    # curr_train_loss = model_info['train_loss']
    # curr_test_loss = model_info['test_loss']
    # epoch = model_info['epoch']

    # model.eval()

    # # do something here

    # model.test()

    # # start test routine here

    pass
