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
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import pymedphys
import sys


if __name__ == "__main__":

    # gamma_val = np.load("/Users/simongutwein/Studium/Masterarbeit/gamma.npy")

    # print(np.nanmin(gamma_val), np.nanmax(gamma_val))
    # dat = ~np.isnan(gamma_val)
    # dat2 = ~np.isnan(gamma_val[gamma_val <= 1])
    # all = np.count_nonzero(dat)
    # true = np.count_nonzero(dat2)
    # print(all, true, true/all)

    # trues = np.copy(gamma_val)
    # idx = gamma_val < 1
    # trues[idx] = np.nan

    # for i in range(trues.shape[2]):
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(gamma_val[:, :, i])
    #     ax[1].imshow(trues[:, :, i])
    #     plt.show()
    root_path = "/home/baumgartner/sgutwein84/container/pytorch-3DUNet/experiments/bs256_ps16_epoch50" + "/"

    for model_spec in [x for x in os.listdir(root_path) if not x.startswith(".") and not "pkl" in x]:

        segment = "p10_12"
        model_path = root_path + model_spec
        model_name = "/".join(model_path.split("/")[-2:])

        print("Segment           :", segment)
        print("Model Name        :", model_name)

        device = torch.device("cuda")

        target_dose = torch.load(
            f"/home/baumgartner/sgutwein84/container/prostate_training_data/{segment}/target_data.pt")
        target_dose = target_dose.squeeze()

        masks = torch.load(
            f"/home/baumgartner/sgutwein84/container/prostate_training_data/{segment}/training_data.pt")
        masks = torch.unsqueeze(masks, 0)

        model = Dose3DUNET()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # checkpoint = torch.load(
        #     "/home/baumgartner/sgutwein84/container/pytorch-3DUNet/experiments/bs256_ps16_epoch50/UNET_95.pt", map_location="cpu")
        checkpoint = torch.load(model_path, map_location="cpu")
        val_loss = np.round(checkpoint["val_loss"], 4)
        print(f"Val Loss of Model : {val_loss}")

        model.load_state_dict(checkpoint["model_state_dict"])
        model = nn.DataParallel(model)
        model.float()

        torch.cuda.empty_cache()
        with torch.no_grad():
            model = model.to(device)
            model.eval()
            preds = []
            ps = 16
            for i in range(0, masks.shape[4], ps):
                mask = masks[0, :, :, :, i:i+ps]
                if mask.shape[3] < ps:
                    num = int(ps-mask.shape[3])
                    added = torch.zeros(
                        mask.shape[0], mask.shape[1], mask.shape[2], ps-mask.shape[3])
                    mask = torch.cat((mask, added), 3)

                mask = mask.unsqueeze(0)
                torch.cuda.empty_cache()
                mask = mask.to(device)

                pred = model(mask)
                detached = pred.to('cpu')

                detached = detached.detach().squeeze()
                preds.append(detached)
                del mask
                del pred
                del detached
                torch.cuda.empty_cache()

            end = torch.cat(preds, 2)
            end = end[:, :, :(-num)]
            # print(end.shape, target_dose.shape)

        # plt.imshow(end[:, :, 39])
        # plt.savefig(
        #     f"/Users/simongutwein/home/baumgartner/sgutwein84/container/test/{model_spec}")
        # for i in range(target_dose.shape[2]):
        #     fig, ax = plt.subplots(1, 2)
        #     ax[0].imshow(end[:, :, i])
        #     ax[1].imshow(target_dose[:, :, i])
        #     plt.show()
        #     plt.savefig(
        #         f"/home/baumgartner/sgutwein84/container/test/out{i}.png")
        #     plt.close(fig)

        # for i in range(200, 300):
        #     fig, ax = plt.subplots(1, 2)
        #     ax[0].imshow(end[i, :, :])
        #     ax[1].imshow(target_dose[i, :, :])
        #     plt.show()
        #     plt.savefig(
        #         f"/home/baumgartner/sgutwein84/container/test/out_x{i}.png")
        #     plt.close(fig)
        # print("saved")
        # sys.stdout.flush()

        gamma_options = {
            'dose_percent_threshold': 3,
            'distance_mm_threshold': 3,
            'lower_percent_dose_cutoff': 20,
            'interp_fraction': 5,  # Should be 10 or more for more accurate results
            'max_gamma': 1.1,
            'ram_available': 2**37,
            'quiet': True,
            'local_gamma': False,
            'random_subset': 20000
        }

        coords = (np.arange(0, 1.17*target_dose.shape[0], 1.17), np.arange(
            0, 1.17*target_dose.shape[1], 1.17), np.arange(0, 3*target_dose.shape[2], 3))

        start = time()
        gamma_val = pymedphys.gamma(
            coords, np.array(target_dose),
            coords, np.array(end),
            **gamma_options)
        #print("gamma took", time()-start)

        #np.save("/home/baumgartner/sgutwein84/container/test/gamma.npy", gamma_val)

        #print(np.nanmin(gamma_val), np.nanmax(gamma_val))
        dat = ~np.isnan(gamma_val)
        dat2 = ~np.isnan(gamma_val[gamma_val <= 1])
        all = np.count_nonzero(dat)
        true = np.count_nonzero(dat2)
        print(all, true, np.round((true/all)*100, 2), "%\n\n")

    # trues = np.copy(gamma_val)
    # idx = gamma_val < 1
    # trues[idx] = np.nan

    # for i in range(trues.shape[2]):
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(np.log(gamma_val[:, :, i]))
    #     ax[1].imshow(np.log(trues[:, :, i]))
    #     plt.savefig(
    #         f"/home/baumgartner/sgutwein84/container/test/gamma{i}.png")
    #     plt.close(fig)

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
