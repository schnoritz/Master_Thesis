from torch.random import get_rng_state
from torch.utils.data import DataLoader
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from utils import save_model
from model import Dose3DUNET
from dataset import setup_loaders
from pprint import pprint
import random
from time import time


def get_train_test_sets(dataset, train_fraction):

    num = len(dataset)

    train_n = int(np.ceil(num*train_fraction))
    test_n = num - train_n

    assert train_n + \
        test_n == num, f"Splitting of Training-Data does not match! num={num}. train_n={train_n}, test_n={test_n}"

    print(
        f"Number of total Samples: {num}\nTraining-Samples: {train_n}\nTest-Samples: {test_n}")

    random.shuffle(dataset)
    train_set, test_set = dataset[:train_n], dataset[train_n:]

    return train_set, test_set


class DataQueue():
    def __init__(self, seg, bs, spq, ps, sps):
        self.seg = seg
        self.bs = bs
        self.ps = ps
        self.spq = spq
        self.sps = sps
        self.curr = 0

    def load_q(self, segs):
        self.sps
        data = []

        for seg in segs:
            data.append(
                (torch.load(f"{seg}/training_data.pt"), torch.load(f"{seg}/target_data.pt")))

        train_patches = []
        target_patches = []
        for masks, target in data:
            idxs = self.get_sample_idx(self.sps, masks)
            for i in idxs:
                train_patches.append(
                    masks[:, int(i[0]-self.ps/2):int(i[0]+self.ps/2), int(i[1]-self.ps/2):int(i[1]+self.ps/2), int(i[2]-self.ps/2):int(i[2]+self.ps/2)])
                target_patches.append(
                    target[:, int(i[0]-self.ps/2):int(i[0]+self.ps/2), int(i[1]-self.ps/2):int(i[1]+self.ps/2), int(i[2]-self.ps/2):int(i[2]+self.ps/2)])

        train_patches = torch.stack(tuple(train_patches))
        target_patches = torch.stack(tuple(target_patches))

        train = []
        target = []
        curr = 0
        for i in range(int(train_patches.shape[0]/self.bs)):
            curr += self.bs
            train.append(train_patches[curr-self.bs:curr, :, :, :])
            target.append(target_patches[curr-self.bs:curr, :, :, :])

        return train, target

    def get_sample_idx(self, sps, masks):

        binary = np.copy(masks[0, int(self.ps/2):masks[0].shape[0]-int(self.ps/2),
                               int(self.ps/2):masks[0].shape[1]-int(self.ps/2),
                               int(self.ps/2):masks[0].shape[2]-int(self.ps/2)])

        binary[binary > 0] = 5
        binary[binary == 0] = 1
        binary = np.pad(binary, ((int(self.ps/2), int(self.ps/2)),
                                 (int(self.ps/2), int(self.ps/2)),
                                 (int(self.ps/2), int(self.ps/2))), 'constant')

        num = random.randint(1, 5)
        if num == 1:
            idxs = []
            for _ in range(sps):
                idxs.append(np.array([random.randint(int(self.ps/2), binary.shape[0]-int(self.ps/2)),
                            random.randint(
                                int(self.ps/2), binary.shape[1]-int(self.ps/2)),
                            random.randint(int(self.ps/2), binary.shape[2]-int(self.ps/2))]))
        else:
            idxs = np.argwhere(binary >= num)
            np.random.shuffle(idxs)

        return idxs[:sps]

    def __len__(self):
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr >= len(self.seg):
            raise StopIteration
        self.curr += self.spq
        train, target = self.load_q(self.seg[self.curr-self.spq:self.curr])

        return train, target


if __name__ == "__main__":

    subject_list = ["/Users/simongutwein/Studium/Masterarbeit/test_data/" + x for x in os.listdir(
        "/Users/simongutwein/Studium/Masterarbeit/test_data") if not x.startswith(".")]

    print(len(subject_list))

    train_set, test_set = get_train_test_sets(subject_list, 0.8)
    print("Train-Set")
    pprint(train_set)
    print()
    print("Test-Set")
    pprint(test_set)
    print("\n")

    # Number of total Samples: 469
    # Training-Samples: 376
    # Test-Samples: 93

    batch_size = 16
    segments_per_queue = 4

    # 128 samples aus 128/32 segmenten laden -> shufflen -> in 4er batches yielden

    data = DataQueue(train_set, batch_size, segments_per_queue, ps=32, sps=503)
    overall_start = time()
    total_number = 0
    for (train_patches, target_patches) in data:
        for num, (train, target) in enumerate(zip(train_patches, target_patches)):
            total_number += train.shape[0]
            print("Batchnumber: ", num+1)
            print(train.shape)
            print(target.shape)
            # plt.imshow(train[0, 0, :, :, 16])
            # plt.show()
            # plt.imshow(target[0, 0, :, :, 16])
            # plt.show()

    print(
        f"loading for all took: {time()-overall_start} seconds for {total_number} Patches\n")

    # for i in train_loader:
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
