import torch
import os
import numpy as np
import random
from numpy.random import randint
from time import time
import sys

import matplotlib.pyplot as plt


class DataQueue():

    def __init__(
            self, segment_list, batch_size, segments_per_queue, patch_size, patches_per_segment,
            sampling_scheme="equal"):
        self.segment_list = segment_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.segments_per_queue = segments_per_queue
        self.patches_per_segment = patches_per_segment
        self.sampling_scheme = sampling_scheme
        self.available_segments = self.segment_list.copy()

    def reset_queue(self):
        self.available_segments = self.segment_list.copy()

    def load_queue(self):

        if self.segments_per_queue > len(self.available_segments):
            self.reset_queue()

        curr_segments = random.sample(
            self.available_segments, self.segments_per_queue)

        for seg in curr_segments:
            self.available_segments.remove(seg)

        self.train, self.target = self.create_queue(curr_segments)

    def create_queue(self, segs):

        data = []

        for seg in segs:
            data.append(
                (torch.load(f"{seg}/training_data.pt"), torch.load(f"{seg}/target_data.pt")))

        train_patches = []
        target_patches = []
        hp = self.patch_size/2
        for masks, target in data:

            idxs = self.get_sample_idx(masks)
            if self.patch_size >= masks.shape[-1]:
                for i in idxs:
                    train_patches.append(
                        masks[:, int(i[0]-hp):int(i[0]+hp), int(i[1]-hp):int(i[1]+hp), :])
                    target_patches.append(
                        target[:, int(i[0]-hp):int(i[0]+hp), int(i[1]-hp):int(i[1]+hp), :])
            else:
                for i in idxs:
                    train_patches.append(
                        masks[:, int(i[0]-hp):int(i[0]+hp), int(i[1]-hp):int(i[1]+hp), int(i[2]-hp):int(i[2]+hp)])
                    target_patches.append(
                        target[:, int(i[0]-hp):int(i[0]+hp), int(i[1]-hp):int(i[1]+hp), int(i[2]-hp):int(i[2]+hp)])

        all = list(zip(train_patches, target_patches))
        random.shuffle(all)
        train_patches, target_patches = zip(*all)

        train_patches = torch.stack(tuple(train_patches))
        target_patches = torch.stack(tuple(target_patches))

        train = []
        target = []
        curr = 0
        for i in range(int(np.ceil(train_patches.shape[0]/self.batch_size))):
            curr += self.batch_size
            train.append(train_patches[curr-self.batch_size:curr, :, :, :])
            target.append(target_patches[curr-self.batch_size:curr, :, :, :])

        return train, target

    def get_sample_idx(self, masks):

        hp = int(self.patch_size/2)
        binary = np.copy(masks[0,
                               hp:masks[0].shape[0]-hp,
                               hp:masks[0].shape[1]-hp,
                               hp:masks[0].shape[2]-hp])

        if self.sampling_scheme == "beam":

            binary[binary > 0] = 5
            binary[binary == 0] = 1
            binary = np.pad(binary, ((hp, hp),
                                     (hp, hp),
                                     (hp, hp)), 'constant')

            num = randint(1, 5)
            if num == 1:
                idxs = []
                for _ in range(self.patches_per_segment):
                    idxs.append(np.array([
                        randint(hp, binary.shape[0]-hp),
                        randint(hp, binary.shape[1]-hp),
                        randint(hp, binary.shape[2]-hp)]))

            else:
                idxs = np.argwhere(binary >= num)

        elif self.sampling_scheme == "equal":

            idxs = []
            for _ in range(self.patches_per_segment):
                idxs.append(np.array([
                    randint(binary.shape[0])+hp,
                    randint(binary.shape[1])+hp,
                    randint(binary.shape[2])+hp]))

        np.random.shuffle(idxs)

        return idxs[:self.patches_per_segment]

    def __len__(self):
        return self.segments_per_queue*self.patches_per_segment

    def __iter__(self):
        curr = 0
        while curr < len(self.train):
            # get segment_list per queue
            yield (self.train[curr], self.target[curr])
            curr += 1


def get_train_val_sets(dataset, train_fraction):

    num = len(dataset)

    train_n = int(np.ceil(num*train_fraction))
    test_n = num - train_n

    assert train_n + \
        test_n == num, f"Splitting of Training-Data does not match! num={num}. train_n={train_n}, test_n={test_n}"

    print(
        f"Number of total Samples: {num}\nTraining-Samples: {train_n}\nTest-Samples: {test_n}\n\n")

    random.shuffle(dataset)
    train_set, test_set = dataset[:train_n], dataset[train_n:]

    return train_set, test_set


class ValidationQueue():

    def __init__(self, segments, batch_size):
        self.segments = segments
        self.batch_size = batch_size
        self.idxs = np.linspace(64, 512-64, 7, endpoint=True)

        self.mask_patches, self.target_patches = self.get_batches()

    def get_batches(self):
        mask_patches = []
        target_patches = []

        for seg in self.segments:

            self.z_idxs = np.linspace(32, 224-32, 11, endpoint=True)
            mask = torch.load(seg + "/training_data.pt")
            target = torch.load(seg + "/target_data.pt")
            self.z_idxs = self.z_idxs[self.z_idxs <= (mask.shape[-1]-16)]

            for x in self.idxs:
                for y in self.idxs:
                    for z in self.z_idxs:
                        mask_patches.append(
                            mask[:, int(x)-32:int(x)+32, int(y)-32:int(y)+32, int(z)-16:int(z)+16])
                        target_patches.append(
                            target[:, int(x)-32:int(x)+32, int(y)-32:int(y)+32, int(z)-16:int(z)+16])

        return mask_patches, target_patches

    def __len__(self):
        return len(self.mask_patches)

    def __iter__(self):
        curr = 0
        while curr < len(self.mask_patches):
            yield(torch.stack(self.mask_patches[curr:curr+self.batch_size]), torch.stack(self.target_patches[curr:curr+self.batch_size]))
            curr += self.batch_size


if __name__ == "__main__":

    subject_list = ["/Users/simongutwein/Studium/Masterarbeit/test_data/" + x for x in os.listdir(
        "/Users/simongutwein/Studium/Masterarbeit/test_data") if not x.startswith(".")]

    # q = ValidationQueue(segments=subject_list, batch_size=16)
    # for batch in q:
    #     fig, ax = plt.subplots(4, 4)
    #     for i in range(int(batch.shape[0]/4)):
    #         for j in range(int(batch.shape[0]/4)):
    #             ax[int(i), int(j)].imshow(
    #                 torch.squeeze(batch[int(i*4) + j, 0, :, :, 16]))
    #     plt.show()

    train_q = ValidationQueue(subject_list, 16)

    segments_num = 0
    for i, j in train_q:
        segments_num += i.shape[0]
        print(i.shape, j.shape)

    print(segments_num)
    # total = 0
    # while total < number_needed:
    #     print(total)
    #     train_q.load_queue()
    #     for train, test in train_q:
    #         total += 16

    # print(total)