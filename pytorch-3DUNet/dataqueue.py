import torch
import os
import numpy as np
import random
from time import time


class DataQueue():

    def __init__(self, segment_list, batch_size, segments_per_queue, patch_size, patches_per_segment):
        self.segment_list = segment_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.segments_per_queue = segments_per_queue
        self.patches_per_segment = patches_per_segment

    def load_q(self, segs):

        data = []

        for seg in segs:
            data.append(
                (torch.load(f"{seg}/training_data.pt"), torch.load(f"{seg}/target_data.pt")))

        train_patches = []
        target_patches = []
        hp = self.patch_size/2
        for masks, target in data:
            idxs = self.get_sample_idx(masks)
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

        binary[binary > 0] = 5
        binary[binary == 0] = 1
        binary = np.pad(binary, ((hp, hp),
                                 (hp, hp),
                                 (hp, hp)), 'constant')

        num = random.randint(1, 5)
        if num == 1:
            idxs = []
            for _ in range(self.patches_per_segment):
                idxs.append(np.array([random.randint(hp, binary.shape[0]-hp),
                            random.randint(hp, binary.shape[1]-hp),
                            random.randint(hp, binary.shape[2]-hp)]))
        else:
            idxs = np.argwhere(binary >= num)
            np.random.shuffle(idxs)

        return idxs[:self.patches_per_segment]

    def __iter__(self):
        curr = 0
        while curr < len(self.segment_list):
            # get segment_list per queue
            train, target = self.load_q(
                self.segment_list[curr:curr+self.segments_per_queue])
            yield (train, target)
            curr += self.segments_per_queue


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


if __name__ == "__main__":

    subject_list = ["/Users/simongutwein/Studium/Masterarbeit/test_data/" + x for x in os.listdir(
        "/Users/simongutwein/Studium/Masterarbeit/test_data") if not x.startswith(".")]

    train, test = get_train_test_sets(subject_list, train_fraction=0.8)

    train_q = DataQueue(train, 16, 2, 32, 5000)
    test_q = DataQueue(test, 16, 2, 32, 5000)
    epochs = 3
    start = time()
    total = 0

    for epoch in range(epochs):
        print("Epoch: ", epoch+1)
        num_batches = 0
        for i, j in train_q:
            for (k, l) in zip(i, j):
                num_batches += 1
                #print(k.shape, l.shape)
                total += k.shape[0]
        print(" ", num_batches, "train_batches")
        num_batches = 0
        for m, n in test_q:
            for (o, p) in zip(m, n):
                num_batches += 1
        print(" ", num_batches, "val_baches")

    print(total, "patches in", time()-start)
