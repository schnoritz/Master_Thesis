import torch
import os
import numpy as np
import random
from numpy.random import randint
from time import time
import threading
import queue
from multiprocessing import Process

import matplotlib.pyplot as plt


class Worker(threading.Thread):
    def __init__(self, q, data_list, *args, **kwargs):
        self.q = q
        self.data_list = data_list
        super().__init__(*args, **kwargs)

    def run(self):
        while True:
            try:
                seg = self.q.get_nowait()
                self.data_list.append(
                    (torch.load(f"{seg}/training_data.pt"), torch.load(f"{seg}/target_data.pt"))
                )
            except queue.Empty:
                return

            self.q.task_done()


class DataQueue():

    def __init__(
            self, segment_list, batch_size, segments_per_queue, patch_size, patches_per_segment,
            sampling_scheme="equal", num_worker=1):
        self.segment_list = segment_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.segments_per_queue = segments_per_queue
        self.patches_per_segment = patches_per_segment
        self.sampling_scheme = sampling_scheme
        self.available_segments = self.segment_list.copy()
        self.num_worker = num_worker

    def reset_queue(self):
        self.available_segments = self.segment_list.copy()

    def load_queue(self):

        print(f"{len(self.available_segments)} available Segments in Queue")

        if self.segments_per_queue > len(self.available_segments):
            self.reset_queue()

        curr_segments = random.sample(self.available_segments, self.segments_per_queue)

        for seg in curr_segments:
            self.available_segments.remove(seg)

        self.train, self.target = self.create_queue(curr_segments)

    def create_queue(self, segs):

        start = time()

        data = []
        q = queue.Queue()
        for seg in segs:
            q.put_nowait(seg)

        for _ in range(self.num_worker):
            Worker(q, data).start()

        q.join()

        print(f"Loading of {len(segs)} segments took {np.round(time()-start,2)} seconds with {self.num_worker} workers.")

        sizes = np.array([x[0].shape[-1] for x in data])
        if np.any(self.patch_size > sizes):

            resized_data = []
            for i,  (mask, target) in enumerate(data):
                sz = mask.shape
                if sz[-1] < self.patch_size:
                    zeros = torch.zeros(5, sz[1], sz[2], self.patch_size-sz[-1])
                    mask = torch.cat((mask, zeros), dim=3)
                    zeros = torch.zeros(1, sz[1], sz[2], self.patch_size-sz[-1])
                    target = torch.cat((target, zeros), dim=3)
                    resized_data.append((mask, target))
                else:
                    resized_data.append((mask, target))

            data = resized_data

        train_patches = []
        target_patches = []
        hp = self.patch_size/2
        for masks, target in data:

            idxs = self.get_sample_idx(masks)

            for i in idxs:
                train_patches.append(masks[:, int(i[0]-hp):int(i[0]+hp), int(i[1]-hp):int(i[1]+hp), int(i[2]-hp):int(i[2]+hp)])
                target_patches.append(target[:, int(i[0]-hp):int(i[0]+hp), int(i[1]-hp):int(i[1]+hp), int(i[2]-hp):int(i[2]+hp)])

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
            train.append(train_patches[curr-self.batch_size:curr])
            target.append(target_patches[curr-self.batch_size:curr])

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
            if hp == int(masks.shape[3]/2):
                for _ in range(self.patches_per_segment):
                    idxs.append(np.array([
                        randint(binary.shape[0])+hp,
                        randint(binary.shape[1])+hp, hp]))
            else:
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
        self.idxs = np.linspace(16, 512-16, 11, endpoint=True)

        self.mask_patches, self.target_patches = self.get_batches()

    def get_batches(self):
        mask_patches = []
        target_patches = []

        for seg in self.segments:

            self.z_idxs = np.linspace(32, 224-32, 6, endpoint=True)
            mask = torch.load(seg + "/training_data.pt")
            target = torch.load(seg + "/target_data.pt")
            self.z_idxs = self.z_idxs[self.z_idxs <= (mask.shape[-1]-16)]

            for x in self.idxs:
                for y in self.idxs:
                    for z in self.z_idxs:
                        mask_patches.append(
                            mask[:, int(x)-16:int(x)+16, int(y)-16:int(y)+16, int(z)-16:int(z)+16])
                        target_patches.append(
                            target[:, int(x)-16:int(x)+16, int(y)-16:int(y)+16, int(z)-16:int(z)+16])

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
        "/Users/simongutwein/Studium/Masterarbeit/test_data/") if not x.startswith(".")]

    # q = ValidationQueue(segments=subject_list, batch_size=16)
    # for batch in q:
    #     fig, ax = plt.subplots(4, 4)
    #     for i in range(int(batch.shape[0]/4)):
    #         for j in range(int(batch.shape[0]/4)):
    #             ax[int(i), int(j)].imshow(
    #                 torch.squeeze(batch[int(i*4) + j, 0, :, :, 16]))
    #     plt.show()

    num = 10

    train_q = DataQueue(subject_list, num, num, 128, 20, num_worker=4)

    start = time()
    train_q.load_queue()
    for i, ii in train_q:
        print(i.shape)

    #train_q = DataQueue(subject_list, 16, num//2, 32, 500, num_worker=8)

    # start = time()

    # print(f"Loading with 8 Worker took {np.round(time()-start,2)} seconds.")

    # train_q = DataQueue(subject_list, 16, num//4, 32, 50, num_worker=8)

    # start = time()
    # train_q.load_queue()
    # print(f"Loading with 2 Worker took {np.round(time()-start,2)} seconds.")
