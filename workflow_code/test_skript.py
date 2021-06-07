import torch
import os
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchio.data import subject
from time import time


def create_dummy_data(size=(512, 512, 110)):

    os.system("cd; cd /home/baumgartner/sgutwein84/container; mkdir training")

    num_training_fields = 8
    patients = ["p0", "p1", "p2", "p3", "p4"]

    # make ct dummy
    ct = torch.randn(size)

    # make radiopdepth dummy
    radio = torch.randn(size)

    # make binary dummy
    binary = torch.randn(size)

    # make center distance dummy
    center_dist = torch.randn(size)

    # make source distance dummy
    source_dist = torch.randn(size)

    # make target dummy
    target = torch.randn(size)

    for patient in patients:
        for field in range(num_training_fields):

            target_name = f"{patient}_{field}"

            trainings_data = torch.stack(
                (ct, radio, binary, center_dist, source_dist))
            target_data = torch.unsqueeze(target, 0)

            os.system(
                f"cd; cd /home/baumgartner/sgutwein84/container/training; mkdir {target_name}")
            torch.save(
                trainings_data, f"/home/baumgartner/sgutwein84/container/training/{target_name}/trainings_data.pt")
            torch.save(
                target_data, f"/home/baumgartner/sgutwein84/container/training/{target_name}/target_data.pt")
            print(f"{target_name} created!")


def create_sampling_map(binary, sampling_scheme):

    assert sampling_scheme == "beam" or sampling_scheme == "equal", f"{sampling_scheme} is not a valid sampling scheme"

    sampling_map = torch.empty_like(binary)

    if sampling_scheme == "beam":
        sampling_map[binary != 0] = 5
        sampling_map[binary == 0] = 1

    elif sampling_scheme == "equal":
        sampling_map[:, :, :] = 1

    return torch.unsqueeze(sampling_map, 0)


def load_subject(subject_dir):

    #dat = torch.load(f"{subject_dir}/trainings_data.pt")[0, :, :, :]
    dat = torch.zeros((512, 512, 110))
    sampling_map = create_sampling_map(dat, sampling_scheme="equal")

    subject = tio.Subject(
        trainings_data=tio.ScalarImage(
            tensor=torch.load(f"{subject_dir}/trainings_data.pt")),
        target_data=tio.ScalarImage(
            tensor=torch.load(f"{subject_dir}/target_data.pt")),
        sampling_map=tio.Image(tensor=sampling_map, type=tio.SAMPLING_MAP)
    )

    return subject


if __name__ == "__main__":

    for dir_path, test, filenames in os.walk("/Users/simongutwein/home/baumgartner/sgutwein84/container/output_20210522"):
        print(dir_path, test, filenames)

    # create_dummy_data(size=(512,512,110))

    # subjects = []
    # training_folder = "/home/baumgartner/sgutwein84/container/training/"
    # fields = [x for x in os.listdir(training_folder) if not x.startswith(".")]
    # for field in fields:
    #     subjects.append(load_subject(training_folder + field))

    # subjects_dataset = tio.SubjectsDataset(subjects)

    # patch_size = 64
    # #
    # samples_per_volume = 256
    # queue_length = samples_per_volume*16
    # batch_size = 32

    # sampler = tio.data.WeightedSampler(patch_size=patch_size, probability_map='sampling_map')

    # patches_queue = tio.Queue(
    #     subjects_dataset,
    #     queue_length,
    #     samples_per_volume,
    #     sampler,
    #     shuffle_patches=True,
    #     #shuffle_subjects=True
    # )

    # patches_loader = torch.utils.data.DataLoader(patches_queue, batch_size=batch_size, num_workers=8)
    # num_epochs = 1

    # num=0
    # for epoch_index in range(num_epochs):
    #     #print(f"Epoch: {epoch_index+1}")
    #     patch_num = 0
    #     start = time()
    #     for patches_batch in patches_loader:
    #         patch_num += patches_batch["trainings_data"]['data'].shape[0]
    #         #print(patches_batch["trainings_data"]['data'].shape)
    #     done = time()
    #     print(f"full batch of {samples_per_volume*len(subjects)} took {np.round(done-start)} seconds")
    #     print(f"USED METRICS:\nsamples_per_volume: {samples_per_volume}\nbatch_size: {batch_size}\nqueue_length: {queue_length}")
    #     print(f"TIME PER PATCH:{(done-start)/(samples_per_volume*len(subjects))}")
