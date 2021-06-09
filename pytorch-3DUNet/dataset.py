import torch
import torchio as tio
import numpy as np
import os
from torch.utils.data import DataLoader


def SubjectDataset(subjects_dir, sampling_scheme="equal"):

    if subjects_dir[-1] != "/":
        subjects_dir += "/"

    segments = [x for x in os.listdir(subjects_dir) if not x.startswith(".")]

    subjects = []
    for segment in segments:
        subjects.append(load_subject(
            subjects_dir + segment, sampling_scheme=sampling_scheme))

    return subjects


def load_subject(subject_path, sampling_scheme):

    binary = torch.load(f"{subject_path}/training_data.pt")[0, :, :, :]
    # -> [0, :, :, :], da binary mask an erster Stelle ist

    #binary = torch.randn((512, 512, 110))

    sampling_map = create_sampling_map(binary, sampling_scheme=sampling_scheme)

    subject = tio.Subject(
        trainings_data=tio.ScalarImage(
            tensor=torch.load(f"{subject_path}/training_data.pt")
        ),
        target_data=tio.ScalarImage(
            tensor=torch.load(f"{subject_path}/target_data.pt")
        ),
        sampling_map=tio.Image(
            tensor=sampling_map,
            type=tio.SAMPLING_MAP
        )
    )

    return subject


def normalize(tensor_4d):

    for i in range(tensor_4d.shape[0]):
        tensor_4d[i, :, :, :] = (
            tensor_4d[i, :, :, :]-tensor_4d[i, :, :, :].min())/(tensor_4d[i, :, :, :].max()-tensor_4d[i, :, :, :].min())

    return tensor_4d


def create_sampling_map(binary, sampling_scheme, probability=5):

    assert sampling_scheme == "beam" or sampling_scheme == "equal", f"{sampling_scheme} is not a valid sampling scheme"

    sampling_map = torch.empty_like(binary)

    if sampling_scheme == "beam":
        sampling_map[binary != 0] = probability
        sampling_map[binary == 0] = 1

    elif sampling_scheme == "equal":
        sampling_map[:, :, :] = 1

    return torch.unsqueeze(sampling_map, 0)


def get_train_test_sets(dataset, train_fraction):

    num = len(dataset)

    train_n, test_n = int(np.ceil(num*train_fraction)
                          ), int(np.floor(num*(1-train_fraction)))

    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_n, test_n])

    return train_set, test_set


# samples_per_volume=512, queue_length=1024,):
def setup_loaders(SubjectList, train_fraction=0.9, patch_size=32, batch_size=64, samples_per_volume=64, queue_length=128,):

    train_set, test_set = get_train_test_sets(SubjectList, train_fraction)

    sampler = tio.data.WeightedSampler(
        patch_size=patch_size,
        probability_map='sampling_map'
    )

    train_queue, test_queue = setup_queue(
        (train_set, test_set),
        queue_length,
        samples_per_volume,
        sampler
    )

    train_loader = DataLoader(
        dataset=train_queue,
        batch_size=batch_size,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_queue,
        batch_size=batch_size,
        num_workers=2
    )

    return train_loader, test_loader


def setup_queue(sets, length, samples_per_volume, sampler, ):

    train_queue = get_queue(sets[0], length, samples_per_volume, sampler)
    test_queue = get_queue(sets[1], length, samples_per_volume, sampler)

    return train_queue, test_queue


def get_queue(set, length, samples_per_volume, sampler):

    queue = tio.Queue(
        set,
        length,
        samples_per_volume,
        sampler,
        shuffle_patches=True,
        # shuffle_subjects=True
    )

    return queue
