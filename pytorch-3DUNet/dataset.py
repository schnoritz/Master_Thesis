import torch
import torchio as tio
import numpy as np
import os


def SubjectDataset(subjects_dir):
    

    if subjects_dir[-1] != "/":
        subjects_dir += "/"
    
    segments = [x for x in os.listdir(subjects_dir) if not x.startswith(".")]
    
    subjects = []
    for segment in segments:
        subjects.append(load_subject(subjects_dir + segment, sampling_scheme="equal"))

    return subjects


def load_subject(subject_path, sampling_scheme):

    #binary = torch.load(f"{subject_path}/trainings_data.pt")[0, :, :, :]
    # -> die null ist noch nicht sicher, muss schauen welches von denen wirklich die binary mask ist
   
    binary = torch.randn((512, 512, 110))

    sampling_map = create_sampling_map(binary, sampling_scheme=sampling_scheme)

    subject = tio.Subject(
        trainings_data=tio.ScalarImage(tensor=torch.load(f"{subject_path}/trainings_data.pt")),
        target_data=tio.ScalarImage(tensor=torch.load(f"{subject_path}/target_data.pt")),
        sampling_map=tio.Image(tensor=sampling_map, type=tio.SAMPLING_MAP)
    )

    return subject


def create_sampling_map(binary, sampling_scheme, probability=5):

    assert sampling_scheme == "beam" or sampling_scheme == "equal", f"{sampling_scheme} is not a valid sampling scheme"

    sampling_map = torch.empty_like(binary)

    if sampling_scheme == "beam":
        sampling_map[binary != 0] = probability
        sampling_map[binary == 0] = 1

    elif sampling_scheme == "equal":
        sampling_map[:, :, :] = 1

    return torch.unsqueeze(sampling_map, 0)