import torch
import numpy as np
import os
from pydicom import dcmread, uid
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def read_in(ct_path):

    for file in os.listdir(ct_path):
        if file.startswith("."):
            os.remove(ct_path + file)
    return [ct_path + x for x in os.listdir(ct_path) if not x.startswith(".") and not "listfile" in x]


def sort_ct_slices(files):

    locations = []
    for file_ in files:
        dcm = dcmread(file_, force=True)
        locations.append(dcm.SliceLocation)

    return sorted(zip(locations, files))


def stack_ct_images(sorted_ct):

    stack = []
    for _, file in sorted_ct:
        dcm = dcmread(file, force=True)
        dcm.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
        stack.append(dcm.pixel_array)

    return np.stack(stack, axis=0)


def convert_stack(stack, tensor=False):

    if tensor:
        return stack.float()
    else:
        return np.array(stack).astype(np.float32)


def convert_ct_array(ct_path, target_size=None, tensor=False):

    if ct_path[-1] != "/":
        ct_path += "/"

    slices = read_in(ct_path)
    sorted_slices = sort_ct_slices(slices)
    stack = stack_ct_images(sorted_slices)
    stack = torch.tensor(np.transpose(
        stack, (1, 2, 0)).astype(np.float32))

    stack = stack.unsqueeze(0)
    stack = stack.unsqueeze(0)

    if target_size:
        stack = torch.nn.functional.interpolate(
            stack, size=tuple(target_size), mode="trilinear", align_corners=True)

    stack = stack.squeeze()
    converted = convert_stack(stack, tensor=tensor)

    return converted


if __name__ == "__main__":
    pass
