import torch
import numpy as np
import os
from pydicom import dcmread
import matplotlib.pyplot as plt


def read_in(ct_path):

    for file in os.listdir(ct_path):
        if file.startswith("."):
            os.remove(ct_path + "/" + file)
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
        dcm = dcmread(file)
        stack.append(dcm.pixel_array)

    return np.stack(stack, axis=0)


def convert_stack(stack, tensor=False):

    if tensor:
        return torch.from_numpy(stack.astype(np.float32))
    else:
        return stack.astype(np.float32)


def convert_ct_array(ct_path, tensor=False):

    if ct_path[-1] != "/":
        ct_path += "/"

    slices = read_in(ct_path)
    sorted_slices = sort_ct_slices(slices)
    stack = stack_ct_images(sorted_slices)
    stack = np.transpose(stack, (1, 2, 0))
    converted = convert_stack(stack, tensor=tensor)

    return converted


if __name__ == "__main__":

    ct_images = "/Users/simongutwein/Studium/Masterarbeit/15"

    stack = convert_ct_array(ct_images)
    print(stack.shape)

    for i in range(stack.shape[2]):
        plt.imshow(stack[:, :, i])
        plt.show()
        plt.close()
