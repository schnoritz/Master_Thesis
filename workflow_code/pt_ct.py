import torch
import numpy as np
import os
from pydicom import dcmread, uid
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def read_in(ct_path):

    return [ct_path + x for x in os.listdir(ct_path) if not x.startswith(".") and "dcm" in x]


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

    intercept = float(dcm.RescaleIntercept)
    stack = np.stack(stack, axis=0)
    stack = stack + intercept

    return stack


def convert_stack(stack, tensor=False):

    if tensor:
        return stack.float()
    else:
        return np.array(stack).astype(np.float32)


def convert_ct_array(ct_path, target_size=None, tensor=False, ED=False):

    if ct_path[-1] != "/":
        ct_path += "/"

    slices = read_in(ct_path)
    sorted_slices = sort_ct_slices(slices)
    stack = stack_ct_images(sorted_slices)
    stack = torch.tensor(np.transpose(
        stack, (1, 2, 0)).astype(np.float32))

    if target_size:
        stack = stack.unsqueeze(0)
        stack = stack.unsqueeze(0)
        stack = torch.nn.functional.interpolate(
            stack, size=tuple(target_size), mode="trilinear", align_corners=False)

    stack = stack.squeeze()

    if ED:
        converted = convert_to_ED(stack)

    converted = convert_stack(converted, tensor=tensor)

    return converted


def convert_to_ED(data):

    print("Converting to Electron Density")

    data = data.to(torch.int)
    converted = np.zeros_like(data).astype(np.float32)

    conversion_data = [
        [-1000,  -807,  -519,   -63,   -25,    -9,    45,    52,   243,   879, 2275],
        [0.001, 0.190, 0.489, 0.949, 0.976, 1.002, 1.043, 1.052, 1.117, 1.456, 2.200]]

    print("Conversion Table used:\n", conversion_data[0], "\n", conversion_data[1], "\n")

    converted[data >= conversion_data[0][-1]] = conversion_data[1][-1]
    converted[data <= conversion_data[0][0]] = conversion_data[1][0]

    for x in range(len(conversion_data[0])-1):
        low_HU = conversion_data[0][x]
        low_ED = conversion_data[1][x]
        high_HU = conversion_data[0][x+1]
        high_ED = conversion_data[1][x+1]

        conversion = np.linspace(low_ED, high_ED, high_HU-low_HU)

        for num, HU_value in enumerate(range(low_HU, high_HU)):
            converted[data == HU_value] = conversion[num]

    return torch.tensor(converted)


if __name__ == "__main__":

    converted = convert_ct_array(
        "/Users/simongutwein/Studium/Masterarbeit/anonymized/h/h0", target_size=(512, 512, 150), tensor=True, ED=True)
    print(converted.shape)
    converted = np.array(converted)
    converted[converted < 0.13] = np.nan
    torch.save(converted, "/Users/simongutwein/Studium/Masterarbeit/test/conversion.pt")
    for i in range(150):
        print(converted[:, :, i].max(), converted[:, :, i].min())
        plt.imshow(converted[:, :, i])
        plt.show()
