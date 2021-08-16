import numpy as np
import os
from pydicom import dcmread, uid
import matplotlib.pyplot as plt
from pprint import pprint


def define_iso_center(egsinp, ct_path):

    iso_center = np.array(egsinp.split(",")[2:5], dtype=float)

    zero_point, slice_spacing, pixel_spacing = extract_phantom_data(ct_path, pixel_spacing=True)

    px_sp = np.array([pixel_spacing[0], pixel_spacing[1], slice_spacing]).astype(float)

    # wird hier geändert, da mein Koordinatensystem immer 3mm slice spacing hat
    # und das Isocentrum auf diesem Koordinatensystem berechnet wird
    px_sp[2] = 3
    iso_slice = np.zeros((3,))
    for i in range(3):
        iso_slice[i] = (iso_center[i] - zero_point[i]) / (px_sp[i]/10)

    iso_slice = np.round(iso_slice).astype(int)

    assert iso_slice[0] < 512, "isocenter lies outside of Volume"
    assert iso_slice[1] < 512, "isocenter lies outside of Volume"

    iso_slice[0], iso_slice[1] = iso_slice[1], iso_slice[0]

    return iso_slice, px_sp


def define_origin_position(egsinp, iso_center, px_sp, SID=1435):

    angle = np.radians(float(egsinp.split(",")[6]) - 270)

    pixel_SID = int(SID / px_sp[0])
    origin = np.array(
        [
            iso_center[0] - np.cos(angle) * pixel_SID,
            iso_center[1] + np.sin(angle) * pixel_SID,
            iso_center[2],
        ]
    ).astype("float")

    return origin


def get_angle(egsinp, radians=False):

    if radians == False:
        return float(egsinp.split(",")[6]) - 270
    else:
        return np.radians(float(egsinp.split(",")[6]) - 270)


def extract_phantom_data(ct_path, pixel_spacing=False):

    if ct_path[-1] != "/":
        ct_path += "/"

    ct_files = [
        x for x in os.listdir(ct_path) if "dcm" in x.lower() and not x.startswith(".")
    ]

    ct_dict = []
    for file in ct_files:
        with dcmread(ct_path + file, force=True) as dcmin:
            ct_dict.append(
                {"filename": file, "slice_location": dcmin.SliceLocation})

    ct_dict.sort(key=lambda t: t["slice_location"])
    first_file = dcmread(ct_path + ct_dict[0]["filename"], force=True)
    zero_point = np.array(first_file.ImagePositionPatient).astype(float)

    if pixel_spacing:
        with dcmread(ct_path + file, force=True) as dcmin:
            slice_spacing = int(dcmin.SliceThickness)
            pixel_spacing = dcmin.PixelSpacing

    if pixel_spacing:
        return zero_point/10, slice_spacing, pixel_spacing

    return zero_point/10


if __name__ == "__main__":

    path = "/Users/simongutwein/Studium/Masterarbeit/test/p0"
    egsinp = open(
        "/home/baumgartner/sgutwein84/container/output_prostate/p0_0/p0_0.egsinp"
    ).readlines()[5]

    iso_center = define_iso_center(egsinp, path)
    if path[-1] != "/":
        path += "/"

    ct_files = [
        x for x in os.listdir(path) if "dcm" in x.lower() and not x.startswith(".")
    ]

    ct_dict = []
    for file in ct_files:
        with dcmread(path + file, force=True) as dcmin:
            ct_dict.append(
                {"filename": file, "slice_location": dcmin.SliceLocation})

    ct_dict.sort(key=lambda t: t["slice_location"])

    ct = np.zeros((512, 512, len(ct_dict)))
    for num, file in enumerate(ct_dict):
        with dcmread(path + file["filename"], force=True) as dcmin:
            dcmin.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
            ct[:, :, num] = dcmin.pixel_array

    for i in range(ct.shape[2]):
        print(i)
        if i == iso_center[2]:
            plt.imshow(ct[:, :, i])
            plt.scatter(iso_center[0], iso_center[1], s=20, color='red')
            plt.show()
        else:
            plt.imshow(ct[:, :, i])
            plt.show()
