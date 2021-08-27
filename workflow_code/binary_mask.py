import nibabel as nib
import sys
import numpy as np
from utils import define_iso_center, get_angle
from pydicom import dcmread
import cv2
from cv2 import resize
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from pt_3ddose import dose_to_pt
import torch
import os
import random


def create_binary_mask(
    egsinp, ct_path, beam_config, target_size, SID=1435, tensor=False
):

    angle, shift, rotated_dim, rotated_iso_loc, px_sp = setup_binary(
        egsinp, ct_path, np.array(target_size)
    )

    percentual_increase, first_fac = scaling_factor(
        px_sp, SID, rotated_dim, rotated_iso_loc
    )

    leafes, jaws = get_leaf_positions(beam_config)

    fieldsize = [np.round(220 / px_sp[2]).astype(int),
                 np.round((576) / px_sp[0]).astype(int)]

    entry_plane = get_entry_plane(
        leafes, jaws, first_fac, fieldsize, rotated_dim
    )

    beam = scale_beam(entry_plane, rotated_dim, percentual_increase)

    beam = np.transpose(beam, (2, 1, 0))

    # äquivalent quadartische Feldgröße
    field_area = calc_field_area(rotated_iso_loc, px_sp, beam)

    output = rotate_crop(angle, target_size, shift, beam)

    output[output < 0] = 0
    output[output > 1] = 1
    output *= field_area / 100
    #########################################################################################

    print("Binary Mask created!")
    if tensor:
        return torch.tensor(output, dtype=torch.float32)
    else:
        return output


def setup_binary(egsinp, ct_path, target_size,):

    egsinp_lines = open(egsinp).readlines()
    iso_center, px_sp = define_iso_center(egsinp_lines[5], ct_path)
    angle = get_angle(egsinp_lines[5], radians=False)
    shift = iso_center - target_size // 2
    rotated_dim = np.array(
        [800, 800, target_size[2] + 2 * abs(shift[2])], dtype=int)
    rotated_iso_loc = iso_center[0] + (rotated_dim[0] - target_size[0]) // 2

    return angle, shift, rotated_dim, rotated_iso_loc, px_sp


def rotate_crop(angle, output_dim, shift, beam):

    if angle == 0:
        rotated = beam
    else:
        rotated = ndimage.rotate(
            beam, -angle, axes=(0, 1), reshape=False, prefilter=False, order=0
        )

    center_window = np.array(
        [
            np.round((rotated.shape[0] - output_dim[0]) / 2),
            np.round((rotated.shape[1] - output_dim[1]) / 2),
            np.round((rotated.shape[2] - output_dim[2]) / 2),
        ],
        dtype=int,
    )

    shifted_window = center_window - shift

    second_shift = np.array(
        [-np.cos(np.radians(angle))*shift[0], np.sin(np.radians(angle))*shift[0], 0])

    second_shifted_window = np.round(shifted_window - second_shift).astype(int)

    output = rotated[
        second_shifted_window[0]: second_shifted_window[0] + output_dim[0],
        second_shifted_window[1]: second_shifted_window[1] + output_dim[1],
        second_shifted_window[2]: second_shifted_window[2] + output_dim[2],
    ]

    return output


def calc_field_area(iso_rotated, px_sp, beam):

    iso_plane = beam[iso_rotated, :, :]
    voxel_area = px_sp[1] * px_sp[2]
    area = voxel_area * np.sum(iso_plane)

    return area


def scaling_factor(px_sp, SID, rotated_dim, rotated_iso_loc):

    # hier verwendung von Strahlensatz mit voxelgröße und SID
    scale_factor = [
        ((SID/px_sp[0])+i)/(SID/px_sp[0]) for i in range(0 - rotated_iso_loc, rotated_dim[0] - rotated_iso_loc)
    ]

    percentual_increase = [scale_factor[i] / scale_factor[0]
                           for i in range(len(scale_factor))]

    return percentual_increase, scale_factor[0]


def get_leaf_positions(config_path):

    with open(config_path, "r") as fin:
        lines = fin.readlines()

    positions = []
    for line in lines:
        positions.append(np.array(line.split(",")[:2], dtype=float))

    jaws = np.array(positions[-1]) * 10

    leafes = positions[:-1]

    return leafes, jaws


def get_entry_plane(leafes, jaws, entry_plane_fac, size, dim):

    # 0.05cm * 2200 -> genaugikeit der leafes auf halben mm
    fl = np.ones((440, 80))
    i = 0
    for top, bot in leafes:
        fl[: 220 + int(np.round(top * 20)), i] = 0
        fl[220 + int(np.round(bot * 20)):, i] = 0
        i += 1

    fl = resize(fl, (576*2, 440),
                interpolation=cv2.INTER_LINEAR)

    l_jaw_idx = fl.shape[1]//2 + int(np.round(jaws[0]*2))
    r_jaw_idx = fl.shape[1]//2+1 + int(np.round(jaws[1]*2))

    fl[:, :l_jaw_idx] = 0
    fl[:, r_jaw_idx:] = 0

    fl = resize(fl, (np.round(size[1]*entry_plane_fac).astype(int), np.round(size[0]*entry_plane_fac).astype(int)),
                interpolation=cv2.INTER_LINEAR)

    fl = np.flip(fl, 0)

    t_padding = [
        int(np.floor((dim[2] - fl.shape[0]) / 2)),
        int(np.ceil((dim[2] - fl.shape[0]) / 2)),
    ]

    s_padding = [
        int(np.floor((dim[0] - fl.shape[1]) / 2)),
        int(np.ceil((dim[0] - fl.shape[1]) / 2)),
    ]

    fl = np.pad(
        fl, ((t_padding[0], t_padding[1]), (s_padding[0], s_padding[1])), "constant", constant_values=0)

    return fl


def scale_beam(fl, dim, scale):

    vol = np.empty((fl.shape[0], fl.shape[1], dim[0]))

    for slice in tqdm(range(dim[0]), file=sys.stdout, postfix="\n"):
        vol[:, :, slice] = paddedzoom(fl, scale[slice])

    return vol


# https://gist.github.com/i-namekawa/74a817683b0e68cee521
def paddedzoom(img, zoomfactor):

    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoomfactor)

    return cv2.warpAffine(img, M, img.shape[::-1])


if __name__ == "__main__":

    segments = ["h0_0"]
    entity = "head"

    root_dir = f"/mnt/qb/baumgartner/sgutwein84/output_{entity}/"
    save_dir = "/home/baumgartner/sgutwein84/container/test"
    # segments = [x for x in os.listdir(
    #     root_dir) if not x.startswith(".") and not "ct" in x]
    #segments = random.sample(segments, 5)

    for segment in segments:

        pat = segment.split("_")[0]
        print(pat)

        path = f"{root_dir}/{segment}/"
        ct_path = f"{root_dir}ct/{pat}/"
        beam_config = path + f"beam_config_{segment}.txt"
        dose_path = path + f"{segment}_1E07.3ddose"
        egsinp = path + f"{segment}.egsinp"

        dose = dose_to_pt(dose_path, ct_path)
        dose = dose / dose.max()

        binary = create_binary_mask(
            egsinp, ct_path, beam_config, target_size=dose.shape)

        # for i in range(0,dose.shape[2], 3):
        #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        #     ax[0].imshow(binary[:, :, i], cmap="bone")
        #     ax[0].imshow(dose[:, :, i], alpha=0.5, cmap="jet")

        #     ax[1].imshow(binary[:, :, i], cmap="bone")
        #     ax[1].imshow(dose[:, :, i], alpha=0.5, cmap="jet")

        #     # plt.savefig(
        #     #     f"/home/baumgartner/sgutwein84/container/test/{segment}.png")
        #     plt.show(fig)
        #     plt.close()

        plt.imshow(binary[256, :, :])
        plt.imshow(dose[256, :, :], alpha=0.5)
        plt.show()

        img = nib.Nifti1Image(np.array(dose), np.eye(4))

        img.header.get_xyzt_units()
        img.to_filename(
            f"{save_dir}/dose_{segment}.nii.gz")

        img = nib.Nifti1Image(np.array(binary), np.eye(4))

        img.header.get_xyzt_units()
        img.to_filename(
            f"{save_dir}/binary_{segment}.nii.gz")

        print(f"{segment} done")
