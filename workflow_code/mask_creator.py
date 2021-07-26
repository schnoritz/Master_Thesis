import argparse
from pt_3ddose import dose_to_pt
from radiological_depth import radiological_depth
from binary_mask import create_binary_mask
from pt_ct import convert_ct_array
from dist_center import distance_center
from dist_source import distance_source
import numpy as np
import torch
import subprocess
import os


from scipy import ndimage
import matplotlib.pyplot as plt


def parse():

    parser = argparse.ArgumentParser(description="Clean dosxyznrc folder.")

    parser.add_argument(
        "egsinp_file",
        type=str,
        metavar="",
        help="",
    )

    # parser.add_argument(
    #     "egsphant_file",
    #     type=str,
    #     metavar="",
    #     help="",
    # )

    parser.add_argument(
        "beam_config_file",
        type=str,
        metavar="",
        help="",
    )

    parser.add_argument(
        "dose_file",
        type=str,
        metavar="",
        help="",
    )

    parser.add_argument(
        "segment",
        type=str,
        metavar="",
        help="",
    )

    parser.add_argument(
        "output_folder",
        type=str,
        metavar="",
        help="",
    )

    args = parser.parse_args()

    return args


def create_mask_files(
    egsinp_file,
    beam_config_file,
    dose_file,
    segment,
    output_folder
):

    set_zero = True
    normalize = True
    patient = segment.split("_")[0]
    ct_path = f"/home/baumgartner/sgutwein84/container/output_{output_folder}/ct/{patient}"

    if not os.path.isdir(f'/home/baumgartner/sgutwein84/container/training_{output_folder}'):

        subprocess.run(
            ['mkdir',
             f'/home/baumgartner/sgutwein84/container/training_{output_folder}']
        )

    subprocess.run(
        ['mkdir',
            f'/home/baumgartner/sgutwein84/container/training_{output_folder}/{segment}']
    )

    dose_mask = dose_to_pt(dose_file, ct_path, tensor=True)

    ct_mask = convert_ct_array(
        ct_path=ct_path,
        target_size=dose_mask.shape,
        tensor=True
    )

    if set_zero:
        # treshhold muss noch genau bestimmt werden -> Daniela meinte 150 wird bei bestrahlungsplanung gemacht
        mask = np.array(ct_mask > 150).astype(bool)
        for i in range(mask.shape[2]):
            mask[:, :, i] = ndimage.binary_fill_holes(
                mask[:, :, i]).astype(bool)
        mask = np.invert(mask)

    assert dose_mask.shape == ct_mask.shape, "shapes of dose and ct dont match"

    if set_zero:
        radio_depth_mask = radiological_depth(
            np.array(ct_mask), egsinp_file, ct_path, mask=np.invert(mask), tensor=True
        )
    else:
        radio_depth_mask = radiological_depth(
            np.array(ct_mask), egsinp_file, ct_path, tensor=True
        )

    center_mask = distance_center(
        egsinp_file, ct_path, target_size=np.array(ct_mask).shape, tensor=True
    )

    source_mask = distance_source(
        egsinp_file, ct_path, target_size=np.array(ct_mask).shape, tensor=True
    )

    binary_mask = create_binary_mask(
        egsinp_file, ct_path, beam_config_file, target_size=np.array(ct_mask).shape, px_sp=np.array([1.171875, 1.171875, 3]), SID=1435, tensor=True
    )

    # creates stack of size (5, 512  512, num_slices)
    stack = torch.stack((
        binary_mask,
        ct_mask,
        radio_depth_mask,
        center_mask,
        source_mask))

    if set_zero:
        stack[0, mask] = 0
        stack[2, mask] = 0
        stack[3, mask] = 0
        stack[4, mask] = 0

    dose_mask = torch.unsqueeze(dose_mask, 0)

    if normalize:
        stack[0] = stack[0]
        stack[1] = stack[1]/1800  # CT Mask
        stack[2] = stack[2]/1800  # Radio Depth Mask
        stack[3] = stack[3]/(1.171875)  # scale by pixel spacing
        stack[4] = stack[4]/(1435/1.171875)  # scale by SID and pixel spacing
        dose_mask = dose_mask / 1E-17

        print(f"Dose Max-Value is: {np.round(dose_mask.max(),2)}")
        print(f"Binary Mask Max-Value is: {np.round(stack[0].max(),2)}\nCT Max-Value is: {np.round(stack[1].max(),2)}\nRadiological-Depth Max-Value is: {np.round(stack[2].max(),2)}\nCenter-Beamline-Distance Max-Value is: {np.round(stack[3].max(),2)}\nSource-Distance Max-Value is: {np.round(stack[4].max(),2)}")

    stack = stack.float()
    dose_mask = dose_mask.float()

    torch.save(
        stack, f"/home/baumgartner/sgutwein84/container/training_{output_folder}/{segment}/training_data.pt"
    )

    torch.save(
        dose_mask, f"/home/baumgartner/sgutwein84/container/training_{output_folder}/{segment}/target_data.pt"
    )


if __name__ == "__main__":

    args = parse()
    create_mask_files(args.egsinp_file,
                      args.beam_config_file,
                      args.dose_file,
                      args.segment,
                      args.output_folder)

    # debug
    # egsinp_file = "/home/baumgartner/sgutwein84/container/output_prostate/p0_0/p0_0.egsinp"
    # beam_config_file = "/home/baumgartner/sgutwein84/container/output_prostate/p0_0/beam_config_p0_0.txt"
    # dose_file = "/home/baumgartner/sgutwein84/container/output_prostate/p0_0/p0_0_1E07.3ddose"
    # segment = "p0_0"
    # output_folder = "prostate"

    # stack, dose_mask = create_mask_files(egsinp_file,
    #                                      beam_config_file,
    #                                      dose_file,
    #                                      segment,
    #                                      output_folder)

    # fig, ax = plt.subplots(1, 6, figsize=(24, 4))
    # dose_mask = dose_mask.squeeze()
    # stack = stack.squeeze()
    # print(dose_mask.shape)
    # for i in range(dose_mask.shape[2]):
    #     ax[0].imshow(stack[0, :, :, i])
    #     ax[1].imshow(stack[1, :, :, i])
    #     ax[2].imshow(stack[2, :, :, i])
    #     ax[3].imshow(stack[3, :, :, i])
    #     ax[4].imshow(stack[4, :, :, i])
    #     ax[5].imshow(dose_mask[:, :, i])
    #     plt.savefig(
    #         f"/home/baumgartner/sgutwein84/container/test/new_{i}.png")
