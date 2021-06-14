import argparse
from pt_3ddose import dose_to_pt
from radiological_depth import radiological_depth
from binary_mask import create_binary_mask
from pt_ct import convert_ct_array
from dist_center import distance_center
from dist_source import distance_source
import numpy as np
import torch
import shutil
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

    parser.add_argument(
        "egsphant_file",
        type=str,
        metavar="",
        help="",
    )

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
    egsphant_file,
    beam_config_file,
    dose_file,
    segment,
    output_folder
):

    set_zero = True
    patient = segment.split("_")[0]

    if not os.path.isdir(f'/home/baumgartner/sgutwein84/container/training_data{output_folder}'):

        subprocess.run(
            ['mkdir',
             f'/home/baumgartner/sgutwein84/container/training_data{output_folder}']
        )

    subprocess.run(
        ['mkdir',
            f'/home/baumgartner/sgutwein84/container/training_data{output_folder}/{segment}']
    )

    dose_mask = dose_to_pt(dose_file, tensor=True)

    ct_mask = convert_ct_array(
        f"/home/baumgartner/sgutwein84/container/output_{output_folder}/ct/{patient}", tensor=True)

    # radio_depth_mask = radiological_depth(
    #     np.array(ct_mask), egsinp_file, egsphant_file, tensor=True
    # )
    radio_depth_mask = torch.ones_like(dose_mask)

    center_mask = distance_center(
        egsinp_file, egsphant_file, ct_mask.shape, tensor=True
    )

    source_mask = distance_source(
        egsinp_file, egsphant_file, ct_mask.shape, tensor=True
    )

    binary_mask = create_binary_mask(
        egsinp_file, egsphant_file, beam_config_file, px_sp=np.array([1.171875, 1.171875, 3]), SID=1435, tensor=True
    )

    print(dose_mask.shape, ct_mask.shape, radio_depth_mask.shape,
          center_mask.shape, source_mask.shape, binary_mask.shape)

    # creates stack of size (5, 512  512, num_slices)
    stack = torch.stack((
        binary_mask,
        ct_mask,
        radio_depth_mask,
        center_mask,
        source_mask))

    if set_zero:
        # treshhold muss noch genau bestimmt werden
        mask = np.array(stack[1, :, :, :] > 150).astype(bool)
        for i in range(mask.shape[2]):
            mask[:, :, i] = ndimage.binary_fill_holes(
                mask[:, :, i]).astype(bool)
        mask = np.invert(mask)

        stack[0, mask] = 0
        stack[2, mask] = 0
        stack[3, mask] = 0
        stack[4, mask] = 0

    dose_mask = torch.unsqueeze(dose_mask, 0)

    stack = stack.float()
    dose_mask = dose_mask.float()

    torch.save(
        stack, f"/home/baumgartner/sgutwein84/container/training_data{output_folder}/{segment}/training_data.pt"
    )

    torch.save(
        dose_mask, f"/home/baumgartner/sgutwein84/container/training_data{output_folder}/{segment}/target_data.pt"
    )

    # shutil.move(f"/home/baumgartner/sgutwein84/container/{output_folder}/{segment}",
    #             f"/home/baumgartner/sgutwein84/container/calculated_segments/{segment}")

#print(stack.shape, dose_mask.shape)


if __name__ == "__main__":

    args = parse()
    create_mask_files(args.egsinp_file,
                      args.egsphant_file,
                      args.beam_config_file,
                      args.dose_file, args.segment,
                      args.output_folder)

    # debug
    # egsinp_file = "/Users/simongutwein/home/baumgartner/sgutwein84/container/output_20210522/p_0/p_0.egsinp"
    # egsphant_file = "/Users/simongutwein/home/baumgartner/sgutwein84/container/output_20210522/p_0/p_0.egsinp"
    # beam_config_file = "/Users/simongutwein/home/baumgartner/sgutwein84/container/output_20210522/p_0/beam_config_p_0.egsinp"
    # dose_file = "/Users/simongutwein/home/baumgartner/sgutwein84/container/output_20210522/p_0/p_0_1E06.3ddose"
    # segment = "p_0"
    # output_folder = "output_20210522"

    # create_mask_files(egsinp_file,
    #                   egsphant_file,
    #                   beam_config_file,
    #                   dose_file,
    #                   segment,
    #                   output_folder)
