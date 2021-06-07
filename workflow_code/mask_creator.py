import argparse
from pt_3ddose import dose_to_pt
from plot_array import plot_array
from radiological_depth import radiological_depth
from binary_mask import create_binary_mask
from pt_ct import convert_ct_array
from dist_center import distance_center
from dist_source import distance_source
import numpy as np
import torch
import shutil
import subprocess


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


def create_mask_files(egsinp_file,
                      egsphant_file, beam_config_file, dose_file, segment, output_folder):

    subprocess.run(
        ['mkdir',
            f'/home/baumgartner/sgutwein84/container/training_data/{segment}']
    )

    dose_mask = dose_to_pt(dose_file, tensor=True)

    ct_mask = np.load(
        "/home/baumgartner/sgutwein84/container/output_20210522/p/p.npy"
    )
    #ct_mask = convert_ct_array("/home/baumgartner/sgutwein84/container/DeepDosePC1-4", tensor=True)

    radio_depth_mask = radiological_depth(
        "/home/baumgartner/sgutwein84/container/output_20210522/p/p.npy", egsinp_file, egsphant_file, tensor=True
    )
    #radio_depth_mask = torch.ones((512, 512, 110))

    center_mask = distance_center(
        egsinp_file, egsphant_file, ct_mask.shape, tensor=True
    )

    source_mask = distance_source(
        egsinp_file, egsphant_file, ct_mask.shape, tensor=True
    )

    binary_mask = create_binary_mask(
        egsinp_file, egsphant_file, beam_config_file, px_sp=np.array([1.171875, 1.171875, 3]), SID=1435, tensor=True
    )
    # creates stack of size (5, 512  512, num_slices)
    stack = torch.stack((binary_mask, torch.tensor(ct_mask), radio_depth_mask,
                        center_mask, source_mask))

    torch.save(
        stack, f"/home/baumgartner/sgutwein84/container/training_data/{segment}/training_data.pt"
    )

    torch.save(
        dose_mask, f"/home/baumgartner/sgutwein84/container/training_data/{segment}/target_data.pt"
    )

    # shutil.move(f"/home/baumgartner/sgutwein84/container/{output_folder}/{segment}",
    #             f"/home/baumgartner/sgutwein84/container/calculated_segments/{segment}")

#print(stack.shape, dose_mask.shape)


if __name__ == "__main__":

    args = parse()
    create_mask_files(args.egsinp_file,
                      args.egsphant_file, args.beam_config_file, args.dose_file, args.segment, args.output_folder)
