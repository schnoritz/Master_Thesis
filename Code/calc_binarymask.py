import os
import argparse
import numpy as np
from binarymask import create_binary_mask
import torch
import sys

def parse():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('x', type=str, metavar='', help='')
    parser.add_argument('y', type=str, metavar='', help='')
    parser.add_argument('z', type=str, default="")

    args = parser.parse_args()

    return args

def calc_binarymask(dir="/home/baumgartner/sgutwein84/container/output/"):

    all_files = os.listdir(dir)

    files = [x for x in all_files if not x.startswith(".") and not x.endswith(".egsphant")]
    fz = [x.split(".")[0].split("_")[-1] for x in files]

    egsinp_files = [x + "/" + x + ".egsinp" for x in files]
    beam_config_files = [files[i] + "/beam_config_" +  fz[i] + ".txt" for i in range(len(files))]

    egsphant_file = [x for x in all_files if x.endswith(".egsphant")][0]

    for num in range(len(egsinp_files)):

        egsinp = dir + egsinp_files[num]
        beam_config = dir + beam_config_files[num]
        egsphant = dir + egsphant_file
        save_path = "/".join(egsinp.split("/")[:-1]) + "/"
        angle_fz_config = "_".join(files[num].split("_")[1:])

        mask = create_binary_mask(egsphant=egsphant, egsinp=egsinp, beam_config=beam_config)
        
        t_mask = torch.from_numpy(mask).float()

        print(sys.getsizeof(t_mask), sys.getsizeof(mask))
        torch.save(t_mask, save_path + "binary_mask_" + angle_fz_config + ".pt")

        print(f"Saved to {save_path}binary_mask_{angle_fz_config}.pt")

if __name__ == "__main__":
    
    calc_binarymask()

        
