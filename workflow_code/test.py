import matplotlib.pyplot as plt
import torch
from pt_3ddose import dose_to_pt
from pt_ct import convert_ct_array
import numpy as np
from pymedphys import gamma
import os
from pprint import pprint


if __name__ == "__main__":

    dir = "/mnt/qb/baumgartner/sgutwein84/output_nodes/"
    files = [x for x in os.listdir(dir) if not x.startswith(".")]
    files.remove('ct')

    all_files = []
    for file in files:
        all_files.extend([dir + file + "/" + x for x in os.listdir(dir+file)])

    new_files = []
    for file in all_files:
        filename = file.split("/")
        filename[-1] = filename[-1].replace("k", "n")
        new_files.append("/".join(filename))

    pprint(new_files)

    for old_file, new_file in zip(all_files, new_files):
        os.rename(old_file, new_file)
