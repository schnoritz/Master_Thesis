import os
from pprint import pprint
import argparse
import shutil

parser = argparse.ArgumentParser(description='Clean MR-Linac folder.')
parser.add_argument(
    '--dir', type=str, default="/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/BEAM_MR-Linac")
args = parser.parse_args()

def clean_dosxyznrc_folder(dir="/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/BEAM_MR-Linac"):

    if dir[-1] != "/":
        dir += "/"

    ext_to_delete = ["errors", "egsdat", "egslst", "pardose", "egslog"]

    files = [x for x in os.listdir(dir) if not x.startswith(".")]
    files_to_delete = [x for x in files if x.split(".")[-1] in ext_to_delete]
    for file_ in files_to_delete:
        os.remove(dir + file_)

    dirs_to_delete = [x for x in os.listdir(dir) if not x.startswith(".") and "privat" in x]
    for folder in dirs_to_delete:
        shutil.rmtree(dir + folder)

if __name__ == "__main__":

    clean_dosxyznrc_folder(args.dir)
