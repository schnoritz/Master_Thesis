import os
from pprint import pprint
import argparse
import shutil

parser = argparse.ArgumentParser(description='Clean dosxyznrc folder.')
parser.add_argument('angle', type=int, metavar='', help='angle to define deleted files')
parser.add_argument('--dir', type=str, default="/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc")
args = parser.parse_args()

def clean_dosxyznrc_folder(angle, dir="/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc"):

    if dir[-1] != "/":
        dir += "/"

    ext_to_delete = ["errors", "egsdat", "egsinp", "egslst", "pardose"]

    files = [x for x in os.listdir(dir) if not x.startswith(".") and "_" in x]
    files = [x for x in files if x.split("_")[1] == str(angle)]
    files_to_delete = [x for x in files if x.split(".")[-1] in ext_to_delete]
    for file_ in files_to_delete:
        os.remove(dir + file_)

    dirs = [x for x in os.listdir(dir) if not x.startswith(".") and "privat" in x]
    dirs_to_delete = [x for x in dirs if x.split("_")[3] == str(angle)]
    for folder in dirs_to_delete:
        shutil.rmtree(dir + folder)

    dose_files = [x for x in os.listdir(
        dir) if not x.startswith(".") and ".3ddose" in x]
    print(dose_files)
    for file_ in dose_files:
        os.rename(dir + file_, dir + "output/" + file_)

if __name__ == "__main__":

    clean_dosxyznrc_folder(args.angle, args.dir)
