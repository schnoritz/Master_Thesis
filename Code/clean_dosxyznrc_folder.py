import os
from pprint import pprint
import argparse
import shutil

def parse():
    parser = argparse.ArgumentParser(description='Clean dosxyznrc folder.')
    parser.add_argument('angle', type=int, metavar='', help='angle to define deleted files')
    parser.add_argument('nhist', type=str, metavar='',
                        help='number of histories')
    parser.add_argument('--dir', type=str, default="/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc")
    args = parser.parse_args()

    return args

def clean_dosxyznrc_folder(angle, nhist, dir="/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc"):

    if dir[-1] != "/":
        dir += "/"

    ext_to_delete = ["errors", "egsdat", "egsinp", "egslst", "pardose"]

    files = [x for x in os.listdir(dir) if not x.startswith(".") and "_" in x]
    files = [x for x in files if x.split("_")[1] == str(angle)]
    files_to_delete = [x for x in files if x.split(".")[-1] in ext_to_delete]
    print(f"The following files were deleted: {files_to_delete}")
    for file_ in files_to_delete:
        os.remove(dir + file_)

    dirs = [x for x in os.listdir(dir) if not x.startswith(".") and "privat" in x]
    dirs_to_delete = [x for x in dirs if x.split("_")[3] == str(angle)]
    print(f"The following folders were deleted: {dirs_to_delete}")
    for folder in dirs_to_delete:
        shutil.rmtree(dir + folder)

    dose_files = [x for x in os.listdir(
        dir) if not x.startswith(".") and ".3ddose" in x]
    
    print(dose_files)

    nhist_dose_files = []
    for file_ in dose_files:
        new_name = file_.split(".")[0] + "_" + str(nhist) + ".3ddose"
        nhist_dose_files.append(new_name)

    for file_ in range(len(dose_files)):
        os.rename(dir + dose_files[file_], dir + "output/" + nhist_dose_files[file_])

    print(f"The following files were created: {nhist_dose_files}")

if __name__ == "__main__":

    args = parse()
    clean_dosxyznrc_folder(args.angle, args.nhist, args.dir)

