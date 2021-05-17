import os
from pprint import pprint
import argparse
import shutil

def parse():
    parser = argparse.ArgumentParser(description='Clean dosxyznrc folder.')
    parser.add_argument('nhist', type=str, metavar='',
                        help='number of histories')
    parser.add_argument('config', type=str, metavar='', help='config name')
    parser.add_argument('num', type=str, metavar='', help='config name')
    parser.add_argument('--dir', type=str, default="/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc")
    parser.add_argument('--angle', type=str, default="None")
    args = parser.parse_args()

    return args

def clean_dosxyznrc_folder(nhist, config, num, dir="/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc", angle="None"):

    if dir[-1] != "/":
        dir += "/"

    ext_to_delete = ["errors", "egsdat", "egsinp", "egslst", "pardose"]

    dose_files, egsinp_files = get_moving_files(num, dir, angle)

    move_output(nhist, config, dir, dose_files, egsinp_files)

    remove_files_dosxyznrc(num, dir, ext_to_delete, angle)

    remove_folders_dosxyznrc(num, dir, angle)

    clean_beam_folder(num)

def clean_beam_folder(num):

    dir = "/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/BEAM_MR-Linac/"
    ext_to_delete = ["errors", "egsdat", "egslst", "pardose", "egslog"]
    files = [x for x in os.listdir(dir) if not x.startswith(".")]

    files_to_delete = [
        x for x in files if x.split(".")[-1] in ext_to_delete
    ]
    files_to_delete = [
        x for x in files_to_delete if x.split(".")[0].split("_")[2] == str(num)
    ]

    for file_ in files_to_delete:
        os.remove(dir + file_)

def move_output(nhist, config, dir, dose_files, egsinp_files):

    nhist_dose_files = []
    for file_ in dose_files:
        new_name = file_.split(".")[0] + "_" + str(nhist) + ".3ddose"
        nhist_dose_files.append(new_name)

    mr_folder = "/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/BEAM_MR-Linac/"

    for file_ in range(len(dose_files)):
        os.rename(dir + dose_files[file_], dir + "output/" + nhist_dose_files[file_])
        os.rename(dir + egsinp_files[file_],dir + "output/" + egsinp_files[file_])
        if "x" in config:
            shutil.copyfile(mr_folder + config + ".egsinp",
                            dir + "output/" + config + ".egsinp")
        else:
            os.rename(mr_folder + config + ".egsinp",
                      dir + "output/" + config + ".egsinp")

    print(f"The following files were moved {dose_files}, {egsinp_files}, {config}.egsinp")

def remove_folders_dosxyznrc(num, dir, angle):

    dirs = [x for x in os.listdir(dir) if not x.startswith(".") and "privat" in x]
    if "x" in num:
        dirs_to_delete = [
            x for x in dirs if x.split("_")[4] == str(num) and x.split("_")[4] == str(angle)
        ]
    else:
        dirs_to_delete = [x for x in dirs if x.split("_")[3] == str(num)]

    print(f"The following folders were deleted: {dirs_to_delete}")

    for folder in dirs_to_delete:
        shutil.rmtree(dir + folder)

def remove_files_dosxyznrc(num, dir, ext_to_delete, angle):

    files = [x for x in os.listdir(dir) if not x.startswith(".") and "_" in x]
    files = [x for x in files if x.split(".")[-1] in ext_to_delete]

    if "x" in num:
        files_to_delete = [
            x for x in files if x.split(".")[0].split("_")[2] == str(num)
            and x.split(".")[0].split("_")[1] == str(angle)
        ]
    else:
        files_to_delete = [
            x for x in files if x.split(".")[0].split("_")[1] == str(num)
        ]

    print(f"The following files were deleted: {files_to_delete}")
    for file_ in files_to_delete:
        os.remove(dir + file_)

def get_moving_files(num, dir, angle):

    dose_files = [
        x for x in os.listdir(dir) if not x.startswith(".") and ".3ddose" in x
        and x.split(".")[0].split("_")[-1] == str(num) 
    ]
    egsinp_files = [
        x for x in os.listdir(dir) if not x.startswith(".") and ".egsinp" in x
        and x.split(".")[0].split("_")[-1] == str(num)
    ]

    if angle != "None":
        dose_files = [ x for x in dose_files if x.split(".")[0].split("_")[1] == str(angle)]
        egsinp_files = [ x for x in egsinp_files if x.split(".")[0].split("_")[1] == str(angle)]

    return dose_files,egsinp_files


if __name__ == "__main__":

    args = parse()
    clean_dosxyznrc_folder(args.nhist, args.config, args.num, dir=args.dir, angle=args.angle)

    # clean_dosxyznrc_folder("1E02", "beam_config_2x2", "2x2", angle="315")
