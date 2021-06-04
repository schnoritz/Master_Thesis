import os
import argparse
import shutil
import glob

def parse():
    parser = argparse.ArgumentParser(description='Clean dosxyznrc folder.')
    parser.add_argument('filename', type=str, metavar='',
                        help='filename')
    parser.add_argument('str_nhist', type=str, metavar='',
                        help='number of histories')
    parser.add_argument('--dir', type=str, default="/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc/")

    args = parser.parse_args()

    return args

def clean_folder(filename, str_nhist, dir="/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc/"):

    beam_folder = "/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/BEAM_MR-Linac/"

    beam_info = filename.split("_")[-1]

    patient = filename.split("_")[0]

    par_files = glob.glob(dir + filename + "_*")
    egsinp_files = glob.glob(dir + filename + ".*")
    egsinp_files = [x for x in egsinp_files if not "3ddose" in x and not "egsinp" in x]

    folders = glob.glob(dir + "*" + filename + "*nemo*")
    beam_folders = glob.glob(beam_folder + "*" + filename + "*nemo*")

    beam_file = "beam_config_" + beam_info + ".egsinp"

    beam_txt_file = "beam_config_" + beam_info + ".txt"

    egsinp_file = filename + ".egsinp"

    for file in par_files:
        os.remove(file)

    for file in egsinp_files:
        os.remove(file)

    for folder in folders:
        shutil.rmtree(folder)

    for folder in beam_folders:
        shutil.rmtree(folder)


    os.rename(dir + egsinp_file, dir + filename + "/" + egsinp_file)
    shutil.copyfile(beam_folder + beam_file, dir + filename + "/" + beam_file)
    shutil.copyfile(beam_folder + beam_txt_file, dir + filename + "/" + beam_txt_file)
    os.rename(dir + filename + ".3ddose", dir + filename + "/" + filename + "_" + str_nhist + ".3ddose")


    return


if __name__ == "__main__":

    args = parse()
    clean_folder(args.filename, args.str_nhist, dir=args.dir)

    # import numpy as np
    # fz = 2
    # str_nhist = "1E03"

    # names = [f"p_{int(angle)}_{fz}x{fz}" for angle in np.linspace(0,360,8, endpoint=False)]

    # for name in names:
    #     clean_folder(name, str_nhist)
