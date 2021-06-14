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
    parser.add_argument(
        '--dir', type=str, default="/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/")

    args = parser.parse_args()

    return args


def clean_folder(filename, str_nhist, dir="/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/"):

    beam_folder = "/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/BEAM_MR-Linac/"

    beam_info = filename.split("_")[-1]

    par_files = glob.glob(dir + filename + "_w*")

    egsinp_files = glob.glob(dir + filename + ".*")
    egsinp_files = [
        x for x in egsinp_files if not "3ddose" in x and not "egsinp" in x]

    folders = glob.glob(dir + "*" + filename + "_" + "*nemo*")
    beam_folders = glob.glob(beam_folder + "*" + filename + "_" + "*nemo*")

    if not "x" in filename:
        temporary_beam_files = glob.glob(
            beam_folder + "beam_config_" + filename + "*.*")

    if "x" in filename:
        beam_file = "beam_config_" + beam_info + ".egsinp"
        beam_txt_file = "beam_config_" + beam_info + ".txt"
    else:
        beam_file = "beam_config_" + filename + ".egsinp"
        beam_txt_file = "beam_config_" + filename + ".txt"

    egsinp_file = filename + ".egsinp"
    dose_file = filename + ".3ddose"

    # if os.path.isfile(dir + egsinp_file):
    #     os.rename(dir + egsinp_file, dir + filename + "/" + egsinp_file)
    # if os.path.isfile(dir + dose_file):
    #     os.rename(dir + dose_file, dir + filename + "/" +
    #               filename + "_" + str_nhist + ".3ddose")

    # shutil.copyfile(beam_folder + beam_file, dir + filename + "/" + beam_file)
    # shutil.copyfile(beam_folder + beam_txt_file, dir +
    #                 filename + "/" + beam_txt_file)

    # shutil.move(dir + filename, dir + "output/" + filename)

    for file in par_files:
        os.remove(file)

    for file in egsinp_files:
        os.remove(file)

    for folder in folders:
        shutil.rmtree(folder)

    for folder in beam_folders:
        shutil.rmtree(folder)

    for file in temporary_beam_files:
        os.remove(file)

    print(
        f"The following files are deleted: {par_files}\n\n\nThe following files were deleted: {egsinp_files}\n\n\nThe following files were deleted: {temporary_beam_files}\n")

    return


if __name__ == "__main__":

    #args = parse()
    #clean_folder(args.filename, args.str_nhist, dir=args.dir)
    clean_folder(
        "p5_74", "1E10", dir="/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/")
    # import numpy as np
    # fz = 2
    # str_nhist = "1E03"

    # names = [f"p_{int(angle)}_{fz}x{fz}" for angle in np.linspace(0,360,8, endpoint=False)]

    # for name in names:
    #     clean_folder(name, str_nhist)
