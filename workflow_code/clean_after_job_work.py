import os
import argparse
import shutil
import glob


def parse():
    parser = argparse.ArgumentParser(description='Clean dosxyznrc folder.')
    parser.add_argument('filename', type=str, metavar='',
                        help='filename')
    parser.add_argument('str_nhist', type=str, metavar='', hä
                        help='number of histories')
    parser.add_argument(
        '--dir', type=str, default="/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/")

    args = parser.parse_args()

    return args


def clean_folder(filename, str_nhist, dir="/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/"):

    beam_folder = "/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/BEAM_MR-Linac/"

    beam_info = filename.split("_")[-1]

    # selects all files like p0_0_w1.[suffix]
    par_files = glob.glob(dir + filename + "_w*")

    # selects all files which are from original main file with various suffixes like .egsdat etc
    main_files = glob.glob(dir + filename + ".*")
    main_files = [
        x for x in main_files if not "3ddose" in x and not "egsinp" in x]

    # folders, might be unneccecary since folder should be deleted after job execution
    folders = glob.glob(dir + "*" + filename + "_" + "*nemo*")
    beam_folders = glob.glob(beam_folder + "*" + filename + "_" + "*nemo*")

    # selects all files in BEAM_MR-Linac folder to be deleted all files with _w[num]
    # and main files, can be deleted since they are copied lated in the code
    if not "x" in filename:
        temporary_beam_files = glob.glob(
            beam_folder + "beam_config_" + filename + "_*.*")
        temporary_beam_files.extend(glob.glob(
            beam_folder + "beam_config_" + filename + ".*"))

    # if test file beaminfo is important part for since testfields are all the same for
    # different patients filename else just segment name is used
    if "x" in filename:
        beam_file = "beam_config_" + beam_info + ".egsinp"
        beam_txt_file = "beam_config_" + beam_info + ".txt"
    else:
        beam_file = "beam_config_" + filename + ".egsinp"
        beam_txt_file = "beam_config_" + filename + ".txt"

    # definition of egsinp filename and dose filename
    egsinp_file = filename + ".egsinp"
    dose_file = filename + ".3ddose"

    # test if needed files are still present or deleted in previous steps or previous jobs
    assert os.path.isfile(
        dir + egsinp_file), ".egsinp file got somehow deleted"
    assert os.path.isfile(
        dir + dose_file), ".3ddose file got somehow deleted"
    assert os.path.isfile(
        beam_folder + beam_file), "beamfile .egsinp file got somehow deleted"
    assert os.path.isfile(
        beam_folder + beam_txt_file), "beamfile .txt file got somehow deleted"

    # move egsinp and 3ddose file
    os.rename(dir + egsinp_file, dir + filename + "/" + egsinp_file)
    os.rename(dir + dose_file, dir + filename + "/" +
              filename + "_" + str_nhist + ".3ddose")

    # copy the 2 main beamconfig files to the segment folder
    shutil.copyfile(beam_folder + beam_file, dir + filename + "/" + beam_file)
    shutil.copyfile(beam_folder + beam_txt_file, dir +
                    filename + "/" + beam_txt_file)
    # move folder to outout destination
    shutil.move(dir + filename, dir + "output/" + filename)

    # delete all unneccesary files
    for file in par_files:
        os.remove(file)

    for file in main_files:
        os.remove(file)

    for folder in folders:
        shutil.rmtree(folder)

    for folder in beam_folders:
        shutil.rmtree(folder)

    for file in temporary_beam_files:
        os.remove(file)

    # print all files that were deleted
    print(
        f"The following files are deleted: {par_files}\n\n\nThe following files were deleted: {main_files}\n\n\nThe following files were deleted: {temporary_beam_files}\n")

    return


if __name__ == "__main__":

    args = parse()
    clean_folder(args.filename, args.str_nhist, dir=args.dir)
