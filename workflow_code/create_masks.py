# imports
import argparse
import os
import glob
import time


def parse():

    parser = argparse.ArgumentParser(description="Clean dosxyznrc folder.")

    parser.add_argument(
        "-dir",
        action='store',
        required=True,
        dest='input_dir'
    )

    parser.add_argument(
        "-p",
        nargs='+',
        action='store',
        dest='patients_list'
    )

    parser.add_argument(
        "-s",
        nargs='+',
        action='store',
        dest='segments_list'
    )

    args = parser.parse_args()

    return args


def get_segments(dir):

    return [
        x for x in os.listdir(dir)
        if not x.startswith(".") and not "egsphant" in x and "_" in x
    ]


def create_mask_files(input_dir, segments=None):

    if input_dir[-1] != "/":
        input_dir += "/"

    if not segments:
        segments = get_segments(input_dir)

    for segment in segments:

        patient = segment.split("_")[0]
        output_folder = input_dir.split("/")[-2].split("_")[-1]
        egsinp_file = f"{input_dir}{segment}/{segment}.egsinp"
        beam_config_file = f"{input_dir}{segment}/beam_config_{segment}.txt"
        dose_file = glob.glob(f"{input_dir}{segment}/{patient}*.3ddose")[0]

        create_segment_job_file(
            egsinp_file, beam_config_file, dose_file, segment, output_folder
        )

        job_id = execute_job_file()
        print(f"JOB-ID for {segment}: {job_id}")

        with open("/home/baumgartner/sgutwein84/container/job_overview.txt", "a") as fout:
            print(f"JOB-ID for {segment}: {job_id}", file=fout)


def create_segment_job_file(
    egsinp_file, beam_config_file, dose_file, segment, output_folder
):

    with open(
        "/home/baumgartner/sgutwein84/container/job_template.sh", "r"
    ) as fin:
        lines = fin.readlines()

    task_line = lines[16]
    task_line = task_line.split()
    task_line.append(egsinp_file)
    task_line.append(beam_config_file)
    task_line.append(dose_file)
    task_line.append(segment)
    task_line.append(output_folder)
    task_line.append("\n")
    lines[16] = " ".join(task_line)
    lines[18] = f'echo "Finished creating masks for {segment}"'
    lines.insert(1, f"#SBATCH --job-name '{segment}'\n")
    with open(
        "/home/baumgartner/sgutwein84/container/job.sh", "w+"
    ) as fout:
        fout.writelines(lines)


def execute_job_file():

    stream = os.popen(
        "cd /home/baumgartner/sgutwein84/container;sbatch job.sh")
    out = stream.read()

    return out.split()[-1]


if __name__ == "__main__":

    args = parse()

    if args.patients_list:

        segments = []
        for patient in args.patients_list:
            if patient[-1] != "_":
                patient += "_"
            segments.extend([x for x in os.listdir(
                args.input_dir) if not "ct" in x and not "egsphant" in x and not x.startswith(".") and patient in x])

    elif args.segments_list:
        segments = args.segments_list

    else:
        segments = None

    create_mask_files(args.input_dir, segments)
