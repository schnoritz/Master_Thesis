# imports
import argparse
import os
import torch
import numpy as np
import glob
import paramiko
import matplotlib.pyplot as plt

# argument parser for


def server_login():
    """logs into the BW-HPC server

    Returns:
        client: returns paramiko client
    """
    hostname = "134.2.168.52"
    username = "sgutwein84"
    password = "Derzauberkoenig1!"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=hostname, username=username, password=password)

    return client


def get_segments(dir):

    return [
        x for x in os.listdir(dir)
        if not x.startswith(".") and not "egsphant" in x and "_" in x
    ]


def create_mask_files(dir):

    if dir[-1] != "/":
        dir += "/"

    segments = get_segments(dir)

    for segment in segments:

        patient = segment.split("_")[0]
        output_folder = dir.split("/")[-2].split("_")[-1]

        egsinp_file = f"{dir}{segment}/{segment}.egsinp"
        egsphant_file = f"{dir}{patient}_listfile.txt.egsphant"
        beam_config_file = f"{dir}{segment}/beam_config_{segment}.txt"
        dose_file = glob.glob(f"{dir}{segment}/{patient}*.3ddose")[0]

        create_segment_job_file(
            egsinp_file, egsphant_file, beam_config_file, dose_file, segment, output_folder
        )

        execute_job_file()


def create_segment_job_file(
    egsinp_file, egsphant_file, beam_config_file, dose_file, segment, output_folder
):

    with open(
        "/Users/simongutwein/home/baumgartner/sgutwein84/container/job_template.sh", "r"
    ) as fin:
        lines = fin.readlines()

    task_line = lines[16]
    task_line = task_line.split()
    task_line.append(egsinp_file)
    task_line.append(egsphant_file)
    task_line.append(beam_config_file)
    task_line.append(dose_file)
    task_line.append(segment)
    task_line.append(output_folder)
    task_line.append("\n")
    lines[16] = " ".join(task_line)
    lines[18] = f'echo "Finished creating masks for {segment}"'
    lines.insert(1, f"#SBATCH --job-name '{segment}'\n")
    with open(
        "/Users/simongutwein/home/baumgartner/sgutwein84/container/job.sh", "w+"
    ) as fout:
        fout.writelines(lines)


def execute_job_file():

    client = server_login()

    _, stdout, _ = client.exec_command(
        f"cd container; sbatch job.sh"
    )

    for line in stdout:
        if line:
            job_id = line.split()[-1]

    print(f"JOB-ID: {job_id}")


if __name__ == "__main__":

    dir = "/home/baumgartner/sgutwein84/container/output_20210522/"

    create_mask_files(dir)
