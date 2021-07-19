# imports
import argparse
import os
import torch
import numpy as np
import glob
import paramiko
import matplotlib.pyplot as plt

from pprint import pprint


def server_login():
    """logs into the ML-Cloud

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


def create_mask_files(dir, segments=False):

    if dir[-1] != "/":
        dir += "/"

    if segments == False:
        segments = get_segments(dir)

    with server_login() as client:
        for num, segment in enumerate(segments):

            patient = segment.split("_")[0]
            output_folder = dir.split("/")[-2].split("_")[-1]
            egsinp_file = f"{dir}{segment}/{segment}.egsinp"
            beam_config_file = f"{dir}{segment}/beam_config_{segment}.txt"
            dose_file = glob.glob(f"{dir}{segment}/{patient}*.3ddose")[0]

            create_segment_job_file(
                egsinp_file, beam_config_file, dose_file, segment, output_folder
            )

            execute_job_file(segment, client)


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


def execute_job_file(segment, client):

    _, stdout, _ = client.exec_command(
        f"cd container; sbatch job.sh"
    )

    for line in stdout:
        if line:
            job_id = line.split()[-1]

    print(f"JOB-ID: {job_id} for segment: {segment}")


if __name__ == "__main__":

    dir = "/home/baumgartner/sgutwein84/container/output_prostate"
    patients = ["p0_", "p1_", "p2_", "p3_", "p4_", "p5_", "p7_", "p8_", "p9_"]
    segments = []
    for patient in patients:
        segments.extend([x for x in os.listdir(
            dir) if not "ct" in x and not "egsphant" in x and not x.startswith(".") and patient in x])

    pprint(segments)

    create_mask_files(dir, segments)
