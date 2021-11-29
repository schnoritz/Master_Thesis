#   Skript welches aus gegebenen CT Bildern einen Jobs in EGSnrc an den BW-HPC Nemo CLuster sendet.
#   Individuelle Eingabe von CT-Bildern, Gantry-Winkel, Anzahl der Simulierten Teilchen und des genutzten Strahls

import os
import shutil
import paramiko
import random
import numpy as np
from pydicom import dcmread, uid
from natsort import natsorted
import math

"""
!!!-----------------------------------------------WICHTIG------------------------------------------------------!!!
"""


global DEIN_USERNAME
DEIN_USERNAME = "tu_zxoys08"  # hier deinen Username eingeben"  # war bei mir "tu_zxoys08"

global DEIN_USER_PASSWORD
DEIN_USER_PASSWORD = "Derzauberkoenig1!"  # "hier deinen Username Passwort eingeben"

global PLAN_FILES_PATH
PLAN_FILES_PATH = "/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/planfiles/"  # hier auf deinen Pfad anpassen

global DOSE_FILES_PATH
DOSE_FILES_PATH = "/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/dosefiles/"  # hier auf deinen Pfad anpassen

global DOSXYZNRC_PATH
DOSXYZNRC_PATH = "/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/"

global JOBS_PATH
JOBS_PATH = "/home/tu/tu_tu/tu_zxoys08/EGSnrc/jobs"
"""
!!!-----------------------------------------------WICHTIG------------------------------------------------------!!!
"""


def server_login():
    """logs into the BW-HPC server

    Returns:
        client: returns paramiko client 
    """
    hostname = "login1.nemo.uni-freiburg.de"
    username = DEIN_USERNAME
    password = DEIN_USER_PASSWORD
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=hostname, username=username, password=password)

    return client


def create_listfile(path, patient):
    """creates a listfile in the directory of the CT images
    Args:
        path (str):       path to the CT images wanted to be used as phantom data
    """
    used_files = [file_ for file_ in natsorted(os.listdir(path)) if not file_.startswith(".") and not ".txt" in file_]

    with open(os.path.join(path, f'{patient}_listfile.txt'), 'w+') as fout:
        for filename in used_files:
            fout.write(path + filename + '\n')


def create_ctcreate_file(dose_file_path, dcm_folder):
    """creates the file needed for the ctcreate command in EGSnrc

    Args:
        path (str):                     path to the dosxyznrc directory
        dose_file_path (str):           path to the dose file used for EGSnrc
        dcm_path (str):                 path to the directory of the used CT images
    """
    dose_file_dat = dcmread(dose_file_path, force=True)
    patient = dcm_folder.split("/")[0]
    with open(os.path.join(DOSXYZNRC_PATH, f'{patient}_ctcreate_file.txt'), 'w+') as fout:

        dose_file_dat.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
        image_position = np.array(dose_file_dat.ImagePositionPatient)/10
        dose_dimensions = dose_file_dat.pixel_array.shape
        pixel_spacing = np.array(dose_file_dat.PixelSpacing)/10

        xlower = image_position[0]-pixel_spacing[0]/2
        ylower = image_position[1]-pixel_spacing[1]/2
        zlower = image_position[2]-pixel_spacing[0]/2

        xupper = xlower + dose_dimensions[2]*pixel_spacing[0]
        yupper = ylower + dose_dimensions[1]*pixel_spacing[1]
        zupper = zlower + dose_dimensions[0]*pixel_spacing[0]

        fout.write("DICOM \n")
        fout.write(DOSXYZNRC_PATH + dcm_folder + f"{patient}_listfile.txt\n")
        fout.write(str("%.4f" % xlower) + ", " + str("%.4f" % xupper) +
                   ", " + str("%.4f" % ylower) + ", " + str("%.4f" % yupper) +
                   ", " + str("%.4f" % zlower) + ", " + str("%.4f" % zupper) + "\n")

        fout.write(str(pixel_spacing[0]) + ", " +
                   str(pixel_spacing[1]) + ", " + str(pixel_spacing[0]) + "\n")

        fout.write("0, 0 \n")


def execute_ct_create(client, path, dcm_folder):
    """executes the ctcreate command on the given server

    Args:
        local_dosxyznrc_path ([type]): path to the local drectory of dosxyznrc
    """
    patient = dcm_folder.split("/")[-2]
    if not os.path.isfile(path + f"{patient}_listfile.txt.egsphant"):
        #os.remove(path + "listfile.txt.egsphant")

        _, stdout, _ = client.exec_command(f'cd {path}; \
            ctcreate {patient}_ctcreate_file.txt -p 700icru')

        for line in stdout:
            continue

        print("Finished: .egsphant file was created.")

    else:

        print("Finished: .egsphant still exists.")

    return path + f"{patient}_listfile.txt.egsphant"


def create_egsinp_file(path, pos_x, pos_y, pos_z, angle, beam_config, n_histories, iparallel, filename):
    """creates the needed .egsinp file which is needed to combine the created .pardose files in 
       parallel calculation. 

    Args:
        path (str):                     path to the local dosxyznrc directory
        pos_x (str):                    iso position of the phamtom in x direction
        pos_y (str):                    iso position of the phamtom in y direction
        pos_z (str):                    iso position of the phamtom in z direction
        angle (float):                  desired angle of gantry for simulation
        beam_config (str):              name of the wanted beam configuration placed inside "BEAM_MR-Linac" folder
        n_histories (int):              number of simulated histories
        iparallel (int):                number of parallel jobs

    Returns:
        str: the lines from the .egsinp file created in this function
    """

    patient = filename.split("_")[0]
    with open(path + filename + '.egsinp', 'w+') as fout:

        fout.write(f'CT Phantom from {patient}_listfile.txt.egsphant\n')
        fout.write('0\n')
        fout.write(f'$EGS_HOME/dosxyznrc/{patient}_listfile.txt.egsphant\n')
        fout.write('0.7, 0.01, 0\n')
        fout.write('1, 0, 0,\n')

        fout.write('2, 9, ' +
                   str(np.round(pos_x, 4)) + ', ' +
                   str(np.round(pos_y, 4)) + ', ' +
                   str(np.round(pos_z, 4)) + ', 90.00, ' + format(angle, '.2f') + ', 30, 270, 1, 40\n')

        fout.write('2, 0, 2, 0, 0, 0, 0, 0\n')
        fout.write('BEAM_MR-Linac,' + beam_config + ',521ELEKTA\n')

        fout.write(str(int(n_histories)) + ", 0, 500," +
                   str(random.randint(0, 10000)) + "," +
                   str(random.randint(0, 10000)) + ", 100.0, 1, 4, 0, 1, 2.0, 0, 0, 0, 40, 0, 0\n")

        fout.write(' #########################\n')
        fout.write(' :Start MC Transport Parameter:\n')
        fout.write(' \n')
        fout.write(' Global ECUT= 0.7\n')
        fout.write(' Global PCUT= 0.01\n')
        fout.write(' Global SMAX= 5\n')
        fout.write(' Magnetic Field= 0.0, 0.0, -1.5\n')
        fout.write(' EM ESTEPE= 0.02\n')
        fout.write(' ESTEPE= 0.02\n')
        fout.write(' XIMAX= 0.5\n')
        fout.write(' Boundary crossing algorithm= PRESTA-I\n')
        fout.write(' Skin depth for BCA= 0\n')
        fout.write(' Electron-step algorithm= PRESTA-II\n')
        fout.write(' Spin effects= On\n')
        fout.write(' Brems angular sampling= Simple\n')
        fout.write(' Brems cross sections= BH\n')
        fout.write(' Bound Compton scattering= Off\n')
        fout.write(' Compton cross sections= default\n')
        fout.write(' Pair angular sampling= Simple\n')
        fout.write(' Pair cross sections= BH\n')
        fout.write(' Photoelectron angular sampling= Off\n')
        fout.write(' Rayleigh scattering= Off\n')
        fout.write(' Atomic relaxations= Off\n')
        fout.write(' Electron impact ionization= Off\n')
        fout.write(' Photon cross sections= xcom\n')
        fout.write(' Photon cross-sections output= Off\n')
        fout.write(' \n')
        fout.write(' :Stop MC Transport Parameter:\n')
        fout.write(' #########################\n')
        fout.close()

    with open(path + filename + '.egsinp', 'r') as fout:
        lines = fout.readlines()

    return lines


def create_parallel_files(egsinp_lines, path, pos_x, pos_y, pos_z, angle,
                          n_histories, iparallel, filename):
    """creates the needed .egsinp files which are needed to calculate the .pardose files
       which will be recombined to .3ddose file
    Args:
        egsinp_lines (list):            given list of text from .egsinp file
        path (str):                     path to the dosxyznrc directory
        pos_x (str):                    iso position of the phamtom in x direction
        pos_y (str):                    iso position of the phamtom in y direction
        pos_z (str):                    iso position of the phamtom in z direction
        angle (float):                  desired angle of gantry for simulation
        n_histories (int):              number of simulated histories
        iparallel (int):                number of parallel jobs
     """

    position_angle_line = egsinp_lines[5].split(",")
    position_angle_line[6] = " " + format(angle, '.2f')
    position_angle_line[2] = " " + str(np.round(pos_x, 4))
    position_angle_line[3] = " " + str(np.round(pos_y, 4))
    position_angle_line[4] = " " + str(np.round(pos_z, 4))
    egsinp_lines[5] = ",".join(position_angle_line)

    parallel_line = egsinp_lines[8].split(",")
    parallel_line[0] = str(int(n_histories/iparallel))
    parallel_line[12] = " " + str(iparallel)
    parallel_line[7] = " 0"

    # files = glob(path + "*" + str(angle) + "_w*.*")
    # if files:
    #     for name in files:
    #         if not "egsrun" in name:
    #             os.remove(name)

    for i in range(iparallel):
        with open(path + filename + '_w' + str(i+1) + '.egsinp', 'w+') as fout:

            parallel_line[13] = " " + str(i+1)
            parallel_line[3] = " " + str(random.randint(0, 10000))
            parallel_line[4] = " " + str(random.randint(0, 10000))
            egsinp_lines[8] = ','.join(parallel_line)

            file_text = "".join(egsinp_lines)
            fout.write(file_text)


def create_job_file(jobs_path, iparallel, nodes, ppn, filename, n_histories):
    """creates the job file which can be executed for parallel simulation

    Args:
        jobs_path (str):       path to the directory where jobs are stored
        local_dcm_path (str):  path to the directory where CT are located
        iparallel (int):       number of desired parallel simulations
    """

    with open(jobs_path + '/job.sh', 'w+') as fout:

        fout.write('#!/bin/bash\n')
        fout.write("#MSUB -l nodes=" + str(nodes) + ":ppn=" + str(ppn) + "\n")
        fout.write(f'#MSUB -N  {filename} calculation\n')
        fout.write('#MSUB -l walltime=1:00:00:00\n')
        fout.write('#MSUB -l pmem=6gb\n')
        fout.write("#MSUB -o /home/tu/tu_tu/tu_zxoys08/EGSnrc/jobs\n")
        fout.write("#MSUB -m a\n\n")
        command = []

        for i in range(iparallel):

            command.append("dosxyznrc -i " + filename + "_w" +
                           str(i+1) + ".egsinp -p 700icru")

        command = " & ".join(command) + "\n\nwait\n\nsleep 2m\n\n"
        command += "dosxyznrc -i " + filename + ".egsinp" + " -p 700icru \n\nwait\n\n"

        command += 'sleep 2m\n\n'
        command += 'echo "Start cleaning!"\n\n'
        str_nhist = f"{n_histories:.0e}".upper().replace("+", "")

        command += f"python3 clean_after_job_work.py {filename} {str_nhist}\n\nwait"

        fout.write(command)


def execute_job_file(client, filename):
    """executes the created job on the BW-HPC cluster 
    Args:
        client (client): paramiko client where the command is executed
    """
    client.exec_command('cd EGSnrc/jobs; chmod +x ./job.sh')
    _, stdout, _ = client.exec_command('msub ./EGSnrc/jobs/job.sh')

    for string in stdout:
        if string.strip():
            job_id = string.strip()
            print(f"Job-ID: {job_id} for segment {filename}")


def create_entire_job(n, gantry, par_jobs, ppn, nodes, beam_config, patient, iso_center=None):

    dcm_folder = patient + "/"  # select folder located in "dosxyznrc" folder
    beam_info = beam_config.split("_")[-1]

    if iso_center is not None:
        target_filename = patient + "_" + beam_info
    else:
        target_filename = patient + "_" + \
            str(int(gantry)-270) + "_" + beam_info

    client = server_login()

    dose_file_path = os.path.join(DOSXYZNRC_PATH, f"dosefiles/{patient}_dose.dcm")

    job_folder = os.path.join(DOSXYZNRC_PATH, target_filename)

    if os.path.isdir(job_folder):
        shutil.rmtree(job_folder)
    os.mkdir(job_folder)

    create_listfile(os.path.join(DOSXYZNRC_PATH, dcm_folder), patient)
    create_ctcreate_file(dose_file_path, dcm_folder)
    egsphant_path = execute_ct_create(client, DOSXYZNRC_PATH, dcm_folder)

    if iso_center is None:

        center_coordinates = get_center(egsphant_path)

        shifted_center = np.squeeze(center_coordinates +
                                    np.random.normal(0, 1, size=(1, 3)))

        iso_x, iso_y, iso_z = shifted_center[0], shifted_center[1], shifted_center[2]

    else:
        iso_x, iso_y, iso_z = iso_center[0], iso_center[1], iso_center[2]

    lines = create_egsinp_file(DOSXYZNRC_PATH, iso_x, iso_y, iso_z, gantry,
                               beam_config, n, par_jobs, target_filename)

    create_parallel_files(lines, DOSXYZNRC_PATH, iso_x, iso_y, iso_z, gantry,
                          n, par_jobs, target_filename)

    create_job_file(JOBS_PATH, par_jobs, nodes, ppn,
                    target_filename, n)

    execute_job_file(client, target_filename)

    client.close()


def get_center(egsphant):

    with open(egsphant, "r") as fin:

        for i in range(7):
            fin.readline()

        x = fin.readline().split()
        y = fin.readline().split()
        z = fin.readline().split()

        x_first, x_last = float(x[0]), float(x[-1])
        y_first, y_last = float(y[0]), float(y[-1])
        z_first, z_last = float(z[0]), float(z[-1])

        center = np.array(
            [(x_last+x_first)/2, (y_last+y_first)/2, (z_last+z_first)/2])

        return center


def extract_plan_infos(plan_file):

    leafes = []
    jaws = []
    isocenters = []
    angles = []

    dat = dcmread(plan_file, force=True)

    for beam in dat.BeamSequence:
        angles.extend(
            [float(beam.ControlPointSequence[0].GantryAngle)] * (len(beam.ControlPointSequence)//2))
        isocenters.extend(
            [beam.ControlPointSequence[0].IsocenterPosition] * (len(beam.ControlPointSequence)//2))

        for num, seq in enumerate(beam.ControlPointSequence):
            if num % 2 == 0:
                jaws.append(
                    seq.BeamLimitingDevicePositionSequence[0].LeafJawPositions)
                leafes.append(
                    seq.BeamLimitingDevicePositionSequence[1].LeafJawPositions)

        assert len(leafes) == len(jaws) == len(isocenters) == len(angles), "Number of Leafes, Jaws , Isocenters or Angles not correct."

    jaws = np.array(jaws)
    leafes = np.array(leafes)
    angles = np.array(angles)
    isocenters = np.array(isocenters)/10

    return jaws, leafes, angles, isocenters


def calculate_new_mlc(leafes, radius=41.5, ssd=143.5, cil=35.77-0.09):

    MLC_egsinp = np.zeros((2, 80))

    # calculate new MLC Positions for egsinp
    for j in range(2):
        for i in range(80):
            if j == 0:
                if leafes[j, i] <= 0:
                    MLC_egsinp[j, i] = (((cil + math.sqrt(pow(radius, 2)-pow(math.cos(abs(leafes[j, i]*10)*0.1/ssd)*radius, 2)))
                                        * abs(leafes[j, i]*10)*0.1/ssd + math.cos(abs(leafes[j, i]*10)*0.1/ssd)*radius)*(-1))
                if leafes[j, i]*10 > 0:
                    MLC_egsinp[j, i] = ((-(cil - math.sqrt(pow(radius, 2)-pow(math.cos(abs(leafes[j, i]*10)*0.1/ssd)*radius, 2)))
                                        * abs(leafes[j, i]*10)*0.1/ssd + math.cos(abs(leafes[j, i]*10)*0.1/ssd)*radius)*(-1))
            else:
                if leafes[j, i]*10 >= 0:
                    MLC_egsinp[j, i] = ((cil + math.sqrt(pow(radius, 2)-pow(math.cos(abs(leafes[j, i]*10)*0.1/ssd)*radius, 2)))
                                        * abs(leafes[j, i]*10)*0.1/ssd + math.cos(abs(leafes[j, i]*10)*0.1/ssd)*radius)
                if leafes[j, i]*10 < 0:
                    MLC_egsinp[j, i] = (-(cil - math.sqrt(pow(radius, 2)-pow(math.cos(abs(leafes[j, i]*10)*0.1/ssd)*radius, 2)))
                                        * abs(leafes[j, i]*10)*0.1/ssd + math.cos(abs(leafes[j, i]*10)*0.1/ssd)*radius)

    return MLC_egsinp


def calculate_new_jaw(jaws, cil=44.35 - 0.09, radius=13.0, ssd=143.5):

    new_JAWS = np.zeros(2)

    # calculate new JAW Positions for egsinp
    if jaws[0] <= 0:
        new_JAWS[0] = ((cil + math.sqrt(
            pow(radius, 2) -
            pow(math.cos(abs(jaws[0] * 10) * 0.1 / ssd) * radius, 2)))
            * abs(jaws[0] * 10) * 0.1 / ssd +
            math.cos(abs(jaws[0] * 10) * 0.1 / ssd) *
            radius) * (-1)
    if jaws[0] > 0:
        new_JAWS[0] = (-(cil - math.sqrt(
            pow(radius, 2) -
            pow(math.cos(abs(jaws[0] * 10) * 0.1 / ssd) * radius, 2)))
            * abs(jaws[0] * 10) * 0.1 / ssd +
            math.cos(abs(jaws[0] * 10) * 0.1 / ssd) *
            radius) * (-1)
    if jaws[1] >= 0:
        new_JAWS[1] = (cil + math.sqrt(
            pow(radius, 2) -
            pow(math.cos(abs(jaws[1] * 10) * 0.1 / ssd) * radius, 2))
        ) * abs(jaws[1] * 10) * 0.1 / ssd + math.cos(
            abs(jaws[1] * 10) * 0.1 / ssd) * radius
    if jaws[1] < 0:
        new_JAWS[1] = -(cil - math.sqrt(
            pow(radius, 2) -
            pow(math.cos(abs(jaws[1] * 10) * 0.1 / ssd) * radius, 2))
        ) * abs(jaws[1] * 10) * 0.1 / ssd + math.cos(
            abs(jaws[1] * 10) * 0.1 / ssd) * radius

    return new_JAWS


def create_beam_config(patient, num, jaws, leafes):

    leafes = np.reshape(leafes, (2, 80))
    leafes /= 10
    jaws /= 10

    leafes_lines = [
        f"{np.round(leafes[0][i],4)}, {np.round(leafes[1][i],4)}, 1\n" for i in range(80)]
    jaws_lines = f"{np.round(jaws[0],4)}, {np.round(jaws[1],4)}, 2\n"

    with open(f"/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/BEAM_MR-Linac/beam_config_{patient}_{num}.txt", "w+") as fout:
        fout.writelines(leafes_lines)
        fout.writelines(jaws_lines)

    leafes_egsinp = calculate_new_mlc(leafes)
    jaws_egsinp = calculate_new_jaw(jaws)

    leafes_lines = [
        f"{np.round(leafes_egsinp[0][i],4)}, {np.round(leafes_egsinp[1][i],4)}, 1\n" for i in range(80)]
    jaws_lines = f"{np.round(jaws_egsinp[0],4)}, {np.round(jaws_egsinp[1],4)}, 2\n"

    template = open(
        "/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/BEAM_MR-Linac/template.egsinp",
        "r")
    lines = template.readlines()
    lines.insert(197, jaws_lines)
    lines.pop(198)

    lines.pop(183)

    i = 0
    for line in leafes_lines:
        lines.insert(183+i, line)
        i += 1
    with open(
            f"/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/BEAM_MR-Linac/beam_config_{patient}_{num}.egsinp",
            "w+") as out:
        out.writelines(lines)
        out.close()

    return f"beam_config_{patient}_{num}"


def setup_plan_calculation(patient, plan_file):
    '''
    creates the needed files for the DOSXYZnrc simulation

    Parameters:
    patient:  patient name
    plan_file:  path to the plan file

    Returns:
    returns: list of all angles, iso_centers and configure files
    '''

    jaws, leafes, angles, iso_centers = extract_plan_infos(plan_file)

    config_files = []
    for config in range(len(angles)):
        config_files.append(create_beam_config(patient, config, jaws[config], leafes[config]))

    return list(zip(angles, iso_centers, config_files))


if __name__ == "__main__":

    plan = True
    patients = ["p0"]
    for patient in patients:

        num_hist = 10000000
        pj = int(num_hist/2000000)
        if pj <= 1:
            pj = 2

        if plan:

            plan_file = os.path.join(PLAN_FILES_PATH, f"{patient}_plan.dcm")
            dose_file = os.path.join(DOSE_FILES_PATH, f"{patient}_dose.dcm")

            infos = setup_plan_calculation(patient, plan_file)

            print(f"Creating {len(infos)} Jobs for all Segments in Plan {patient}.")

            for angle, iso_center, config_file in infos:

                create_entire_job(n=num_hist,
                                  gantry=angle + 270,
                                  par_jobs=pj,
                                  ppn=1,
                                  nodes=pj,
                                  beam_config=config_file,
                                  patient=patient,
                                  iso_center=iso_center)

        else:

            beam_configs = [f"beam_config_{patient}_10x10"]
            iso_center = [30, 30, 30]
            angle = 0

            for beam in beam_configs:

                create_entire_job(n=num_hist,
                                  gantry=angle + 270,
                                  par_jobs=pj,
                                  ppn=1,
                                  nodes=pj,
                                  beam_config=beam,
                                  patient=patient,
                                  iso_center=iso_center)
