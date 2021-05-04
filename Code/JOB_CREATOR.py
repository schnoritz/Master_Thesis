#   Skript welches aus gegebenen CT Bildern einen Jobs in EGSnrc an den BW-HPC Nemo CLuster sendet.
#   Individuelle Eingabe von CT-Bildern, Gantry-Winkel, Anzahl der Simulierten Teilchen und des genutzten Strahls

import os
import paramiko
import random
import numpy as np
from pydicom import dcmread
from glob import glob

def server_login():
    """logs into the BW-HPC server

    Returns:
        client: returns paramiko client 
    """
    hostname = "login1.nemo.uni-freiburg.de"
    username = "tu_zxoys08"
    password = "Derzauberkoenig1!"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=hostname, username=username, password=password)
    
    return client


def create_listfile(path):
    """creates a listfile in the directory of the CT images
    Args:
        local_dcm_path (str):       path to the CT images wanted to be used as phantom data
        server_dcm_path (str):      path to the same directory but on BW-HPC
    """
    used_files = [file_ for file_ in sorted(os.listdir(path)) if not file_.startswith('._') \
        and not file_.startswith(".") and not file_ == "listfile.txt"]

    with open(path + '/listfile.txt', 'w+') as fout:
        for filename in used_files:
            fout.write(path + filename + '\n')


def create_ctcreate_file(path, dose_file_path, dcm_folder):
    """creates the file needed for the ctcreate command in EGSnrc

    Args:
        local_dosxyznrc_path (str):     path to the local dosxyznrc directory
        dose_file_path (str):           path to the local dose file used for EGSnrc
        server_dcm_path (str):          path to the server directory of the used CT images
    """
    dose_file_dat = dcmread(dose_file_path)

    with open(path + 'ctcreate_file.txt', 'w+') as fout:
    
        image_position = np.array(dose_file_dat.ImagePositionPatient)/10
        dose_dimensions = dose_file_dat.pixel_array.shape
        pixel_spacing = np.array(dose_file_dat.PixelSpacing)/10

        xlower = image_position[0]-pixel_spacing[0]/2
        ylower = image_position[1]-pixel_spacing[1]/2
        zlower = image_position[2]-pixel_spacing[0]/2

        xupper = image_position[0]-pixel_spacing[0] / 2+dose_dimensions[2]*pixel_spacing[0]
        yupper = image_position[1]-pixel_spacing[1] / 2+dose_dimensions[1]*pixel_spacing[1]
        zupper = image_position[2]-pixel_spacing[0] / 2+dose_dimensions[0]*pixel_spacing[0]

        fout.write("DICOM \n")
        fout.write(path + dcm_folder + "listfile.txt\n")
        fout.write(str("%.4f" % xlower) + ", " + str("%.4f" % xupper) + \
            ", " + str("%.4f" % ylower) + ", " + str("%.4f" % yupper) + \
            ", " + str("%.4f" % zlower) + ", " + str("%.4f" % zupper) + "\n")
        
        fout.write(str(pixel_spacing[0]) + ", " +
                str(pixel_spacing[1]) + ", " + str(pixel_spacing[0]) + "\n")
        
        fout.write("0, 0 \n")


def execute_ct_create(client, path):
    """executes the ctcreate command on the given server

    Args:
        local_dosxyznrc_path ([type]): path to the local drectory of dosxyznrc
    """    

    if os.path.isfile(path + "listfile.txt.egsphant"):
        os.remove(path + "listfile.txt.egsphant")

    _, stdout, _ = client.exec_command('cd EGSnrc/egs_home/dosxyznrc; \
         ctcreate ctcreate_file.txt -p 700icru')

    for line in stdout:
        continue

    print("Finished: .egsphant file was created.")


def get_iso_position(path):
    """reads out iso center position from egsphant file

    Args:
        local_dosxyznrc_path (str):         path to the local dosxyznrc folder

    Returns:
        str, str, str: returns the iso center position as strings to be used in .egsinp files
    """  

    with open(path + "listfile.txt.egsphant", "r") as fout:

        for i in range(7):
            fout.readline()
    
        x = np.array(fout.readline().split()).astype("float")
        y = np.array(fout.readline().split()).astype("float")
        z = np.array(fout.readline().split()).astype("float")

        pos_x = np.round(np.take(x, x.size//2), 2)
        pos_y = np.round(np.take(y, y.size//2), 2)
        pos_z = np.round(np.take(z, z.size//2), 2)

        return str(pos_x), str(pos_y), str(pos_z)


def create_egsinp_file(path, pos_x, pos_y, pos_z, angle, beam_config, n_histories, iparallel, filename):
    """creates the needed .egsinp file which is needed to combine the created .pardose files in 
       parallel calculation. 

    Args:
        path (str):     path to the local dosxyznrc directory
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

    with open(path + filename + '.egsinp', 'w+') as fout:

        fout.write('CT Phantom from listfile.txt.egsphant\n')
        fout.write('0\n')
        fout.write('$EGS_HOME/dosxyznrc/listfile.txt.egsphant\n')
        fout.write('0.7, 0.01, 0\n')
        fout.write('1, 0, 0,\n')

        fout.write('2, 9, ' +  
            pos_x + ', ' +  
            pos_y + ', ' + 
            pos_z + ', 90.00, ' + format(angle, '.2f') + ', 30, 270, 1, 40\n')

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
        fout.write(' Magnetic Field= 0.0, 0.0, 1.5\n')
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


def create_parallel_files(egsinp_lines, path, pos_x, pos_y, pos_z, angle, beam_config, n_histories, iparallel, filename):
    """creates the needed .egsinp files which are needed to calculate the .pardose files
       which will be recombined to .3ddose file
    Args:
        egsinp_lines (list):            given list of text from .egsinp file
        path (str):                     path to the dosxyznrc directory
        pos_x (str):                    iso position of the phamtom in x direction
        pos_y (str):                    iso position of the phamtom in y direction
        pos_z (str):                    iso position of the phamtom in z direction
        angle (float):                  desired angle of gantry for simulation
        beam_config (str):              name of the wanted beam configuration placed inside "BEAM_MR-Linac" folder
        n_histories (int):              number of simulated histories
        iparallel (int):                number of parallel jobs
     """

    position_angle_line = egsinp_lines[5].split(",")
    position_angle_line[6] = " " + format(angle, '.2f')
    position_angle_line[2] = " " + pos_x
    position_angle_line[3] = " " + pos_y
    position_angle_line[4] = " " + pos_z
    egsinp_lines[5] = ",".join(position_angle_line)

    parallel_line = egsinp_lines[8].split(",")
    parallel_line[12] = " " + str(iparallel)
    parallel_line[7] = " 0"

    files = glob(path + "*" + str(angle) +"_w*.*")
    if files:
        for name in files:
            if not "egsrun" in name:
                os.remove(name)

    for i in range(iparallel):
        with open(path + filename + '_w' + str(i+1) + '.egsinp', 'w+') as fout:
            
            parallel_line[13] = " " + str(i+1)
            parallel_line[3] = " " + str(random.randint(0, 10000)) 
            parallel_line[4] = " " + str(random.randint(0, 10000))
            egsinp_lines[8] = ','.join(parallel_line)

            file_text = "".join(egsinp_lines)
            fout.write(file_text)


def create_job_file(jobs_path, iparallel, nodes, ppn, filename, gantry):
    """creates the job file which can be executed for parallel simulation

    Args:
        jobs_path (str):       path to the directory where jobs are stored
        local_dcm_path (str):  path to the directory where CT are located
        iparallel (int):       number of desired parallel simulations
    """

    with open(jobs_path + '/job.sh', 'w+') as fout:

        fout.write('#!/bin/bash\n')
        fout.write("#MSUB -l nodes=" + str(nodes) +":ppn=" + str(ppn) + "\n")
        fout.write('#MSUB -l walltime=4:00:00:00\n')
        fout.write('#MSUB -l mem=64gb\n')
        fout.write('#MSUB -N EGSnrc\n')
        fout.write("#MSUB -o /home/tu/tu_tu/tu_zxoys08/EGSnrc/jobs\n")
        command = []

        for i in range(iparallel):

            command.append("dosxyznrc -i " + filename + "_w" +
                        str(i+1) + ".egsinp -p 700icru")

        command = " & ".join(command) + " & wait\n\n"
        command += "dosxyznrc -i " + filename + ".egsinp" + " -p 700icru & wait\n\n"
        
        command += "python3 clean_dosxyznrc_folder.py " + gantry

        fout.write(command)


def execute_job_file(client):
    """executes the created job on the BW-HPC cluster 
    Args:
        client (client): paramiko client where the command is executed
    """
    client.exec_command('cd EGSnrc/jobs; chmod +x ./job.sh')
    _, stdout, _ = client.exec_command('msub ./EGSnrc/jobs/job.sh')

    for string in stdout:
        if string.strip():
            job_id = string.strip()
    print("Job-ID: " + job_id)


def create_entire_job(n, gantry, par_jobs, ppn, nodes, beam_config, patient):

    dcm_folder = patient #select folder located in "dosxyznrc" folder
    beam_info = beam_config.split("_")[-1]

    target_filename = dcm_folder[:-1] + "_" + str(int(gantry - 270)) + "_" + beam_info

    client = server_login()
    dosxyznrc_path = "/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc/"
    dose_file_path = dosxyznrc_path + "MbaseMRL_Dose.dcm"
    jobs_path = "/home/tu/tu_tu/tu_zxoys08/EGSnrc/jobs"

    create_listfile(dosxyznrc_path + dcm_folder)
    create_ctcreate_file(dosxyznrc_path, dose_file_path, dcm_folder)
    execute_ct_create(client, dosxyznrc_path)
    iso_x, iso_y, iso_z = get_iso_position(dosxyznrc_path)
    lines = create_egsinp_file(dosxyznrc_path, iso_x, iso_y, iso_z, gantry, beam_config, n, par_jobs, target_filename)
    create_parallel_files(lines, dosxyznrc_path, iso_x, iso_y, iso_z, gantry, beam_config, n, par_jobs, target_filename)
    create_job_file(jobs_path, par_jobs, nodes, ppn,
                    target_filename, str(int(gantry - 270)))
    execute_job_file(client)

if __name__ == "__main__":
    nums = 8
    beam = "MR-Linac_model_3x3"

    for angle in np.linspace(0,360, nums, endpoint=False):

        create_entire_job(n=10000, gantry=angle + 270, par_jobs=10, ppn=20, nodes=4, beam_config=beam, patient="p/")
