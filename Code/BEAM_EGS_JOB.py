#%%

import paramiko
import os
import time
from pydicom import dcmread
import random
import re
import numpy as np

hostname = "login1.nemo.uni-freiburg.de"
username = "tu_zxoys08"
password = "Derzauberkoenig1!"

#log into server
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname=hostname, username=username, password=password)

#get folder of all DCM files
dcm_path = "/Users/simongutwein/localfolder/EGSnrc/egs_home/dosxyznrc/p_pat"
dose_file = "/Users/simongutwein/localfolder/EGSnrc/egs_home/dosxyznrc/MbaseMRL_Dose.dcm"
server_path = "/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc/p_pat"
server_dosxyznrc_path = "/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc"
dosxyznrc_path = "/".join(dcm_path.split("/")[:-1])
egsinp_beam = "MR-Linac_model_10x10_0x0"
n_histories = 1000000
IPARALLEL = 20
ANGLE = 315.00
jobs_path = "/Users/simongutwein/localfolder/EGSnrc/jobs"

all_files = os.listdir(dosxyznrc_path)
for line in range(len(all_files)):
    all_files[line] += "\n"


#remove files which start with ._ form list 
used_files = [file_ for file_ in sorted(os.listdir(
    dcm_path)) if not file_.startswith('._') and not file_.startswith(".") and not file_ == "listfile.txt"]
dose_file = dcmread(dose_file)


with open(dcm_path + '/listfile.txt', 'w+') as fout:
    for filename in used_files:
        fout.write(server_path + '/' + filename + '\n')


with open(dosxyznrc_path+ '/ctcreate_file.txt', 'w+') as fout:
    
    image_position = dose_file.ImagePositionPatient
    dose_dimensions = dose_file.pixel_array.shape
    pixel_spacing = dose_file.PixelSpacing

    xlower = image_position[0]/10-pixel_spacing[0]/20
    xupper = image_position[0]/10-pixel_spacing[0] / 20+dose_dimensions[2]*pixel_spacing[0]/10
    ylower = image_position[1]/10-pixel_spacing[1]/20
    yupper = image_position[1]/10-pixel_spacing[1] / 20+dose_dimensions[1]*pixel_spacing[1]/10
    zlower = image_position[2]/10-pixel_spacing[0]/20
    zupper = image_position[2]/10-pixel_spacing[0] / 20+dose_dimensions[0]*pixel_spacing[0]/10

    fout.write("DICOM \n")
    fout.write(server_path + "/listfile.txt\n")
    fout.write(str("%.4f" % xlower) + ", " + str("%.4f" % xupper) + ", " + str("%.4f" % ylower) +
            ", " + str("%.4f" % yupper) + ", " + str("%.4f" % zlower) + ", " + str("%.4f" % zupper) + "\n")
    fout.write(str(pixel_spacing[0]/10) + ", " +
            str(pixel_spacing[1]/10) + ", " + str(pixel_spacing[0]/10) + "\n")
    fout.write("0, 0 \n")

if os.path.isfile(dosxyznrc_path + "/listfile.txt.egsphant"):
    os.remove(dosxyznrc_path + "/listfile.txt.egsphant")

_, stdout, _ = client.exec_command(
    'cd EGSnrc/egs_home/dosxyznrc; ls; ctcreate ctcreate_file.txt -p 700icru')
for i in stdout:
    if i.strip():
        print(i)

with open(dosxyznrc_path + "/listfile.txt.egsphant", "r") as fout:
    lines = fout.readlines()
    x = np.array(lines[7].split()).astype("float")
    pos_x = np.round(np.take(x, x.size//2),2)
    y = np.array(lines[8].split()).astype("float")
    pos_y = np.round(np.take(y, y.size//2),2)
    z = np.array(lines[9].split()).astype("float")
    pos_z = np.round(np.take(z, z.size//2),2)

#%%

with open(dosxyznrc_path + '/phantom_file.egsinp', 'w+') as fout:

    fout.write('CT Phantom from listfile.txt.egsphant\n')
    fout.write('0\n')
    fout.write('$EGS_HOME/dosxyznrc/listfile.txt.egsphant\n')
    fout.write('0.7, 0.01, 0\n')
    fout.write('1, 0, 0,\n')
    fout.write('2, 9, ' + str(pos_x) + ', ' + str(pos_y) + ', ' +
               str(pos_z) + ', 90.00, ' + format(ANGLE,'.2f') + ', 30, 270, 1, 40\n')
    fout.write('2, 0, 2, 0, 0, 0, 0, 0\n')
    fout.write('BEAM_MR-Linac,' +  egsinp_beam + ',521ELEKTA\n')
    fout.write( str(int(n_histories/IPARALLEL)) + ", 0, 500," +  str(random.randint(0,10000)) + "," + str(random.randint(0,10000)) + ", 100.0, 1, 4, 0, 1, 2.0, 0, 0, 0, 40, 0, 0\n")
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

with open(dosxyznrc_path + '/phantom_file.egsinp', 'r') as fout:
    txt = fout.read()


split_txt = txt.split("\n")
position_angle_line = split_txt[5].split(",")
position_angle_line[6] = " " + format(ANGLE,'.2f')
position_angle_line[2] = " " + str(pos_x)
position_angle_line[3] = " " + str(pos_y)
position_angle_line[4] = " " + str(pos_z)
split_txt[5] = ",".join(position_angle_line)

parallel_line = split_txt[8].split(",")
parallel_line[12] = " " + str(IPARALLEL)
parallel_line[7] = " 0"

for i in range(IPARALLEL):
    with open(dosxyznrc_path + '/phantom_file_w' + str(i+1) + '.egsinp', 'w+') as fout:
        parallel_line[13] = " " + str(i+1)
        parallel_line[3], parallel_line[4] = str(random.randint(
            0, 10000)), str(random.randint(0, 10000))
        split_txt[8] = ','.join(parallel_line)

        file_text = "\n".join(split_txt)
        fout.write(file_text)

        
with open(jobs_path + '/job_' + dcm_path.split("/")[-1] + '.sh', 'w+') as fout:

    fout.write('#!/bin/bash\n')
    fout.write("#MSUB -l nodes=4:ppn=20\n")
    fout.write('#MSUB -l walltime=4:00:00:00\n')
    fout.write('#MSUB -l mem=64gb\n')
    fout.write('#MSUB -N EGSnrc\n')
    fout.write("#MSUB -o /home/tu/tu_tu/tu_zxoys08/EGSnrc/jobs\n")
    command = []

    for i in range(IPARALLEL):
        command.append("dosxyznrc -i phantom_file_w" + str(i+1) + ".egsinp -p 700icru")
    command = " & ".join(command) + " & wait\n\n\n"
    command += "dosxyznrc -i phantom_file.egsinp -p 700icru"
    
    fout.write(command)

client.exec_command('cd EGSnrc/jobs; chmod +x ./job.sh')
_, stdout, _ =  client.exec_command('msub ./EGSnrc/jobs/job_p_pat.sh')

for string in stdout:
    if string.strip():
        job_id = string.strip()
print("Job-ID: " + job_id)
# _, stdout, _ = client.exec_command('showq -u tu_zxoys08')

# text = []
# for string in stdout:
#     if string.strip():
#         text.append(string)

# for line in text:
#     if job_id in line and "Running" in line:
#         finished = False

# with open(dosxyznrc_path + "/" + job_id + ".txt", "w+") as fout:
#     fout.write("".join(all_files))


# if __name__ == '__main__': 

#     CT_folder = input("Ender the path to the CT Files Folder:  ")
#     dosxyznrc_path = input("Enter the path to dosxyznrc Folder:  ")
#     dose_file = input("Enter the path to the dose file:  ")
#     field_file = input("Enter the name of the field file:  ")
#     n_histories = input("Enter Number of histories:  ")
#     num_angles = input("Enter the number of different Angles:  ")
#     angles = []
#     for i in range(num_angles):
#        angles.append(input("Enter Gantry Angle:  "))
#     IPARALLEL = input("Enter # of parallel jobs:  ")
    



